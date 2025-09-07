#!/usr/bin/env python3
"""
Batch Processor for ChipForge Validator
Handles batch evaluation and processing logic
"""

import asyncio
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handles batch processing logic for the validator"""
    
    def __init__(self, api_client, state, emission_manager, weight_manager):
        self.api_client = api_client
        self.state = state
        self.emission_manager = emission_manager
        self.weight_manager = weight_manager
        
        # Directories
        self.base_dir = Path('./validator_data')
        self.submissions_dir = self.base_dir / 'submissions'
        self.submissions_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_hotkeys_from_filenames(self, batch_id: str, successful_submissions: dict) -> dict:
        """Extract hotkeys from downloaded filenames"""
        submission_hotkeys = {}
        batch_dir = self.submissions_dir / batch_id
        
        try:
            # Look for files in the batch directory
            if batch_dir.exists():
                for file_path in batch_dir.glob("*.zip"):
                    filename = file_path.name
                    # Parse filename: challenge_id__hotkey__submission_id__attempt__processing.zip
                    parts = filename.replace('.zip', '').split('__')
                    if len(parts) >= 3:
                        hotkey = parts[1]
                        submission_id = parts[2]
                        
                        if submission_id in successful_submissions:
                            submission_hotkeys[submission_id] = hotkey
                            logger.info(f"Extracted hotkey from filename: {submission_id} -> {hotkey[:12]}...")
        
        except Exception as e:
            logger.error(f"Error extracting hotkeys from filenames: {e}")
        
        return submission_hotkeys
    
    def calculate_weights_from_hotkeys(self, evaluations: Dict[str, Dict], submission_hotkeys: Dict[str, str]) -> Dict[str, float]:
        """Calculate weights based on evaluation scores and map to miner hotkeys"""
        if not evaluations or not submission_hotkeys:
            return {}
        
        # Create hotkey to best score mapping
        hotkey_scores = {}
        
        for submission_id, eval_data in evaluations.items():
            miner_hotkey = submission_hotkeys.get(submission_id)
            if miner_hotkey:
                score = eval_data['overall_score']
                # Keep the best score for each miner
                if miner_hotkey not in hotkey_scores or score > hotkey_scores[miner_hotkey]:
                    hotkey_scores[miner_hotkey] = score
        
        if not hotkey_scores:
            return {}
        
        # Sort miners by best score
        sorted_miners = sorted(hotkey_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Winner-takes-all approach
        weights = {}
        for i, (hotkey, score) in enumerate(sorted_miners):
            if i == 0:  # Highest score gets weight 1
                weights[hotkey] = 1.0
            else:  # All others get weight 0
                weights[hotkey] = 0.0
        
        winner_hotkey, winner_score = sorted_miners[0]
        logger.info(f"Calculated weights: winner={winner_hotkey[:12]}... (score: {winner_score})")
        logger.info(f"Total miners evaluated: {len(weights)}")
        
        return weights
    
    async def process_batch(self, challenge_id: str, batch: Dict) -> bool:
        """Process a complete batch evaluation with challenge-wide best score tracking"""
        batch_id = batch['batch_id']
        
        try:
            logger.info(f"Starting batch processing for {batch_id}")
            self.state.evaluation_in_progress = True
            self.state.current_batch_id = batch_id
            self.state.save_state()
            
            # Download submissions
            logger.info(f"Downloading submissions for batch {batch_id}")
            downloaded_submissions = await self.api_client.download_batch_submissions(challenge_id, batch)
            logger.info(f"Downloaded {len(downloaded_submissions)} submissions")

            if not downloaded_submissions:
                logger.warning(f"No submissions downloaded for batch {batch_id}")
                return False
            
            # Evaluate with EDA server
            logger.info(f"Evaluating {len(downloaded_submissions)} submissions with EDA server")
            evaluations = await self.api_client.evaluate_submissions_with_eda_server(downloaded_submissions)
            if not evaluations:
                logger.error(f"No evaluations received from EDA server")
                return False
            
            logger.info(f"Received {len(evaluations)} evaluations from EDA server")
            
            # Submit evaluations to challenge server
            logger.info(f"Submitting {len(evaluations)} evaluations to challenge server")
            submission_results = await self.api_client.submit_all_evaluations(challenge_id, evaluations)
            successful_submissions = {k: v for k, v in evaluations.items() if submission_results.get(k, False)}
            
            if not successful_submissions:
                logger.error("No evaluations were successfully submitted")
                return False
            
            logger.info(f"Successfully submitted {len(successful_submissions)} evaluations")
            
            # Extract hotkeys from filenames
            submission_hotkeys = self.extract_hotkeys_from_filenames(batch_id, successful_submissions)
            
            current_best_score = self.state.current_challenge_best[1] if hasattr(self.state, 'current_challenge_best') else 0.0
            current_best_hotkey = self.state.current_challenge_best[0] if hasattr(self.state, 'current_challenge_best') else None
            
            logger.info(f"Current challenge best: {current_best_hotkey[:12] if current_best_hotkey else 'None'}... -> {current_best_score}")
            
            # Find if any submission in this batch beats the challenge-wide best
            new_champion = None
            new_best_score = current_best_score
            
            for submission_id, eval_data in successful_submissions.items():
                hotkey = submission_hotkeys.get(submission_id)
                overall_score = eval_data.get('overall_score', 0)
                
                if hotkey and overall_score > new_best_score:
                    new_best_score = overall_score
                    new_champion = hotkey
                    logger.info(f"New challenge champion found: {hotkey[:12]}... -> {overall_score} (beats previous: {current_best_score})")

            # Update challenge-wide best if we found a new champion
            if new_champion:
                self.state.update_best_miner(challenge_id, new_champion, new_best_score)
    
                # Update current challenge best (separate tracking)
                self.state.current_challenge_best = (new_champion, new_best_score)
                
                # Update emission manager with new winner
                self.emission_manager.update_winner(new_champion, challenge_active=True)
                
                # Check if new champion is on subnet and get UID
                try:
                    uid = self.weight_manager.get_hotkey_uid(new_champion)
                    if uid is not None:
                        # Set weights: 1.0 for new champion, 0.0 for others
                        weights = {new_champion: 1.0}
                        logger.info(f"Setting NEW CHAMPION weights: {new_champion[:12]}... (UID {uid}) = 1.0, score: {new_best_score}")
                        
                        weight_success = self.weight_manager.set_weights(weights)
                        if not weight_success:
                            logger.error("Failed to set weights, burning emissions")
                            self.weight_manager.set_burn_weights()
                    else:
                        logger.warning(f"New champion {new_champion[:12]}... not found on subnet, burning emissions")
                        self.weight_manager.set_burn_weights()
                except Exception as e:
                    logger.error(f"Error getting UID for new champion: {e}, burning emissions")
                    self.weight_manager.set_burn_weights()
                    
            else:
                # No new champion found - check emission management policy
                logger.info(f"No submissions beat challenge best of {current_best_score}")
                
                # Get reward hotkey from emission manager (considers winner reward period)
                reward_hotkey = self.emission_manager.get_reward_hotkey(current_best_hotkey)
                should_burn = self.emission_manager.should_burn_emissions(current_best_score)
                
                if reward_hotkey and not should_burn:
                    # Continue rewarding current winner/champion
                    try:
                        uid = self.weight_manager.get_hotkey_uid(reward_hotkey)
                        if uid is not None:
                            weights = {reward_hotkey: 1.0}
                            logger.info(f"Challenge active, winner {reward_hotkey[:12]}... taking reward until next good submission")
                            
                            weight_success = self.weight_manager.set_weights(weights)
                            if not weight_success:
                                logger.error("Failed to set weights, burning emissions")
                                self.weight_manager.set_burn_weights()
                        else:
                            logger.warning(f"Reward target {reward_hotkey[:12]}... not found on subnet, burning emissions")
                            self.weight_manager.set_burn_weights()
                    except Exception as e:
                        logger.error(f"Error getting UID for reward target: {e}, burning emissions")
                        self.weight_manager.set_burn_weights()
                else:
                    # Should burn emissions
                    if self.emission_manager.current_winner:
                        logger.info("Challenge active, submissions found but winner reward period expired - burning emissions")
                    else:
                        logger.info("Challenge active, submissions checking but no qualified winner - burning emissions")
                    self.weight_manager.set_burn_weights()
            
            # Mark batch as processed
            self.state.evaluated_batches.add(batch_id)
            self.state.current_batch_id = None
            self.state.evaluation_in_progress = False
            self.state.save_state()
            
            logger.info(f"Successfully processed batch {batch_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            self.state.evaluation_in_progress = False
            self.state.save_state()
            return False