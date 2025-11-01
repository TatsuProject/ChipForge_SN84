#!/usr/bin/env python3
# neurons/validator.py
"""
ChipForge Subnet Validator (Refactored)
Main validator entry point with modular architecture
"""
import bittensor as bt
from chipforge.protocol import SimpleMessage

import asyncio
import aiohttp
import logging
import traceback
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Import validator utilities
from validator_utils import (
    EmissionManager,
    ValidatorState,
    BatchProcessor,
    WeightManager,
    APIClient,
    MinerCommunications,
    setup_validator_logging
)

# Configure logging with daily rotation
setup_validator_logging(log_level="INFO")
logger = logging.getLogger(__name__)


class ChipForgeValidator:
    """ChipForge Subnet Validator with modular architecture"""
    
    def __init__(self, config):
        self.config = config
        self.wallet = bt.wallet(config=config)
        self.subtensor = bt.subtensor(config=config)
        self.metagraph = self.subtensor.metagraph(config.netuid)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        
        # Initialize components
        self.state = ValidatorState()
        self.emission_manager = EmissionManager()
        self.weight_manager = WeightManager(
            self.wallet, self.subtensor, self.metagraph, self.config
        )
        
        # Initialize subnet
        self.emission_manager.initialize_subnet()
        
        # Session for HTTP requests
        self.session = None
        
        # API client and other components will be initialized after session creation
        self.api_client = None
        self.batch_processor = None
        self.miner_comms = None
        
        # Batch timing
        self.next_batch_check = datetime.now(timezone.utc)
        self.consecutive_errors = 0
        
        logger.info(f"ChipForge Validator initialized")
        logger.info(f"Validator hotkey: {self.wallet.hotkey.ss58_address}")
    
    def initialize_components(self):
        """Initialize components that require the HTTP session"""
        self.api_client = APIClient(self.config, self.wallet, self.session)
        self.batch_processor = BatchProcessor(
            self.api_client, self.state, self.emission_manager, self.weight_manager
        )
        self.miner_comms = MinerCommunications(self.dendrite, self.metagraph)
    
    async def recover_from_crash(self):
        """Enhanced crash recovery with current challenge focus"""
        try:
            logger.info("Starting enhanced crash recovery...")
            
            # Check if there's a currently active challenge
            challenge = await self.api_client.get_active_challenge()
            current_challenge_active = challenge not in [None, {"status": "no_active_challenge"}, {"status": "None"}]
            
            if current_challenge_active:
                challenge_id = challenge['challenge_id']
                
                # Check if it's the same challenge we were working on
                if self.state.last_challenge_id == challenge_id:
                    logger.info(f"Continuing with same challenge {challenge_id}")
                    # Keep current challenge best as-is
                else:
                    logger.info(f"New challenge {challenge_id} detected during crash recovery")
                    # Reset for new challenge
                    self.state.current_challenge_best = (None, 0.0)
                    self.state.last_challenge_id = challenge_id
                    self.state.save_state()
                
                remaining_time = await self.api_client.get_challenge_remaining_time(challenge_id)
                if remaining_time and remaining_time > 0:
                    logger.info(f"Active challenge {challenge_id}: {remaining_time/3600:.1f}h remaining")
                    self.next_batch_check = datetime.now(timezone.utc) + timedelta(seconds=10)
                else:
                    logger.info(f"Challenge {challenge_id} found but expired")
                    current_challenge_active = False
            else:
                logger.info("No active challenge found")
                # Reset current challenge since no challenge is active
                self.state.current_challenge_best = (None, 0.0)
                self.state.last_challenge_id = None
                self.state.save_state()
                
            # Recover emission state - pass current challenge winner if any
            current_challenge_winner = self.state.current_challenge_best[0] if hasattr(self.state, 'current_challenge_best') else None
            current_challenge_score = self.state.current_challenge_best[1] if hasattr(self.state, 'current_challenge_best') else 0.0
            
            self.emission_manager.recover_emission_state_after_crash(
                current_challenge_active, 
                (current_challenge_winner, current_challenge_score) if current_challenge_winner else None
            )
            
            # Log the recovery status
            if not current_challenge_active and self.emission_manager.current_winner:
                remaining_time = None
                if self.emission_manager.winner_reward_start_time:
                    end_time = self.emission_manager.winner_reward_start_time + timedelta(hours=self.emission_manager.total_hours_for_winner_reward)
                    remaining_seconds = (end_time - datetime.now(timezone.utc)).total_seconds()
                    if remaining_seconds > 0:
                        remaining_time = remaining_seconds / 3600
                
                if remaining_time and remaining_time > 0:
                    logger.info(f"Challenge has expired, decided winner {self.emission_manager.current_winner[:12]}... is taking reward, remaining time {remaining_time:.1f}h")
                else:
                    logger.info("Challenge has expired, winner reward period also expired")
            
            # Restore winner state if we have current challenge winner
            if hasattr(self.state, 'current_challenge_best') and self.state.current_challenge_best[0]:
                winner_hotkey, winner_score = self.state.current_challenge_best
                
                # Check if this winner should still be getting rewards
                reward_hotkey = self.emission_manager.get_reward_hotkey(winner_hotkey)
                if reward_hotkey:
                    # Try to set weights for the winner if they're still on subnet
                    uid = self.weight_manager.get_hotkey_uid(winner_hotkey)
                    if uid is not None:
                        weights = {winner_hotkey: 1.0}
                        logger.info(f"Crash recovery: Restoring weights for current challenge winner {winner_hotkey[:12]}... (UID {uid})")
                    else:
                        logger.warning(f"Crash recovery: Current challenge winner {winner_hotkey[:12]}... not found on subnet")
            
            # If no current challenge winner but we have historical winners, check emission manager
            elif self.emission_manager.current_winner:
                winner_hotkey = self.emission_manager.current_winner
                
                # Check if this historical winner should still be getting rewards
                reward_hotkey = self.emission_manager.get_reward_hotkey(winner_hotkey)
                if reward_hotkey:
                    uid = self.weight_manager.get_hotkey_uid(winner_hotkey)
                    if uid is not None:
                        weights = {winner_hotkey: 1.0}
                        logger.info(f"Crash recovery: Restoring weights for emission manager winner {winner_hotkey[:12]}... (UID {uid})")
                    else:
                        logger.warning(f"Crash recovery: Emission manager winner {winner_hotkey[:12]}... not found on subnet")
            
            # Set appropriate next check time
            if not current_challenge_active:
                self.next_batch_check = datetime.now(timezone.utc) + timedelta(seconds=30)
            
            logger.info("Enhanced crash recovery completed")
                        
        except Exception as e:
            logger.error(f"Error during enhanced crash recovery: {e}")
            logger.error(traceback.format_exc())

    async def check_challenge_expiration_update(self, challenge_id: str):
        """Check if challenge expiration time has been updated by admin"""
        try:
            # Get current challenge info from API
            challenge = await self.api_client.get_active_challenge()
            
            if challenge in [None, {"status": "no_active_challenge"}, {"status": "None"}] or challenge['challenge_id'] != challenge_id:
                return  # Challenge no longer active or different challenge
            
            if 'expires_at' not in challenge:
                return  # No expiration time in response
            
            # Parse new expiration time
            new_expires_at = datetime.fromisoformat(challenge['expires_at'].replace('Z', '+00:00'))
            
            # Check if we have a stored expiration time and compare
            if hasattr(self.state, 'current_challenge_expires_at') and self.state.current_challenge_expires_at:
                stored_expires_at = self.state.current_challenge_expires_at
                
                # Check if expiration time has changed
                if new_expires_at != stored_expires_at:
                    time_diff = new_expires_at - stored_expires_at
                    hours_diff = time_diff.total_seconds() / 3600
                    
                    if hours_diff > 0:
                        logger.info(f"Challenge {challenge_id} deadline extended by {hours_diff:.2f} hours")
                        logger.info(f"Old expiration: {stored_expires_at}")
                        logger.info(f"New expiration: {new_expires_at}")
                    else:
                        logger.info(f"Challenge {challenge_id} deadline shortened by {abs(hours_diff):.2f} hours")
                    
                    # Update stored expiration time
                    self.state.current_challenge_expires_at = new_expires_at
                    self.state.save_state()
                    
                    logger.info(f"Updated challenge {challenge_id} expiration time in validator state")
            
        except Exception as e:
            logger.error(f"Error checking challenge expiration update: {e}")
            # Don't raise exception - this is a non-critical update check

    
    async def run_evaluation_cycle(self):
        """Main evaluation cycle with dynamic scheduling"""
        try:
            now = datetime.now(timezone.utc)
            
            # Sync metagraph to see current miners
            self.metagraph.sync(subtensor=self.subtensor)
            
            # Check if challenge server is accessible
            try:
                challenge = await self.api_client.get_active_challenge()
                server_accessible = True
                
                # Check if response indicates no active challenge
                if isinstance(challenge, dict) and challenge.get('status') in ['no_active_challenge']:
                    logger.debug("Server accessible: no active challenge")
                    challenge = None  # Convert to None for easier handling below

            except ConnectionError as e:
                logger.error(f"Challenge server unreachable: {e}")
                challenge = None
                server_accessible = False
            except Exception as e:
                logger.error(f"Error getting challenge: {e}")
                challenge = None
                server_accessible = False

            # Initialize grace period tracking if not exists
            if not hasattr(self, '_challenge_ended_grace_period_start'):
                self._challenge_ended_grace_period_start = None
            
            # Handle case: had challenge before, but not now
            if not challenge and self.state.last_challenge_id:
                
                if not server_accessible:
                    # SERVER UNREACHABLE - use cached state
                    logger.warning("Challenge server unreachable - using cached local state")
                    
                    if hasattr(self.state, 'current_challenge_expires_at') and self.state.current_challenge_expires_at:
                        now = datetime.now(timezone.utc)
                        remaining = (self.state.current_challenge_expires_at - now).total_seconds() / 3600
                        
                        if remaining > 0:
                            logger.info(f"Cached state: {self.state.last_challenge_id} has {remaining:.1f}h remaining")
                            
                            # Check emission manager for reward eligibility (even when server unreachable)
                            current_challenge_score = self.state.current_challenge_best[1] if hasattr(self.state, 'current_challenge_best') else 0.0
                            
                            if self.emission_manager.should_burn_emissions(current_challenge_score):
                                logger.info("Server unreachable - burning emissions (reward period expired)")
                                self.weight_manager.set_burn_weights()
                            else:
                                # Get reward hotkey from emission manager (respects timer)
                                current_best_hotkey = self.state.current_challenge_best[0] if hasattr(self.state, 'current_challenge_best') else None
                                current_best_score = self.state.current_challenge_best[1] if hasattr(self.state, 'current_challenge_best') else 0.0
                                
                                reward_hotkey = self.emission_manager.get_reward_hotkey(current_best_hotkey, current_best_score)
                                if reward_hotkey:
                                    uid = self.weight_manager.get_hotkey_uid(reward_hotkey)
                                    if uid is not None:
                                        weights = {reward_hotkey: 1.0}
                                        logger.info(f"Server unreachable - using cached winner {reward_hotkey[:12]}... (reward period active)")
                                        self.weight_manager.set_weights(weights)
                                    else:
                                        logger.info("Server unreachable - winner not on subnet, burning emissions")
                                        self.weight_manager.set_burn_weights()
                                else:
                                    logger.info("Server unreachable - no qualified winner, burning emissions")
                                    self.weight_manager.set_burn_weights()
                        else:
                            logger.info(f"Cached state: {self.state.last_challenge_id} expired locally")
                            # Handle expiration based on local time
                            if hasattr(self.state, 'current_challenge_best') and self.state.current_challenge_best[0]:
                                winner_hotkey = self.state.current_challenge_best[0]
                                winner_score = self.state.current_challenge_best[1]
                                winner_discovery_time = self.state.current_challenge_best_timestamp or datetime.now(timezone.utc)
                                self.emission_manager.update_winner(winner_hotkey, winner_score, winner_discovery_time)
                            else:
                                self.emission_manager.mark_challenge_ended()
                            
                            self.state.last_challenge_id = None
                            self.state.current_challenge_expires_at = None
                            self.state.save_state()
                    
                    self.next_batch_check = datetime.now(timezone.utc) + timedelta(seconds=30)
                    return
                
                else:
                    # SERVER ACCESSIBLE but no challenge
                    # Start or continue grace period
                    if self._challenge_ended_grace_period_start is None:
                        # Start 10-minute grace period
                        self._challenge_ended_grace_period_start = datetime.now(timezone.utc)
                        logger.info(f"Server accessible but no challenge - starting 10-minute grace period for {self.state.last_challenge_id}")
                        
                        # Check emission manager for reward eligibility
                        current_challenge_score = self.state.current_challenge_best[1] if hasattr(self.state, 'current_challenge_best') else 0.0
                        
                        if self.emission_manager.should_burn_emissions(current_challenge_score):
                            logger.info("Grace period started - burning emissions (no active reward period)")
                            self.weight_manager.set_burn_weights()
                        else:
                            # Get reward hotkey from emission manager (respects timer)
                            current_best_hotkey = self.state.current_challenge_best[0] if hasattr(self.state, 'current_challenge_best') else None
                            current_best_score = self.state.current_challenge_best[1] if hasattr(self.state, 'current_challenge_best') else 0.0
                            
                            reward_hotkey = self.emission_manager.get_reward_hotkey(current_best_hotkey, current_best_score)
                            if reward_hotkey:
                                uid = self.weight_manager.get_hotkey_uid(reward_hotkey)
                                if uid is not None:
                                    weights = {reward_hotkey: 1.0}
                                    logger.info(f"Grace period: Using cached winner {reward_hotkey[:12]}... (reward period active)")
                                    self.weight_manager.set_weights(weights)
                                else:
                                    logger.info("Grace period: Winner not on subnet - burning emissions")
                                    self.weight_manager.set_burn_weights()
                            else:
                                logger.info("Grace period: No qualified winner - burning emissions")
                                self.weight_manager.set_burn_weights()
                        
                        self.next_batch_check = datetime.now(timezone.utc) + timedelta(seconds=30)
                        return
                    else:
                        # Check if grace period expired
                        grace_period_elapsed = (datetime.now(timezone.utc) - self._challenge_ended_grace_period_start).total_seconds()
                        grace_period_minutes = grace_period_elapsed / 60
                        
                        if grace_period_elapsed < 600:  # 10 minutes = 600 seconds
                            logger.info(f"Grace period active ({grace_period_minutes:.1f}/10 min) - waiting for {self.state.last_challenge_id}")
                            
                            # Check emission manager for reward eligibility
                            current_challenge_score = self.state.current_challenge_best[1] if hasattr(self.state, 'current_challenge_best') else 0.0
                            
                            if self.emission_manager.should_burn_emissions(current_challenge_score):
                                logger.info("Grace period active - burning emissions (reward period expired)")
                                self.weight_manager.set_burn_weights()
                            else:
                                # Get reward hotkey from emission manager (respects timer)
                                current_best_hotkey = self.state.current_challenge_best[0] if hasattr(self.state, 'current_challenge_best') else None
                                current_best_score = self.state.current_challenge_best[1] if hasattr(self.state, 'current_challenge_best') else 0.0
                                
                                reward_hotkey = self.emission_manager.get_reward_hotkey(current_best_hotkey, current_best_score)
                                if reward_hotkey:
                                    uid = self.weight_manager.get_hotkey_uid(reward_hotkey)
                                    if uid is not None:
                                        weights = {reward_hotkey: 1.0}
                                        logger.info(f"Grace period: Using cached winner {reward_hotkey[:12]}... (reward period active)")
                                        self.weight_manager.set_weights(weights)
                                    else:
                                        logger.info("Grace period: Winner not on subnet - burning emissions")
                                        self.weight_manager.set_burn_weights()
                                else:
                                    logger.info("Grace period: Reward period expired - burning emissions")
                                    self.weight_manager.set_burn_weights()
                            
                            self.next_batch_check = datetime.now(timezone.utc) + timedelta(seconds=30)
                            return
                        else:
                            # Grace period expired - challenge truly ended
                            logger.info(f"Grace period expired (10 min) - {self.state.last_challenge_id} confirmed expired")
                            self._challenge_ended_grace_period_start = None  # Reset timer
                            
                            # Handle challenge expiration
                            if hasattr(self.state, 'current_challenge_expires_at') and self.state.current_challenge_expires_at:
                                now = datetime.now(timezone.utc)
                                if now >= self.state.current_challenge_expires_at:
                                    logger.info("Challenge expired naturally")
                                else:
                                    remaining = (self.state.current_challenge_expires_at - now).total_seconds() / 3600
                                    logger.info(f"Challenge completed manually ({remaining:.1f}h before expiration)")
                            
                            # Start winner reward
                            if hasattr(self.state, 'current_challenge_best') and self.state.current_challenge_best[0]:
                                winner_hotkey = self.state.current_challenge_best[0]
                                winner_score = self.state.current_challenge_best[1]
                                winner_discovery_time = self.state.current_challenge_best_timestamp or datetime.now(timezone.utc)
                                
                                logger.info(f"Starting winner reward period for {winner_hotkey[:12]}... (score: {winner_score})")
                                self.emission_manager.update_winner(winner_hotkey, winner_score, winner_discovery_time)
                            else:
                                logger.info(f"Challenge {self.state.last_challenge_id} completed but no winner found")
                                self.emission_manager.mark_challenge_ended()
                            
                            self.state.last_challenge_id = None
                            self.state.current_challenge_expires_at = None
                            self.state.save_state()

            if not challenge or challenge=={"status": "no_active_challenge"} or challenge=={"status": "None"}:
                # No active challenge (confirmed by server)
                current_winner = self.emission_manager.current_winner
                
                if self.emission_manager.should_burn_emissions(0.0):
                    if current_winner:
                        logger.info(f"No active challenge, winner {current_winner[:12]}... reward period expired - burning emissions")
                    else:
                        logger.info("No active challenge, no submissions found - burning emissions")
                    self.weight_manager.set_burn_weights()
                else:
                    reward_hotkey = self.emission_manager.get_reward_hotkey()
                    
                    if reward_hotkey:
                        uid = self.weight_manager.get_hotkey_uid(reward_hotkey)
                        if uid is not None:
                            weights = {reward_hotkey: 1.0}
                            logger.info(f"No active challenge, but rewarding winner {reward_hotkey[:12]}... (emission manager active)")
                            self.weight_manager.set_weights(weights)
                        else:
                            logger.info("No active challenge, winner not on subnet - burning emissions")
                            self.weight_manager.set_burn_weights()
                    else:
                        logger.info("No active challenge, no winner reward period - burning emissions")
                        self.weight_manager.set_burn_weights()
                
                self.next_batch_check = datetime.now(timezone.utc) + timedelta(seconds=10)
                return
            
            # Challenge exists - check if it's the same one or new one
            challenge_id = challenge['challenge_id']

            if 'winner_reward_hours' in challenge:
                self.emission_manager.update_winner_reward_hours_from_server(challenge['winner_reward_hours'])
            else:
                # Server didn't provide value, use local fallback
                logger.warning("Challenge server didn't provide winner_reward_hours, using local .env fallback")
                self.emission_manager.update_winner_reward_hours_from_server(None)
            
            if self._challenge_ended_grace_period_start is not None:
                # We were in grace period, but challenge appeared
                if self.state.last_challenge_id == challenge_id:
                    # Same challenge reappeared - reset grace period and continue
                    logger.info(f"Challenge {challenge_id} reappeared during grace period - continuing with cached state")
                    self._challenge_ended_grace_period_start = None
                    # Continue with existing state (don't reset)
                else:
                    # Different challenge appeared - previous challenge expired
                    logger.info(f"New challenge {challenge_id} appeared - previous challenge {self.state.last_challenge_id} expired")
                    self._challenge_ended_grace_period_start = None
                    
                    # Handle previous challenge expiration
                    if hasattr(self.state, 'current_challenge_best') and self.state.current_challenge_best[0]:
                        winner_hotkey = self.state.current_challenge_best[0]
                        winner_score = self.state.current_challenge_best[1]
                        winner_discovery_time = self.state.current_challenge_best_timestamp or datetime.now(timezone.utc)
                        
                        logger.info(f"Starting winner reward for previous challenge winner {winner_hotkey[:12]}... (score: {winner_score})")
                        self.emission_manager.update_winner(winner_hotkey, winner_score, winner_discovery_time)

            # Check for expiration time updates periodically (every 1 minute)
            if not hasattr(self, '_last_expiration_check'):
                self._last_expiration_check = {}

            current_time = datetime.now(timezone.utc)
            last_check = self._last_expiration_check.get(challenge_id, datetime.min.replace(tzinfo=timezone.utc))

            if (current_time - last_check).total_seconds() > 60:
                await self.check_challenge_expiration_update(challenge_id)
                self._last_expiration_check[challenge_id] = current_time
            
            # NEW CHALLENGE DETECTED - Reset both validator state AND emission manager
            if self.state.last_challenge_id != challenge_id:
                await self.miner_comms.notify_miners_challenge_active(challenge_id, challenge['github_url'])

                # Download test cases for new challenge
                logger.info(f"Downloading test cases for new challenge {challenge_id}")
                try:
                    test_cases_success = await self.api_client.download_test_cases(challenge_id)
                    if test_cases_success:
                        logger.info(f"Successfully downloaded and extracted test cases for challenge {challenge_id}")
                    else:
                        logger.warning(f"Failed to download test cases for challenge {challenge_id}")
                except Exception as e:
                    logger.error(f"Error downloading test cases for challenge {challenge_id}: {e}")

                # Store challenge expiration time
                if 'expires_at' in challenge:
                    challenge_expires_at = datetime.fromisoformat(challenge['expires_at'].replace('Z', '+00:00'))
                    self.state.current_challenge_expires_at = challenge_expires_at
                    logger.info(f"Challenge {challenge_id} expires at: {challenge_expires_at}")

                # Reset validator state for new challenge
                logger.info(f"New challenge {challenge_id} started - resetting current challenge best score")
                self.state.current_challenge_best = (None, 0.0)
                
                # CRITICAL: Also reset emission manager score for new challenge
                logger.info("Resetting emission manager winner score for new challenge")
                self.emission_manager.current_winner_score = 0.0
                self.emission_manager.save_state()
                
                # Clean up old evaluated batches
                if len(self.state.evaluated_batches) > 50:
                    batch_list = list(self.state.evaluated_batches)
                    self.state.evaluated_batches = set(batch_list[-50:])
                    logger.info(f"Cleaned up old evaluated batches, keeping {len(self.state.evaluated_batches)}")
                
                self.state.last_challenge_id = challenge_id
                self.state.save_state()

            # Check if test cases exist before processing batch
            if not self.api_client.check_testcase_files_exist(challenge_id):
                logger.info(f"Test case files missing for challenge {challenge_id}, downloading...")
                try:
                    test_cases_success = await self.api_client.download_test_cases(challenge_id)
                    if test_cases_success:
                        logger.info(f"Successfully downloaded test cases for challenge {challenge_id}")
                except Exception as e:
                    logger.error(f"Error downloading test cases for challenge {challenge_id}: {e}")
            
            # Get current batch
            batch = await self.api_client.get_current_batch(challenge_id)
            already_processed_batch = False
            if batch:
                # Validate batch structure
                if not isinstance(batch, dict):
                    logger.error(f"Invalid batch response type: {type(batch)}")
                    batch = None
                elif 'batch_id' not in batch:
                    logger.error(f"Batch missing 'batch_id' field: {batch}")
                    batch = None
                else:
                    batch_id = batch['batch_id']
                    if batch_id in self.state.evaluated_batches:
                        logger.info(f"Batch {batch_id} already processed")
                        already_processed_batch = True

            if not batch or already_processed_batch:
                # No batch available
                current_challenge_score = self.state.current_challenge_best[1] if hasattr(self.state, 'current_challenge_best') else 0.0
                
                if self.emission_manager.should_burn_emissions(current_challenge_score):
                    logger.info("Burning emissions - no submissions in current challenge")
                    self.weight_manager.set_burn_weights()
                else:
                    # Get reward hotkey with score comparison
                    current_best_hotkey = self.state.current_challenge_best[0] if hasattr(self.state, 'current_challenge_best') else None
                    current_best_score = self.state.current_challenge_best[1] if hasattr(self.state, 'current_challenge_best') else 0.0

                    reward_hotkey = self.emission_manager.get_reward_hotkey(current_best_hotkey, current_best_score)
                    if reward_hotkey:
                        uid = self.weight_manager.get_hotkey_uid(reward_hotkey)
                        if uid is not None:
                            weights = {reward_hotkey: 1.0}
                            logger.info(f"Setting current challenge winner weights: {reward_hotkey[:12]}...")
                            self.weight_manager.set_weights(weights)
                        else:
                            logger.info("Current challenge winner not on subnet - burning emissions")
                            self.weight_manager.set_burn_weights()
                    else:
                        logger.info("No qualified winner for rewards - burning emissions")
                        self.weight_manager.set_burn_weights()
                return

            batch_id = batch['batch_id']
            logger.info(f"Processing batch {batch_id} with {len(batch.get('submissions', []))} submissions")

            # Process the batch
            logger.info(f"Starting to process batch {batch_id}")
            try:
                success = await asyncio.wait_for(
                    self.batch_processor.process_batch(challenge_id, batch), 
                    timeout=2700
                )
                logger.info(f"Batch processing completed with success: {success}")
            except asyncio.TimeoutError:
                logger.error(f"Batch processing timed out for batch {batch_id}")
                success = False
            except Exception as e:
                logger.error(f"Exception during batch processing: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                success = False
            
            if success:
                await self.miner_comms.notify_miners_batch_complete(batch_id)
            
            if not success:
                # Fallback
                if hasattr(self.state, 'current_challenge_best') and self.state.current_challenge_best[0]:
                    current_winner = self.state.current_challenge_best[0]
                    uid = self.weight_manager.get_hotkey_uid(current_winner)
                    if uid is not None:
                        weights = {current_winner: 1.0}
                        logger.info(f"Batch processing failed, using current challenge winner {current_winner[:12]}...")
                        self.weight_manager.set_weights(weights)
                    else:
                        logger.info("Batch processing failed, current winner not on subnet - burning emissions")
                        self.weight_manager.set_burn_weights()
                else:
                    logger.info("Batch processing failed, no current winner - burning emissions")
                    self.weight_manager.set_burn_weights()

            self.consecutive_errors = 0
            
        except Exception as e:
            self.consecutive_errors += 1
            backoff_time = min(300, 30 * (2 ** min(self.consecutive_errors - 1, 4)))
            
            logger.error(f"Error in evaluation cycle (#{self.consecutive_errors}): {e}")
            logger.error(traceback.format_exc())
            logger.info(f"Will retry in {backoff_time} seconds")
            
            self.next_batch_check = datetime.now(timezone.utc) + timedelta(seconds=backoff_time)
            
            # Fallback after many errors
            if self.consecutive_errors >= 5:
                logger.warning("Multiple consecutive errors, applying fallback emission logic")
                
                if self.emission_manager.should_burn_emissions(0.0):
                    self.weight_manager.set_burn_weights()
                elif hasattr(self.state, 'current_challenge_best') and self.state.current_challenge_best[0]:
                    current_winner = self.state.current_challenge_best[0]
                    uid = self.weight_manager.get_hotkey_uid(current_winner)
                    if uid is not None:
                        weights = {current_winner: 1.0}
                        logger.info(f"Error fallback: Using current challenge winner {current_winner[:12]}...")
                        self.weight_manager.set_weights(weights)
                    else:
                        self.weight_manager.set_burn_weights()
                else:
                    self.weight_manager.set_burn_weights()

    async def run(self):
        """Main validator loop"""
        logger.info("Starting ChipForge Validator")
        
        # Create HTTP session
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=300)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        # Initialize components that depend on the session
        self.initialize_components()

        # Add crash recovery logic here
        await self.recover_from_crash()
        
        try:
            while True:
                try:
                    await self.run_evaluation_cycle()
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal")
                    break
                except Exception as e:
                    logger.error(f"Error in validator loop: {e}")
                
                # Sleep for 10 seconds before next check
                await asyncio.sleep(10)
                
        finally:
            await self.session.close()
            logger.info("ChipForge Validator stopped")


def get_config():
    """Get validator configuration"""
    parser = argparse.ArgumentParser(description="ChipForge Subnet Validator")
    
    # Add bittensor arguments
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    
    # Add custom arguments
    parser.add_argument("--challenge_api_url", type=str, default="http://localhost:8000",
                       help="Challenge server API URL")
    parser.add_argument("--validator_secret_key", type=str, required=True,
                       help="Validator secret key for API authentication")
    parser.add_argument("--netuid", type=int, required=True,
                       help="Subnet netuid")
    
    # Parse arguments and create config
    config = bt.config(parser)  # Pass parser, not args
    
    return config


async def main():
    """Main function"""
    try:
        config = get_config()
        validator = ChipForgeValidator(config)
        await validator.run()
        
    except KeyboardInterrupt:
        logger.info("Validator stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())