#!/usr/bin/env python3
"""
ChipForge Subnet Validator (Refactored)
Main validator entry point with modular architecture
"""
import bittensor as bt
from chipforge.protocol import SimpleMessage

import asyncio
import aiohttp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    MinerCommunications
)


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
            current_challenge_active = challenge is not None
            
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
    
    async def run_evaluation_cycle(self):
        """Main evaluation cycle with dynamic scheduling"""
        try:
            now = datetime.now(timezone.utc)
            
            # Sync metagraph to see current miners
            self.metagraph.sync(subtensor=self.subtensor)
            
            # Get active challenge
            challenge = await self.api_client.get_active_challenge()

            # Check if challenge just expired (we had one before but don't now)
            if not challenge and self.state.last_challenge_id:
                logger.info(f"Challenge {self.state.last_challenge_id} has expired")
                
                # If we have a current challenge winner, start their reward period
                if hasattr(self.state, 'current_challenge_best') and self.state.current_challenge_best[0]:
                    winner_hotkey = self.state.current_challenge_best[0]
                    logger.info(f"Starting winner reward period for {winner_hotkey[:12]}... from expired challenge {self.state.last_challenge_id}")
                    
                    # Use current time as effective expiration time (handles manual completion)
                    effective_expiration_time = datetime.now(timezone.utc)
                    
                    # Update emission manager to start winner reward period from effective expiration
                    self.emission_manager.current_winner = winner_hotkey
                    self.emission_manager.winner_reward_start_time = effective_expiration_time
                    self.emission_manager.last_challenge_end_time = effective_expiration_time
                    self.emission_manager.current_phase = "winner_reward"  # Now it's a timed period
                    self.emission_manager.save_state()
                    
                    logger.info(f"Winner reward period started at {effective_expiration_time} for {self.emission_manager.total_hours_for_winner_reward}h")
                else:
                    logger.info(f"Challenge {self.state.last_challenge_id} expired but no winner found")
                    self.emission_manager.mark_challenge_ended()
                
                # Clear the last challenge ID since it's expired
                self.state.last_challenge_id = None
                self.state.save_state()

            if not challenge:
                # No active challenge - check emission manager and current challenge best
                current_winner = self.emission_manager.current_winner
                
                if self.emission_manager.should_burn_emissions(0.0):  # Use 0.0 since no active challenge
                    if current_winner:
                        logger.info(f"No active challenge, winner {current_winner[:12]}... reward period expired - burning emissions")
                    else:
                        logger.info("No active challenge, no submissions found - burning emissions")
                    self.weight_manager.set_burn_weights()
                else:
                    # Check if we have a current challenge winner who should still get rewards
                    if hasattr(self.state, 'current_challenge_best') and self.state.current_challenge_best[0]:
                        current_winner_hotkey = self.state.current_challenge_best[0]
                        reward_hotkey = self.emission_manager.get_reward_hotkey(current_winner_hotkey)
                        
                        if reward_hotkey:
                            uid = self.weight_manager.get_hotkey_uid(reward_hotkey)
                            if uid is not None:
                                weights = {reward_hotkey: 1.0}
                                logger.info(f"No active challenge, but rewarding current best miner {reward_hotkey[:12]}...")
                                self.weight_manager.set_weights(weights)
                            else:
                                logger.info("No active challenge, current winner not on subnet - burning emissions")
                                self.weight_manager.set_burn_weights()
                        else:
                            logger.info("No active challenge, current winner reward expired - burning emissions")
                            self.weight_manager.set_burn_weights()
                    else:
                        logger.info("No active challenge, no current winner - burning emissions")
                        self.weight_manager.set_burn_weights()
                
                self.next_batch_check = now + timedelta(seconds=10)
                return
            
            challenge_id = challenge['challenge_id']
            
            # Notify miners if challenge changed
            if self.state.last_challenge_id != challenge_id:
                await self.miner_comms.notify_miners_challenge_active(challenge_id, challenge['github_url'])

                # Store challenge expiration time for winner reward calculations
                if 'expires_at' in challenge:
                    challenge_expires_at = datetime.fromisoformat(challenge['expires_at'].replace('Z', '+00:00'))
                    self.state.current_challenge_expires_at = challenge_expires_at
                    logger.info(f"Challenge {challenge_id} expires at: {challenge_expires_at}")

                # Reset current challenge best for new challenge
                logger.info(f"New challenge {challenge_id} started - resetting current challenge best score")
                self.state.current_challenge_best = (None, 0.0)
                
                # Clean up old evaluated batches (keep only last 50 to prevent infinite growth)
                if len(self.state.evaluated_batches) > 50:
                    # Convert to list, sort, and keep only recent ones
                    batch_list = list(self.state.evaluated_batches)
                    self.state.evaluated_batches = set(batch_list[-50:])
                    logger.info(f"Cleaned up old evaluated batches, keeping {len(self.state.evaluated_batches)}")
                
                self.state.last_challenge_id = challenge_id
                self.state.save_state()
            
            # Get current batch
            batch = await self.api_client.get_current_batch(challenge_id)
            if not batch:
                # No batch available - use current challenge best, not historical best
                current_challenge_score = self.state.current_challenge_best[1] if hasattr(self.state, 'current_challenge_best') else 0.0
                
                if self.emission_manager.should_burn_emissions(current_challenge_score):
                    logger.info("Burning emissions - no submissions in current challenge")
                    self.weight_manager.set_burn_weights()
                else:
                    # Get reward hotkey from emission manager
                    reward_hotkey = self.emission_manager.get_reward_hotkey(self.state.current_challenge_best[0] if hasattr(self.state, 'current_challenge_best') else None)
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
                        # No current challenge winner and no reward period - should burn
                        logger.info("No current challenge winner - burning emissions")
                        self.weight_manager.set_burn_weights()
                return
            
            batch_id = batch['batch_id']
            logger.info(f"Processing batch {batch_id} with {len(batch.get('submissions', []))} submissions")
            
            # Check if we've already processed this batch
            if batch_id in self.state.evaluated_batches:
                logger.info(f"Batch {batch_id} already processed")
                return
            
            # Process the batch
            logger.info(f"Starting to process batch {batch_id}")
            try:
                # Add timeout to prevent hanging forever
                success = await asyncio.wait_for(
                    self.batch_processor.process_batch(challenge_id, batch), 
                    timeout=150  # 2.5 minutes timeout
                )
                logger.info(f"Batch processing completed with success: {success}")
            except asyncio.TimeoutError:
                logger.error(f"Batch processing timed out after 2.5 minutes for batch {batch_id}")
                success = False
            except Exception as e:
                logger.error(f"Exception during batch processing: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                success = False
            
            logger.info(f"Batch processing completed with success: {success}")
            
            if success:
                # Notify miners about batch completion
                await self.miner_comms.notify_miners_batch_complete(batch_id)
            
            if not success:
                # Use current challenge winner as fallback, or burn if none
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
            
            # Exponential backoff: 30s, 60s, 120s, 240s, max 300s
            backoff_time = min(300, 30 * (2 ** min(self.consecutive_errors - 1, 4)))
            
            logger.error(f"Error in evaluation cycle (#{self.consecutive_errors}): {e}")
            logger.error(traceback.format_exc())
            logger.info(f"Will retry in {backoff_time} seconds")
            
            # Update next check time with backoff
            self.next_batch_check = datetime.now(timezone.utc) + timedelta(seconds=backoff_time)
            
            # After too many consecutive errors, use fallback logic
            if self.consecutive_errors >= 5:
                logger.warning("Multiple consecutive errors, applying fallback emission logic")
                best_score = 0.0  # No active challenge means no current submissions
                
                if self.emission_manager.should_burn_emissions(best_score):
                    self.weight_manager.set_burn_weights()
                elif hasattr(self.state, 'current_challenge_best') and self.state.current_challenge_best[0]:
                    # Use current challenge winner as fallback
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