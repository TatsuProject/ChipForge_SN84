#!/usr/bin/env python3
"""
ChipForge Subnet Validator
Handles batch evaluation of hardware design submissions with emission management
"""

import asyncio
import hashlib
import json
import os
import time
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import traceback
import argparse

import aiohttp
import aiofiles
import bittensor as bt
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
import binascii
import torch

# Import protocol
from chipforge.protocol import ChallengeNotification, BatchEvaluationComplete

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmissionManager:
    """Manages emission phases based on challenge lifecycle"""
    
    def __init__(self, config_file: str = "emission_config.json"):
        self.config_file = config_file
        self.load_config()
        
        # State tracking with persistence
        self.subnet_start_time = None
        self.first_challenge_end_time = None
        self.winner_reward_start_time = None
        self.current_phase = "initial_burn"
        self.current_winner = None  # Add persistence for current winner
        self.last_challenge_end_time = None  # Track when last challenge ended
        
        self.load_state()
    
    def load_config(self):
        """Load timing configuration from env or defaults"""
        self.total_hours_for_winner_reward = float(os.getenv('TOTAL_HOURS_FOR_WINNER_REWARD', '0.1'))
        self.burn_after_winner_days = float(os.getenv('BURN_AFTER_WINNER_DAYS', '0'))
        
        # Convert hours to days for internal consistency
        self.initial_burn_days = self.total_hours_for_winner_reward / 24
        self.winner_reward_days = self.total_hours_for_winner_reward / 24
        
        logger.info(f"Emission config: Winner reward={self.total_hours_for_winner_reward}h ({self.winner_reward_days}d)")
    
    def load_state(self):
        """Load emission state from file"""
        try:
            if os.path.exists('emission_state.json'):
                with open('emission_state.json', 'r') as f:
                    data = json.load(f)
                    self.subnet_start_time = data.get('subnet_start_time')
                    self.first_challenge_end_time = data.get('first_challenge_end_time')
                    self.winner_reward_start_time = data.get('winner_reward_start_time')
                    self.current_phase = data.get('current_phase', 'initial_burn')
                    self.current_winner = data.get('current_winner')  # Restore winner
                    self.last_challenge_end_time = data.get('last_challenge_end_time')  # Restore last challenge end
                    
                    # Convert ISO strings back to datetime objects
                    if self.subnet_start_time:
                        self.subnet_start_time = datetime.fromisoformat(self.subnet_start_time)
                    if self.first_challenge_end_time:
                        self.first_challenge_end_time = datetime.fromisoformat(self.first_challenge_end_time)
                    if self.winner_reward_start_time:
                        self.winner_reward_start_time = datetime.fromisoformat(self.winner_reward_start_time)
                    if self.last_challenge_end_time:
                        self.last_challenge_end_time = datetime.fromisoformat(self.last_challenge_end_time)
                        
                logger.info(f"Loaded emission state: phase={self.current_phase}, winner={self.current_winner[:12] + '...' if self.current_winner else 'None'}")
        except Exception as e:
            logger.error(f"Error loading emission state: {e}")
    
    def save_state(self):
        """Save emission state to file"""
        try:
            data = {
                'subnet_start_time': self.subnet_start_time.isoformat() if self.subnet_start_time else None,
                'first_challenge_end_time': self.first_challenge_end_time.isoformat() if self.first_challenge_end_time else None,
                'winner_reward_start_time': self.winner_reward_start_time.isoformat() if self.winner_reward_start_time else None,
                'current_phase': self.current_phase,
                'current_winner': self.current_winner,
                'last_challenge_end_time': self.last_challenge_end_time.isoformat() if self.last_challenge_end_time else None,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            with open('emission_state.json', 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving emission state: {e}")
    
    def initialize_subnet(self):
        """Initialize subnet start time"""
        if not self.subnet_start_time:
            self.subnet_start_time = datetime.now(timezone.utc)
            self.current_phase = "initial_burn"
            self.save_state()
            logger.info("Initialized subnet start time")
    
    def mark_first_challenge_complete(self, winner_hotkey: str):
        """Mark first challenge as complete with winner"""
        self.first_challenge_end_time = datetime.now(timezone.utc)
        self.winner_reward_start_time = datetime.now(timezone.utc)
        self.current_phase = "winner_reward"
        self.current_winner = winner_hotkey
        self.last_challenge_end_time = datetime.now(timezone.utc)
        self.save_state()
        logger.info(f"First challenge completed, winner: {winner_hotkey[:12]}...")
    
    def update_winner(self, winner_hotkey: str):
        """Update current winner and start reward period"""
        self.current_winner = winner_hotkey
        self.winner_reward_start_time = datetime.now(timezone.utc)
        self.current_phase = "winner_reward"
        self.save_state()
        logger.info(f"New winner: {winner_hotkey[:12]}...")
    
    def mark_challenge_ended(self):
        """Mark that current challenge has ended"""
        self.last_challenge_end_time = datetime.now(timezone.utc)
        self.save_state()
        logger.info("Challenge ended, marked in emission state")
    
    def recover_emission_state_after_crash(self, current_challenge_active: bool, validator_best_miner: Optional[Tuple[str, float]] = None):
        """Recover emission state after validator crash/restart"""
        now = datetime.now(timezone.utc)
        
        # Handle case where we have a winner in validator state but not in emission state
        if not self.current_winner and validator_best_miner and validator_best_miner[0]:
            logger.info(f"Found winner in validator state but not in emission state: {validator_best_miner[0][:12]}...")
            
            # If we have a last challenge end time, use it to determine reward period
            if self.last_challenge_end_time:
                time_since_challenge_end = now - self.last_challenge_end_time
                
                if time_since_challenge_end < timedelta(hours=self.total_hours_for_winner_reward):
                    # Start winner reward period from challenge end time
                    self.current_winner = validator_best_miner[0]
                    self.winner_reward_start_time = self.last_challenge_end_time
                    self.current_phase = "winner_reward"
                    remaining_time = timedelta(hours=self.total_hours_for_winner_reward) - time_since_challenge_end
                    logger.info(f"Crash recovery: Starting winner reward for {self.current_winner[:12]}... ({remaining_time.total_seconds()/3600:.1f}h remaining)")
                    self.save_state()
                    return
                else:
                    # Winner reward period would have expired
                    logger.info(f"Crash recovery: Winner reward period expired ({time_since_challenge_end.total_seconds()/3600:.1f}h since challenge end)")
                    self.current_phase = "burn_until_submission"
                    self.save_state()
                    return
            else:
                # No challenge end time, assume winner period expired
                logger.info("Crash recovery: No challenge end time, assuming winner period expired")
                self.current_phase = "burn_until_submission"
                self.save_state()
                return
        
        # If we have a winner and reward period info
        if self.current_winner and self.winner_reward_start_time:
            time_since_reward_start = now - self.winner_reward_start_time
            reward_period_remaining = timedelta(hours=self.total_hours_for_winner_reward) - time_since_reward_start
            
            if reward_period_remaining > timedelta(0):
                # Still in reward period
                self.current_phase = "winner_reward"
                logger.info(f"Crash recovery: Continuing winner reward for {self.current_winner[:12]}... ({reward_period_remaining.total_seconds()/3600:.1f}h remaining)")
                return
            else:
                logger.info(f"Crash recovery: Winner reward period has expired")
                # Clear the winner since period expired
                self.current_winner = None
                self.current_phase = "burn_until_submission"
                self.save_state()
                return
        
        # Check if there was a challenge that ended while we were down
        if self.last_challenge_end_time and not current_challenge_active:
            time_since_challenge_end = now - self.last_challenge_end_time
            
            if time_since_challenge_end < timedelta(hours=self.total_hours_for_winner_reward):
                # We're still in the post-challenge winner reward period
                if self.current_winner:
                    self.current_phase = "winner_reward"
                    remaining_time = timedelta(hours=self.total_hours_for_winner_reward) - time_since_challenge_end
                    logger.info(f"Crash recovery: Restoring winner {self.current_winner[:12]}... reward ({remaining_time.total_seconds()/3600:.1f}h remaining)")
                    return
            else:
                # Winner reward period has expired, should burn
                logger.info(f"Crash recovery: Winner reward period expired, will burn emissions")
                self.current_phase = "burn_until_submission"
                self.current_winner = None  # Clear expired winner
                self.save_state()
                return
        
        # Default case - determine phase based on current state
        if current_challenge_active:
            self.current_phase = "normal"
        else:
            self.current_phase = "burn_until_submission"
        
        self.save_state()
        logger.info(f"Crash recovery: Set phase to {self.current_phase}")

    
    def should_burn_emissions(self, best_miner_score: float = 0.0) -> bool:
        """Determine if emissions should be burned"""
        now = datetime.now(timezone.utc)
        
        if not self.subnet_start_time:
            logger.info("No subnet start time - burning emissions")
            return True
        
        # Phase 1: Initial burn period (X days from subnet start)
        if self.current_phase == "initial_burn":
            initial_period_end = self.subnet_start_time + timedelta(days=self.initial_burn_days)
            
            if now < initial_period_end:
                if best_miner_score > 0:
                    logger.info("Good submission found during initial burn period - transitioning to normal")
                    self.current_phase = "normal"
                    self.save_state()
                    return False
                logger.info("Initial burn period active - burning emissions")
                return True
            else:
                # Initial period ended
                if best_miner_score > 0:
                    logger.info("Initial burn period ended, good submission found - transitioning to normal")
                    self.current_phase = "normal"
                    self.save_state()
                    return False
                else:
                    logger.info("Initial burn period ended, no good submissions - burning until submission")
                    self.current_phase = "burn_until_submission"
                    self.save_state()
                    return True
        
        # Phase 2: Winner reward period
        elif self.current_phase == "winner_reward":
            if self.winner_reward_start_time:
                reward_period_end = self.winner_reward_start_time + timedelta(hours=self.total_hours_for_winner_reward)
                if now < reward_period_end:
                    remaining_hours = (reward_period_end - now).total_seconds() / 3600
                    logger.info(f"Winner {self.current_winner[:12] if self.current_winner else 'Unknown'}... reward period active - {remaining_hours:.1f}h remaining")
                    return False  # Don't burn during winner reward period
                else:
                    # Winner period ended - CLEAR THE WINNER
                    logger.info(f"Winner reward period expired for {self.current_winner[:12] if self.current_winner else 'Unknown'}... - clearing winner")
                    self.current_winner = None  # CLEAR WINNER
                    
                    if best_miner_score > 0:
                        logger.info("Winner period ended, new good submission found - transitioning to normal")
                        self.current_phase = "normal"
                        self.save_state()
                        return False
                    else:
                        logger.info("Winner period ended, no new submissions - burning until submission")
                        self.current_phase = "burn_until_submission"
                        self.save_state()
                        return True
        
        # Phase 3: Burn until good submission
        elif self.current_phase == "burn_until_submission":
            if best_miner_score > 0:
                logger.info("Good submission found during burn phase - transitioning to normal")
                self.current_phase = "normal"
                self.save_state()
                return False
            logger.info("No good submissions found - burning emissions")
            return True
        
        # Phase 4: Normal operation
        elif self.current_phase == "normal":
            if best_miner_score > 0:
                logger.info("Normal operation with submissions - not burning emissions")
                return False
            else:
                logger.info("Normal operation but no submissions - burning emissions")
                return True
        
        logger.info("Unknown phase - burning emissions as fallback")
        return True
    
    def get_reward_hotkey(self, current_best_hotkey: Optional[str] = None) -> Optional[str]:
        """Get hotkey that should receive rewards - returns None if winner period expired"""
        if self.current_phase == "winner_reward" and self.current_winner:
            # Double-check that winner period hasn't expired
            if self.winner_reward_start_time:
                now = datetime.now(timezone.utc)
                reward_period_end = self.winner_reward_start_time + timedelta(hours=self.total_hours_for_winner_reward)
                if now < reward_period_end:
                    return self.current_winner
                else:
                    # Period expired, clear winner
                    logger.info(f"Winner reward period expired - clearing winner {self.current_winner[:12]}...")
                    self.current_winner = None
                    self.save_state()
                    return None
        return current_best_hotkey


class ValidatorState:
    """Manages validator state and persistence"""
    
    def __init__(self, state_file: str = "validator_state.json"):
        self.state_file = state_file
        self.current_batch_id: Optional[str] = None
        # self.last_weights: Dict[str, float] = {}
        self.evaluation_in_progress: bool = False
        self.last_challenge_id: Optional[str] = None
        self.evaluated_batches: set = set()
        self.challenge_best_miners: Dict[str, Tuple[str, float]] = {}  # challenge_id -> (hotkey, score)
        # self.overall_best_miner: Tuple[Optional[str], float] = (None, 0.0)
        self.active_challenges: Dict[str, Dict] = {}  # challenge_id -> challenge_info
        self.expired_challenges: List[str] = []
        self.current_challenge_best: Tuple[Optional[str], float] = (None, 0.0)  # Current challenge only
        
        self.load_state()
    
    def load_state(self):
        """Load state from file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.current_batch_id = data.get('current_batch_id')
                    # self.last_weights = data.get('last_weights', {})
                    self.evaluation_in_progress = data.get('evaluation_in_progress', False)
                    self.last_challenge_id = data.get('last_challenge_id')
                    self.evaluated_batches = set(data.get('evaluated_batches', []))
                    self.challenge_best_miners = data.get('challenge_best_miners', {})
                    
                    # overall_best = data.get('overall_best_miner', [None, 0.0])
                    # self.overall_best_miner = (overall_best[0], overall_best[1])

                    current_challenge_best = data.get('current_challenge_best', [None, 0.0])
                    self.current_challenge_best = (current_challenge_best[0], current_challenge_best[1])
                    
                    # logger.info(f"Loaded validator state: {len(self.last_weights)} last weights")
        except Exception as e:
            logger.error(f"Error loading validator state: {e}")
    
    def save_state(self):
        """Save state to file"""
        try:
            data = {
                'current_batch_id': self.current_batch_id,
                # 'last_weights': self.last_weights,
                'evaluation_in_progress': self.evaluation_in_progress,
                'last_challenge_id': self.last_challenge_id,
                'evaluated_batches': list(self.evaluated_batches),
                'challenge_best_miners': self.challenge_best_miners,
                # 'overall_best_miner': list(self.overall_best_miner),
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'active_challenges': self.active_challenges,
                'expired_challenges': self.expired_challenges,
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'current_challenge_best': list(self.current_challenge_best),
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving validator state: {e}")
    
    def update_best_miner(self, challenge_id: str, hotkey: str, score: float):
        """Update best miner for challenge and current challenge"""
        # Update challenge-specific best
        if challenge_id not in self.challenge_best_miners or score > self.challenge_best_miners[challenge_id][1]:
            self.challenge_best_miners[challenge_id] = (hotkey, score)
            logger.info(f"New best miner for {challenge_id}: {hotkey[:12]}... (score: {score})")
        
        # Update current challenge best (remove overall_best_miner tracking)
        if score > self.current_challenge_best[1]:
            self.current_challenge_best = (hotkey, score)
            logger.info(f"New current challenge best: {hotkey[:12]}... (score: {score})")
        
        self.save_state()


class ChipForgeValidator:
    """ChipForge Subnet Validator with batch processing and emission management"""
    
    def __init__(self, config):
        self.config = config
        self.wallet = bt.wallet(config=config)
        self.subtensor = bt.subtensor(config=config)
        self.metagraph = self.subtensor.metagraph(config.netuid)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        
        # Challenge server configuration
        self.api_url = getattr(config, 'challenge_api_url', 'http://localhost:8000')
        self.validator_secret = getattr(config, 'validator_secret_key', '')
        
        # Directories
        self.base_dir = Path('./validator_data')
        self.submissions_dir = self.base_dir / 'submissions'
        self.submissions_dir.mkdir(parents=True, exist_ok=True)
        
        # State management
        self.state = ValidatorState()
        self.emission_manager = EmissionManager()
        
        # Initialize subnet
        self.emission_manager.initialize_subnet()
        
        # Session for HTTP requests
        self.session = None
        
        # Validator authentication
        self.validator_hotkey = self.wallet.hotkey.ss58_address
        self.private_key = self._load_private_key()
        
        # Batch timing
        self.next_batch_check = datetime.now(timezone.utc)

        self.consecutive_errors = 0

        # EDA Server configuration
        self.eda_server_url = os.getenv("EDA_SERVER_URL", "http://localhost:8080")
        self.eda_api_key = os.getenv("EDA_API_KEY", "auth-eda-api-tatsu")
        self.use_dummy_evaluation = os.getenv("USE_DUMMY_EVALUATION", "false").lower() == "true"
        
        logger.info(f"ChipForge Validator initialized")
        logger.info(f"Validator hotkey: {self.validator_hotkey}")
        logger.info(f"Challenge API URL: {self.api_url}")


    async def recover_from_crash(self):
        """Enhanced crash recovery with current challenge focus"""
        try:
            logger.info("Starting enhanced crash recovery...")
            
            # Check if there's a currently active challenge
            challenge = await self.get_active_challenge()
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
                
                remaining_time = await self.get_challenge_remaining_time(challenge_id)
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
                    uid = self.get_hotkey_uid(winner_hotkey)
                    if uid is not None:
                        weights = {winner_hotkey: 1.0}
                        logger.info(f"Crash recovery: Restoring weights for current challenge winner {winner_hotkey[:12]}... (UID {uid})")
                        # self.state.last_weights = weights
                    else:
                        logger.warning(f"Crash recovery: Current challenge winner {winner_hotkey[:12]}... not found on subnet")
            
            # If no current challenge winner but we have historical winners, check emission manager
            elif self.emission_manager.current_winner:
                winner_hotkey = self.emission_manager.current_winner
                
                # Check if this historical winner should still be getting rewards
                reward_hotkey = self.emission_manager.get_reward_hotkey(winner_hotkey)
                if reward_hotkey:
                    uid = self.get_hotkey_uid(winner_hotkey)
                    if uid is not None:
                        weights = {winner_hotkey: 1.0}
                        logger.info(f"Crash recovery: Restoring weights for emission manager winner {winner_hotkey[:12]}... (UID {uid})")
                        self.state.last_weights = weights
                    else:
                        logger.warning(f"Crash recovery: Emission manager winner {winner_hotkey[:12]}... not found on subnet")
            
            # Set appropriate next check time
            if not current_challenge_active:
                self.next_batch_check = datetime.now(timezone.utc) + timedelta(seconds=30)
            
            logger.info("Enhanced crash recovery completed")
                        
        except Exception as e:
            logger.error(f"Error during enhanced crash recovery: {e}")
            logger.error(traceback.format_exc())

    
    def _load_private_key(self) -> ed25519.Ed25519PrivateKey:
        """Load Ed25519 private key from wallet"""
        try:
            # Get private key as bytes array/list from bittensor wallet
            private_key_data = self.wallet.hotkey.private_key
            
            # Convert list to bytes if needed
            if isinstance(private_key_data, list):
                private_key_bytes = bytes(private_key_data)
            else:
                private_key_bytes = private_key_data
            
            # Take only first 32 bytes for Ed25519 (in case it's expanded form)
            if len(private_key_bytes) > 32:
                private_key_bytes = private_key_bytes[:32]
            
            return ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
            
        except Exception as e:
            logger.error(f"Error loading private key: {e}")
            raise
    
    def create_signature(self, message: str) -> str:
        """Create signature using Bittensor's native signing method"""
        try:
            # Use Bittensor wallet's native signing (same as submission signatures)
            signature_bytes = self.wallet.hotkey.sign(data=message)
            signature_hex = signature_bytes.hex()
            
            logger.debug(f"Validator signature created:")
            logger.debug(f"  Message: {message}")
            logger.debug(f"  Signature: {signature_hex}")
            
            return signature_hex
            
        except Exception as e:
            logger.error(f"Error creating validator signature: {e}")
            raise
    
    async def get_active_challenge(self) -> Optional[Dict]:
        """Get active challenge from server"""
        try:
            url = f"{self.api_url}/api/v1/challenges/active"
            async with self.session.get(url) as response:
                if response.status == 200:
                    challenge = await response.json()
                    
                    # Check if response is null/None
                    if challenge is None:
                        logger.info("Currently, no challenge is active!")
                        return None
                    
                    # Check if challenge has required fields
                    if not isinstance(challenge, dict) or 'challenge_id' not in challenge:
                        logger.warning("Invalid challenge response format")
                        return None
                    
                    logger.info(f"Active challenge: {challenge['challenge_id']}")
                    return challenge
                else:
                    logger.debug(f"No active challenge found: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting active challenge: {e}")
            return None

    async def get_challenge_remaining_time(self, challenge_id: str) -> Optional[float]:
        """Get remaining time for challenge in seconds"""
        try:
            url = f"{self.api_url}/api/v1/challenges/{challenge_id}/info"
            async with self.session.get(url) as response:
                if response.status == 200:
                    challenge = await response.json()
                    if challenge and 'expires_at' in challenge:
                        expires_at = datetime.fromisoformat(challenge['expires_at'].replace('Z', '+00:00'))
                        remaining = (expires_at - datetime.now(timezone.utc)).total_seconds()
                        return max(0, remaining)
            return None
        except Exception as e:
            logger.error(f"Error getting challenge remaining time: {e}")
            return None
    
    async def notify_miners_challenge_active(self, challenge_id: str, github_url: str) -> Dict[int, str]:
        """Notify all miners about active challenge"""
        try:
            # Get serving miners
            serving_axons = []
            serving_uids = []
            
            for uid, axon in enumerate(self.metagraph.axons):
                if axon.is_serving:
                    serving_axons.append(axon)
                    serving_uids.append(uid)
            
            if not serving_axons:
                logger.warning("No serving miners found")
                return {}
            
            # Create synapse
            timestamp = datetime.now(timezone.utc).isoformat()
            message = f"{challenge_id} is active now"
            
            synapse = ChallengeNotification(
                challenge_id=challenge_id,
                github_url=github_url,
                message=message,
                timestamp=timestamp
            )
            
            logger.info(f"Notifying {len(serving_axons)} miners about challenge {challenge_id}")
            
            # Send to miners with 1 minute timeout
            responses = await self.dendrite.forward(
                axons=serving_axons,
                synapse=synapse,
                timeout=60  # 1 minute timeout
            )
            
            # Process responses
            miner_responses = {}
            for uid, response in zip(serving_uids, responses):
                if hasattr(response, 'response') and response.response:
                    miner_responses[uid] = response.response
                    if response.response.upper() == "OK":
                        logger.debug(f"Miner {uid} acknowledged challenge")
                    else:
                        logger.warning(f"Miner {uid} unexpected response: {response.response}")
                else:
                    logger.warning(f"Miner {uid} did not respond")
            
            logger.info(f"Received {len(miner_responses)} responses from miners")
            return miner_responses
            
        except Exception as e:
            logger.error(f"Error notifying miners about challenge: {e}")
            return {}
    
    async def notify_miners_batch_complete(self, batch_id: str = None) -> Dict[int, str]:
        """Notify miners that current batch evaluation is complete"""
        try:
            # Get serving miners
            serving_axons = []
            serving_uids = []
            
            for uid, axon in enumerate(self.metagraph.axons):
                if axon.is_serving:
                    serving_axons.append(axon)
                    serving_uids.append(uid)
            
            if not serving_axons:
                logger.warning("No serving miners found")
                return {}
            
            # Create synapse
            timestamp = datetime.now(timezone.utc).isoformat()
            
            synapse = BatchEvaluationComplete(
                message="current batch of submission is evaluated",
                batch_id=batch_id,
                timestamp=timestamp
            )
            
            logger.info(f"Notifying {len(serving_axons)} miners about batch completion")
            
            # Send to miners with 1 minute timeout
            responses = await self.dendrite.forward(
                axons=serving_axons,
                synapse=synapse,
                timeout=60  # 1 minute timeout
            )
            
            # Process responses  
            miner_responses = {}
            for uid, response in zip(serving_uids, responses):
                if hasattr(response, 'response') and response.response:
                    miner_responses[uid] = response.response
                    if response.response.upper() == "OK":
                        logger.debug(f"Miner {uid} acknowledged batch completion")
                else:
                    logger.warning(f"Miner {uid} did not respond to batch completion")
            
            logger.info(f"Received {len(miner_responses)} batch completion responses")
            return miner_responses
            
        except Exception as e:
            logger.error(f"Error notifying miners about batch completion: {e}")
            return {}

    async def get_current_batch(self, challenge_id: str) -> Optional[Dict]:
        """Get current evaluation batch with dynamic scheduling"""
        try:
            url = f"{self.api_url}/api/v1/challenges/{challenge_id}/batch/current"
            headers = {'X-Validator-Secret': self.validator_secret}
            
            # Create signature
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            message = f"{self.validator_hotkey}{timestamp}"
            signature = self.create_signature(message)
            
            params = {
                'validator_hotkey': self.validator_hotkey,
                'signature': signature,
                'timestamp': timestamp
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    batch = await response.json()
                    if batch.get('batch_id'):
                        # Parse expiration time for dynamic scheduling
                        expires_at = datetime.fromisoformat(batch['evaluation_ends_at'].replace('Z', '+00:00'))
                        remaining_time = (expires_at - datetime.now(timezone.utc)).total_seconds()
                        
                        # Schedule next check after batch expires + buffer
                        self.next_batch_check = expires_at + timedelta(seconds=30)
                        
                        logger.info(f"Found current batch: {batch['batch_id']} with {batch.get('available_submissions', 0)} submissions")
                        logger.info(f"Batch expires in {remaining_time/60:.1f} minutes, next check at {self.next_batch_check}")
                        
                        return batch
                else:
                    logger.debug(f"No current batch: {response.status}")
                    # No batch available, check again in 10 seconds
                    self.next_batch_check = datetime.now(timezone.utc) + timedelta(seconds=10)
                return None
                
        except Exception as e:
            logger.error(f"Error getting current batch: {e}")
            self.next_batch_check = datetime.now(timezone.utc) + timedelta(seconds=60)  # Retry in 1 minute on error
            return None

    # Additional methods for submission processing, EDA integration, etc.
    # [Previous implementation methods would continue here...]
    
    async def run_evaluation_cycle(self):
        """Main evaluation cycle with dynamic scheduling"""
        try:
            now = datetime.now(timezone.utc)
            
            # Check if it's time for next batch check
            if now < self.next_batch_check:
                return
            
            # Sync metagraph to see current miners
            self.metagraph.sync(subtensor=self.subtensor)
            
            # Get active challenge
            challenge = await self.get_active_challenge()

            # Check if challenge just expired (we had one before but don't now)
            if not challenge and self.state.last_challenge_id:
                logger.info(f"Challenge {self.state.last_challenge_id} has expired")
                
                # If we have a current challenge winner, start their reward period
                if hasattr(self.state, 'current_challenge_best') and self.state.current_challenge_best[0]:
                    winner_hotkey = self.state.current_challenge_best[0]
                    logger.info(f"Starting winner reward period for {winner_hotkey[:12]}... from expired challenge {self.state.last_challenge_id}")
                    
                    # Update emission manager to start winner reward period
                    self.emission_manager.update_winner(winner_hotkey)
                    self.emission_manager.mark_challenge_ended()
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
                    self.set_burn_weights()
                else:
                    # Check if we have a current challenge winner who should still get rewards
                    if hasattr(self.state, 'current_challenge_best') and self.state.current_challenge_best[0]:
                        current_winner_hotkey = self.state.current_challenge_best[0]
                        reward_hotkey = self.emission_manager.get_reward_hotkey(current_winner_hotkey)
                        
                        if reward_hotkey:
                            uid = self.get_hotkey_uid(reward_hotkey)
                            if uid is not None:
                                weights = {reward_hotkey: 1.0}
                                logger.info(f"No active challenge, but rewarding current best miner {reward_hotkey[:12]}...")
                                self.set_weights(weights)
                            else:
                                logger.info("No active challenge, current winner not on subnet - burning emissions")
                                self.set_burn_weights()
                        else:
                            logger.info("No active challenge, current winner reward expired - burning emissions")
                            self.set_burn_weights()
                    else:
                        logger.info("No active challenge, no current winner - burning emissions")
                        self.set_burn_weights()
                
                self.next_batch_check = now + timedelta(seconds=10)
                return
            
            challenge_id = challenge['challenge_id']
            
            # Notify miners if challenge changed
            if self.state.last_challenge_id != challenge_id:
                await self.notify_miners_challenge_active(challenge_id, challenge['github_url'])

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
            batch = await self.get_current_batch(challenge_id)
            if not batch:
                # No batch available - use current challenge best, not historical best
                current_challenge_score = self.state.current_challenge_best[1] if hasattr(self.state, 'current_challenge_best') else 0.0
                
                if self.emission_manager.should_burn_emissions(current_challenge_score):
                    logger.info("Burning emissions - no submissions in current challenge")
                    self.set_burn_weights()
                else:
                    # Get reward hotkey from emission manager
                    reward_hotkey = self.emission_manager.get_reward_hotkey(self.state.current_challenge_best[0] if hasattr(self.state, 'current_challenge_best') else None)
                    if reward_hotkey:
                        uid = self.get_hotkey_uid(reward_hotkey)
                        if uid is not None:
                            weights = {reward_hotkey: 1.0}
                            logger.info(f"Setting current challenge winner weights: {reward_hotkey[:12]}...")
                            self.set_weights(weights)
                        else:
                            logger.info("Current challenge winner not on subnet - burning emissions")
                            self.set_burn_weights()
                    else:
                        # No current challenge winner and no reward period - should burn
                        logger.info("No current challenge winner - burning emissions")
                        self.set_burn_weights()
                return
            
            batch_id = batch['batch_id']
            logger.info(f"Processing batch {batch_id} with {len(batch.get('submissions', []))} submissions")
            
            # Check if we've already processed this batch
            if batch_id in self.state.evaluated_batches:
                logger.debug(f"Batch {batch_id} already processed")
                return
            
            # Process the batch
            logger.info(f"Starting to process batch {batch_id}")
            try:
                # Add timeout to prevent hanging forever
                success = await asyncio.wait_for(
                    self.process_batch(challenge_id, batch), 
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
                await self.notify_miners_batch_complete(batch_id)
            
            if not success:
                # Use current challenge winner as fallback, or burn if none
                if hasattr(self.state, 'current_challenge_best') and self.state.current_challenge_best[0]:
                    current_winner = self.state.current_challenge_best[0]
                    uid = self.get_hotkey_uid(current_winner)
                    if uid is not None:
                        weights = {current_winner: 1.0}
                        logger.info(f"Batch processing failed, using current challenge winner {current_winner[:12]}...")
                        self.set_weights(weights)
                    else:
                        logger.info("Batch processing failed, current winner not on subnet - burning emissions")
                        self.set_burn_weights()
                else:
                    logger.info("Batch processing failed, no current winner - burning emissions")
                    self.set_burn_weights()

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
                    self.set_burn_weights()
                elif hasattr(self.state, 'current_challenge_best') and self.state.current_challenge_best[0]:
                    # Use current challenge winner as fallback
                    current_winner = self.state.current_challenge_best[0]
                    uid = self.get_hotkey_uid(current_winner)
                    if uid is not None:
                        weights = {current_winner: 1.0}
                        logger.info(f"Error fallback: Using current challenge winner {current_winner[:12]}...")
                        self.set_weights(weights)
                    else:
                        self.set_burn_weights()
                else:
                    self.set_burn_weights()
    
    def set_burn_weights(self):
        """Set weight 1.0 for uid 0 and 0.0 for all others to burn emissions"""
        try:
            # Get all UIDs
            all_uids = list(range(len(self.metagraph.neurons)))
            
            if not all_uids:
                logger.error("No neurons found in metagraph")
                return
            
            # Set weight 1.0 for uid 0, 0.0 for all others
            uids = [0] + [uid for uid in all_uids if uid != 0]
            weights = [1.0] + [0.0] * (len(uids) - 1)
            
            uids_tensor = torch.tensor(uids, dtype=torch.int64)
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            
            success = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=uids_tensor,
                weights=weights_tensor,
                wait_for_inclusion=True,
            )
            
            if success:
                logger.info("Successfully set burn weights (uid 0 = 1.0, others = 0.0)")
            else:
                logger.error("Failed to set burn weights")
                
        except Exception as e:
            logger.error(f"Error setting burn weights: {e}")
    
    def set_weights(self, weights_dict: Dict[str, float]) -> bool:
        """Set weights on the blockchain"""
        try:
            if not weights_dict:
                logger.warning("No weights to set")
                return False
            
            # Get UIDs from hotkeys
            hotkeys = list(weights_dict.keys())
            uid_mapping = self.get_uids_from_hotkeys(hotkeys)
            
            if not uid_mapping:
                logger.warning("No valid UIDs found for hotkeys")
                return False
            
            # Prepare weights for bittensor
            uids = []
            weights = []
            
            for hotkey, weight in weights_dict.items():
                if hotkey in uid_mapping:
                    uids.append(uid_mapping[hotkey])
                    weights.append(weight)
            
            if not uids:
                logger.warning("No valid UIDs to set weights for")
                return False
            
            # Convert to tensors
            uids_tensor = torch.tensor(uids, dtype=torch.int64)
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            
            # Set weights on chain
            success = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=uids_tensor,
                weights=weights_tensor,
                wait_for_inclusion=True,
            )
            
            if success:
                logger.info(f"Successfully set weights for {len(uids)} UIDs")
                return True
            else:
                logger.error("Failed to set weights on chain")
                return False
                
        except Exception as e:
            logger.error(f"Error setting weights: {e}")
            return False
    
    def get_uids_from_hotkeys(self, hotkeys: List[str]) -> Dict[str, int]:
        """Extract UIDs from hotkeys using metagraph"""
        uid_mapping = {}
        
        for hotkey in hotkeys:
            for uid, neuron in enumerate(self.metagraph.neurons):
                if neuron.hotkey == hotkey:
                    uid_mapping[hotkey] = uid
                    break
        
        return uid_mapping

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

    def get_hotkey_uid(self, hotkey: str) -> Optional[int]:
        """Get UID for hotkey on current subnet"""
        try:
            # Use your existing metagraph to find the UID
            if hasattr(self, 'metagraph') and self.metagraph:
                hotkeys = self.metagraph.hotkeys
                if hotkey in hotkeys:
                    uid = hotkeys.index(hotkey)
                    logger.info(f"Found UID {uid} for hotkey {hotkey[:12]}...")
                    return uid
                else:
                    logger.warning(f"Hotkey {hotkey[:12]}... not found in metagraph")
                    return None
            else:
                logger.error("No metagraph available")
                return None
        except Exception as e:
            logger.error(f"Error getting UID for hotkey: {e}")
            return None
    
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
            downloaded_submissions = await self.download_batch_submissions(challenge_id, batch)
            logger.info(f"Downloaded {len(downloaded_submissions)} submissions")

            if not downloaded_submissions:
                logger.warning(f"No submissions downloaded for batch {batch_id}")
                return False
            
            # Evaluate with EDA server
            logger.info(f"Evaluating {len(downloaded_submissions)} submissions with EDA server")
            evaluations = await self.evaluate_submissions_with_eda_server(downloaded_submissions)
            if not evaluations:
                logger.error(f"No evaluations received from EDA server")
                return False
            
            logger.info(f"Received {len(evaluations)} evaluations from EDA server")
            
            # Submit evaluations to challenge server
            logger.info(f"Submitting {len(evaluations)} evaluations to challenge server")
            submission_results = await self.submit_all_evaluations(challenge_id, evaluations)
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
                self.emission_manager.update_winner(new_champion)
                
                # Check if new champion is on subnet and get UID
                try:
                    uid = self.get_hotkey_uid(new_champion)
                    if uid is not None:
                        # Set weights: 1.0 for new champion, 0.0 for others
                        weights = {new_champion: 1.0}
                        logger.info(f"Setting NEW CHAMPION weights: {new_champion[:12]}... (UID {uid}) = 1.0, score: {new_best_score}")
                        
                        weight_success = self.set_weights(weights)
                        if not weight_success:
                            logger.error("Failed to set weights, burning emissions")
                            self.set_burn_weights()
                    else:
                        logger.warning(f"New champion {new_champion[:12]}... not found on subnet, burning emissions")
                        self.set_burn_weights()
                except Exception as e:
                    logger.error(f"Error getting UID for new champion: {e}, burning emissions")
                    self.set_burn_weights()
                    
            else:
                # No new champion found - check emission management policy
                logger.info(f"No submissions beat challenge best of {current_best_score}")
                
                # Get reward hotkey from emission manager (considers winner reward period)
                reward_hotkey = self.emission_manager.get_reward_hotkey(current_best_hotkey)
                should_burn = self.emission_manager.should_burn_emissions(current_best_score)
                
                if reward_hotkey and not should_burn:
                    # Continue rewarding current winner/champion
                    try:
                        uid = self.get_hotkey_uid(reward_hotkey)
                        if uid is not None:
                            weights = {reward_hotkey: 1.0}
                            logger.info(f"Challenge active, winner {reward_hotkey[:12]}... taking reward until next good submission")
                            
                            weight_success = self.set_weights(weights)
                            if not weight_success:
                                logger.error("Failed to set weights, burning emissions")
                                self.set_burn_weights()
                        else:
                            logger.warning(f"Reward target {reward_hotkey[:12]}... not found on subnet, burning emissions")
                            self.set_burn_weights()
                    except Exception as e:
                        logger.error(f"Error getting UID for reward target: {e}, burning emissions")
                        self.set_burn_weights()
                else:
                    # Should burn emissions
                    if self.emission_manager.current_winner:
                        logger.info("Challenge active, submissions found but winner reward period expired - burning emissions")
                    else:
                        logger.info("Challenge active, submissions checking but no qualified winner - burning emissions")
                    self.set_burn_weights()
            
            # Mark batch as processed
            self.state.evaluated_batches.add(batch_id)
            self.state.current_batch_id = None
            self.state.evaluation_in_progress = False
            self.state.save_state()
            
            logger.info(f"Successfully processed batch {batch_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            self.state.evaluation_in_progress = False
            self.state.save_state()
            return False
    
    async def download_batch_submissions(self, challenge_id: str, batch: Dict) -> Dict[str, bytes]:
        """Download all submissions in batch in parallel with proper filename handling"""
        submissions = batch.get('submissions', [])
        if not submissions:
            return {}
        
        logger.info(f"Downloading {len(submissions)} submissions in parallel")
        
        # Create download tasks
        tasks = []
        for submission in submissions:
            submission_id = submission['submission_id']
            task = self.download_submission(challenge_id, submission_id)
            tasks.append((submission_id, task))
        
        # Execute downloads in parallel
        downloaded = {}
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for (submission_id, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to download {submission_id}: {result}")
            elif result is not None and isinstance(result, dict):
                # Handle new return format with content, filename, and submission_id
                content = result['content']
                filename = result['filename']
                downloaded[submission_id] = content
                
                logger.info(f"Successfully downloaded {submission_id}: {len(content)} bytes")
                
                # Save to local file using server-provided filename
                batch_dir = self.submissions_dir / batch['batch_id']
                batch_dir.mkdir(exist_ok=True)
                
                file_path = batch_dir / filename
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(content)
                logger.info(f"Saved {submission_id} as: {filename}")
            else:
                logger.warning(f"Invalid or empty result for {submission_id}")
        
        logger.info(f"Successfully downloaded {len(downloaded)} submissions")
        return downloaded
    
    async def download_submission(self, challenge_id: str, submission_id: str) -> Optional[Dict]:
        """Download a single submission with enhanced debugging and filename extraction"""
        max_retries = 3
        
        logger.info(f"Starting download for submission {submission_id} in challenge {challenge_id}")
        
        for attempt in range(max_retries):
            try:
                url = f"{self.api_url}/api/v1/challenges/{challenge_id}/submissions/{submission_id}/download"
                
                # Create fresh signature for each attempt
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                message = f"{self.validator_hotkey}{timestamp}"
                
                # Use the updated create_signature method (Bittensor native signing)
                signature = self.create_signature(message)
                
                headers = {
                    'X-Validator-Secret': self.validator_secret
                }
                
                params = {
                    'validator_hotkey': self.validator_hotkey,
                    'signature': signature,
                    'timestamp': timestamp
                }
                
                # Debug logging
                logger.info(f"Download attempt {attempt + 1} for {submission_id}")
                logger.info(f"  URL: {url}")
                logger.info(f"  Validator hotkey: {self.validator_hotkey}")
                logger.info(f"  Timestamp: {timestamp}")
                logger.info(f"  Message for signing: {message}")
                logger.info(f"  Generated signature: {signature}")
                logger.info(f"  Validator secret present: {'Yes' if self.validator_secret else 'No'}")
                
                async with self.session.get(url, headers=headers, params=params, timeout=30) as response:
                    logger.info(f"Response status for {submission_id}: {response.status}")
                    
                    if response.status == 200:
                        content = await response.read()
                        content_length = len(content)
                        logger.info(f"Successfully downloaded submission {submission_id}: {content_length} bytes")
                        
                        # Verify it's actually a ZIP file
                        if content.startswith(b'PK'):
                            logger.info(f"Downloaded content appears to be a valid ZIP file")
                        else:
                            logger.warning(f"Downloaded content may not be a valid ZIP file")
                        
                        # Extract filename from Content-Disposition header
                        content_disposition = response.headers.get('Content-Disposition', '')
                        if 'filename=' in content_disposition:
                            # Extract filename from header (handles both quoted and unquoted)
                            filename_part = content_disposition.split('filename=')[1]
                            if filename_part.startswith('"') and filename_part.endswith('"'):
                                filename = filename_part[1:-1]  # Remove quotes
                            else:
                                filename = filename_part.split(';')[0].strip()  # Handle multiple params
                            logger.info(f"Using server-provided filename: {filename}")
                        else:
                            filename = f"{submission_id}.zip"
                            logger.info(f"No Content-Disposition header, using fallback: {filename}")
                        
                        return {
                            'content': content,
                            'filename': filename,
                            'submission_id': submission_id
                        }
                        
                    else:
                        # Read the error response body for detailed error info
                        try:
                            error_text = await response.text()
                            logger.error(f"Failed to download submission {submission_id}")
                            logger.error(f"  Status: {response.status}")
                            logger.error(f"  Error response: {error_text}")
                            
                            # Log response headers for additional debugging
                            response_headers = dict(response.headers)
                            if response_headers:
                                logger.error(f"  Response headers: {response_headers}")
                                
                        except Exception as read_error:
                            logger.error(f"Failed to read error response body: {read_error}")
                        
                        # Handle specific error codes
                        if response.status == 401:
                            logger.error("Authentication failed - check signature generation and server verification")
                        elif response.status == 403:
                            logger.error("Forbidden - check validator secret or batch permissions")
                        elif response.status == 404:
                            logger.error("Not found - submission may not exist or not in current batch")
                        elif response.status == 409:
                            logger.error("Conflict - you may have already evaluated this submission")
                        
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            logger.info(f"Retrying in {wait_time} seconds...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Max retries exceeded for {submission_id}")
                            return None
                            
            except asyncio.TimeoutError:
                logger.error(f"Timeout downloading submission {submission_id} (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying after timeout in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Max retries exceeded due to timeouts for {submission_id}")
                    return None
                    
            except Exception as e:
                logger.error(f"Exception downloading submission {submission_id} (attempt {attempt + 1}): {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                
                # Log full traceback for debugging
                import traceback
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying after exception in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Max retries exceeded after exceptions for {submission_id}")
                    return None
        
        logger.error(f"Complete failure: could not download {submission_id} after all attempts")
        return None
    
    async def get_submission_details(self, challenge_id: str, submission_ids: List[str]) -> Dict[str, str]:
        """Get miner hotkeys for submission IDs with enhanced debugging"""
        submission_hotkeys = {}
        
        logger.info(f"Getting submission details for {len(submission_ids)} submissions")
        logger.info(f"Submission IDs: {submission_ids}")
        
        try:
            url = f"{self.api_url}/api/v1/challenges/{challenge_id}/submissions"
            headers = {'X-Validator-Secret': self.validator_secret}
            
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            message = f"{self.validator_hotkey}{timestamp}"
            signature = self.create_signature(message)
            
            params = {
                'validator_hotkey': self.validator_hotkey,
                'signature': signature,
                'timestamp': timestamp
            }
            
            logger.info(f"Making API request to: {url}")
            logger.debug(f"Request params: {params}")
            
            async with self.session.get(url, headers=headers, params=params) as response:
                logger.info(f"API response status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"API response data keys: {list(data.keys())}")
                    
                    submissions = data.get('submissions', [])
                    logger.info(f"Found {len(submissions)} total submissions in API response")
                    
                    # Debug: Show structure of first submission
                    if submissions:
                        first_sub = submissions[0]
                        logger.info(f"First submission structure: {list(first_sub.keys())}")
                        logger.info(f"First submission sample: {first_sub}")
                    
                    for submission in submissions:
                        sub_id = submission.get('submission_id')
                        # Try both possible field names
                        miner_hotkey = submission.get('miner_hotkey') or submission.get('hotkey')
                        
                        logger.debug(f"Processing submission: id={sub_id}, hotkey={miner_hotkey}")
                        
                        if sub_id in submission_ids:
                            if miner_hotkey:
                                submission_hotkeys[sub_id] = miner_hotkey
                                logger.info(f"✅ Mapped {sub_id} -> {miner_hotkey[:12]}...")
                            else:
                                logger.warning(f"❌ No hotkey found for submission {sub_id}")
                                logger.warning(f"Available fields: {list(submission.keys())}")
                        else:
                            logger.debug(f"Skipping submission {sub_id} (not in requested list)")
                    
                    logger.info(f"Successfully mapped {len(submission_hotkeys)} of {len(submission_ids)} submissions to hotkeys")
                    
                    # Show what we couldn't map
                    unmapped = set(submission_ids) - set(submission_hotkeys.keys())
                    if unmapped:
                        logger.warning(f"Could not map these submissions: {unmapped}")
                    
                else:
                    error_text = await response.text()
                    logger.error(f"API request failed with status {response.status}")
                    logger.error(f"Error response: {error_text}")
                    
                    if response.status == 401:
                        logger.error("Authentication failed - check signature generation")
                    elif response.status == 403:
                        logger.error("Forbidden - check validator secret")
                    elif response.status == 404:
                        logger.error("Challenge or submissions not found")
                        
        except Exception as e:
            logger.error(f"Exception getting submission details: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        if not submission_hotkeys:
            logger.error("FAILED: No submissions mapped to hotkeys")
        
        return submission_hotkeys
    
    async def evaluate_submissions_with_eda_server(self, submissions: Dict[str, bytes]) -> Dict[str, Dict]:
        """Send submissions to EDA server for evaluation"""
        logger.info(f"Evaluating {len(submissions)} submissions with EDA server")
        
        # Fallback to dummy evaluation if configured
        if self.use_dummy_evaluation:
            return await self._dummy_evaluate_submissions(submissions)
        
        evaluations = {}
        
        for submission_id, submission_data in submissions.items():
            try:
                logger.info(f"Evaluating submission {submission_id} with EDA server")
                
                # Create a temporary file for the submission
                import tempfile
                import io
                
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                    temp_file.write(submission_data)
                    temp_file.flush()
                    
                    # Prepare the request to EDA server
                    import aiohttp
                    
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
                        with open(temp_file.name, 'rb') as zip_file:
                            form_data = aiohttp.FormData()
                            form_data.add_field('design_zip', zip_file, 
                                            filename=f'{submission_id}.zip', 
                                            content_type='application/zip')
                            
                            headers = {'Authorization': f'Bearer {self.eda_api_key}'}
                            
                            async with session.post(
                                f"{self.eda_server_url}/evaluate",
                                data=form_data,
                                headers=headers
                            ) as response:
                                if response.status == 200:
                                    result = await response.json()
                                    
                                    # Transform EDA server response to expected format
                                    evaluations[submission_id] = self._transform_eda_response(result, submission_id)
                                    logger.info(f"Successfully evaluated {submission_id}")
                                else:
                                    error_text = await response.text()
                                    logger.error(f"EDA server error for {submission_id}: {response.status} - {error_text}")
                                    # Use dummy evaluation as fallback
                                    evaluations[submission_id] = self._generate_fallback_evaluation(submission_id)
                    
                    # Clean up temporary file
                    import os
                    os.unlink(temp_file.name)
                    
            except Exception as e:
                logger.error(f"Error evaluating submission {submission_id}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Use dummy evaluation as fallback
                evaluations[submission_id] = self._generate_fallback_evaluation(submission_id)
        
        logger.info(f"EDA server evaluation completed for {len(evaluations)} submissions")
        return evaluations

    def _transform_eda_response(self, eda_result: Dict, submission_id: str) -> Dict:
        """Transform EDA server response to expected format"""
        # Adjust this based on your actual EDA server response format
        return {
            'overall_score': eda_result.get('overall_score', 0),
            'functionality_score': eda_result.get('functionality_score', 0),
            'area_score': eda_result.get('area_score', 0),
            'delay_score': eda_result.get('delay_score', 0),
            'power_score': eda_result.get('power_score', 0),
            'passed_testbench': eda_result.get('passed_testbench', False),
            'evaluation_notes': eda_result.get('evaluation_notes', f"EDA evaluation for {submission_id}")
        }

    def _generate_fallback_evaluation(self, submission_id: str) -> Dict:
        """Generate fallback evaluation when EDA server fails"""
        import random
        logger.warning(f"Using fallback evaluation for {submission_id}")
        
        return {
            'overall_score': 0.0,
            'functionality_score': 0.0,
            'area_score': 0.0,
            'delay_score': 0.0,
            'power_score': 0.0,
            'passed_testbench': False,
            'evaluation_notes': f"Fallback evaluation for {submission_id} (EDA server unavailable)"
        }

    async def _dummy_evaluate_submissions(self, submissions: Dict[str, bytes]) -> Dict[str, Dict]:
        """Original dummy evaluation for testing"""
        evaluations = {}
        for submission_id in submissions.keys():
            import random
            evaluations[submission_id] = {
                'overall_score': random.uniform(60, 95),
                'functionality_score': random.uniform(70, 100),
                'area_score': random.uniform(50, 90),
                'delay_score': random.uniform(60, 95),
                'power_score': random.uniform(55, 88),
                'passed_testbench': random.choice([True, True, True, False]),
                'evaluation_notes': f"Dummy evaluation for {submission_id}"
            }
        
        await asyncio.sleep(2)  # Simulate processing time
        return evaluations
    
    async def submit_all_evaluations(self, challenge_id: str, evaluations: Dict[str, Dict]) -> Dict[str, bool]:
        """Submit all evaluations in parallel"""
        logger.info(f"Submitting {len(evaluations)} evaluations")
        
        tasks = []
        for submission_id, evaluation in evaluations.items():
            task = self.submit_evaluation(challenge_id, submission_id, evaluation)
            tasks.append((submission_id, task))
        
        results = {}
        submission_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for (submission_id, _), result in zip(tasks, submission_results):
            if isinstance(result, Exception):
                logger.error(f"Failed to submit evaluation for {submission_id}: {result}")
                results[submission_id] = False
            else:
                results[submission_id] = result
        
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Successfully submitted {successful}/{len(evaluations)} evaluations")
        
        return results
    
    async def submit_evaluation(self, challenge_id: str, submission_id: str, evaluation: Dict) -> bool:
        """Submit evaluation for a single submission using form data"""
        try:
            url = f"{self.api_url}/api/v1/challenges/{challenge_id}/submissions/{submission_id}/submit_score"
            
            # Create signature for authentication
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            message = f"{self.validator_hotkey}{timestamp}"
            signature = self.create_signature(message)
            
            # Authentication parameters go in query params
            params = {
                'validator_hotkey': self.validator_hotkey,
                'signature': signature,
                'timestamp': timestamp
            }
            
            # Headers (no Content-Type needed for form data)
            headers = {
                'X-Validator-Secret': self.validator_secret
            }
            
            # Evaluation data as FORM DATA (not JSON)
            form_data = {
                'overall_score': str(evaluation['overall_score']),
                'functionality_score': str(evaluation['functionality_score']),
                'area_score': str(evaluation['area_score']),
                'delay_score': str(evaluation['delay_score']),
                'power_score': str(evaluation['power_score']),
                'passed_testbench': str(evaluation['passed_testbench']).lower(),  # boolean as string
                'evaluation_notes': evaluation.get('evaluation_notes', '')
            }
            
            logger.info(f"Submitting evaluation for {submission_id} as form data:")
            logger.info(f"  Form data: {form_data}")
            
            # Send as form data (not JSON)
            async with self.session.post(url, params=params, headers=headers, data=form_data) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Successfully submitted evaluation for {submission_id}: score {evaluation['overall_score']}")
                    logger.info(f"Server response: {result}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to submit evaluation for {submission_id}: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error submitting evaluation for {submission_id}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
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
    
    async def run(self):
        """Main validator loop"""
        logger.info("Starting ChipForge Validator")
        
        # Create HTTP session
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=300)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

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
    import argparse
    
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