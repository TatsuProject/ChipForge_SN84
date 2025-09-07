#!/usr/bin/env python3
"""
Validator State Management for ChipForge Validator
Manages validator state and persistence
"""

import json
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ValidatorState:
    """Manages validator state and persistence"""
    
    def __init__(self, state_file: str = "validator_state.json"):
        self.state_file = state_file
        self.current_batch_id: Optional[str] = None
        self.evaluation_in_progress: bool = False
        self.last_challenge_id: Optional[str] = None
        self.evaluated_batches: set = set()
        self.challenge_best_miners: Dict[str, Tuple[str, float]] = {}  # challenge_id -> (hotkey, score)
        self.active_challenges: Dict[str, Dict] = {}  # challenge_id -> challenge_info
        self.expired_challenges: List[str] = []
        self.current_challenge_best: Tuple[Optional[str], float] = (None, 0.0)  # Current challenge only
        self.current_challenge_expires_at: Optional[datetime] = None
        
        self.load_state()
    
    def load_state(self):
        """Load state from file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.current_batch_id = data.get('current_batch_id')
                    self.evaluation_in_progress = data.get('evaluation_in_progress', False)
                    self.last_challenge_id = data.get('last_challenge_id')
                    self.evaluated_batches = set(data.get('evaluated_batches', []))
                    self.challenge_best_miners = data.get('challenge_best_miners', {})
                    self.active_challenges = data.get('active_challenges', {})
                    self.expired_challenges = data.get('expired_challenges', [])

                    current_challenge_best = data.get('current_challenge_best', [None, 0.0])
                    self.current_challenge_best = (current_challenge_best[0], current_challenge_best[1])

                    self.current_challenge_expires_at = data.get('current_challenge_expires_at', None)
                    if self.current_challenge_expires_at:
                        self.current_challenge_expires_at = datetime.fromisoformat(self.current_challenge_expires_at)
                    
                logger.info(f"Loaded validator state: batch={self.current_batch_id}, challenge={self.last_challenge_id}")
                    
        except Exception as e:
            logger.error(f"Error loading validator state: {e}")
    
    def save_state(self):
        """Save state to file"""
        try:
            data = {
                'current_batch_id': self.current_batch_id,
                'evaluation_in_progress': self.evaluation_in_progress,
                'last_challenge_id': self.last_challenge_id,
                'evaluated_batches': list(self.evaluated_batches),
                'challenge_best_miners': self.challenge_best_miners,
                'active_challenges': self.active_challenges,
                'expired_challenges': self.expired_challenges,
                'current_challenge_best': list(self.current_challenge_best),
                'current_challenge_expires_at': self.current_challenge_expires_at.isoformat() if self.current_challenge_expires_at else None,
                'updated_at': datetime.now(timezone.utc).isoformat(),
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