#!/usr/bin/env python3
"""
Weight Manager for ChipForge Validator
Handles weight setting and subnet interactions
"""

import logging
import torch
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class WeightManager:
    """Manages weight setting on the blockchain"""
    
    def __init__(self, wallet, subtensor, metagraph, config):
        self.wallet = wallet
        self.subtensor = subtensor
        self.metagraph = metagraph
        self.config = config
    
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