#!/usr/bin/env python3
"""
Miner Communications for ChipForge Validator (Simplified)
Handles communication with miners via Bittensor protocol using default Synapse
"""

import bittensor as bt
from chipforge.protocol import SimpleMessage

import logging
from datetime import datetime, timezone
from typing import Dict, List
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

class MinerCommunications:
    """Handles communication with miners using default Synapse"""
    
    def __init__(self, dendrite, metagraph):
        self.dendrite = dendrite
        self.metagraph = metagraph
    
    # In your validator_utils/miner_comms.py, update the functions:

    async def notify_miners_challenge_active(self, challenge_id: str, github_url: str) -> Dict[int, str]:
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
            
            # Create simple custom synapse
            timestamp = datetime.now(timezone.utc).isoformat()
            message = f"CHALLENGE_ACTIVE:{challenge_id}:{github_url}:{timestamp}"
            
            synapse = SimpleMessage()
            synapse.message = message
            
            logger.info(f"Notifying {len(serving_axons)} miners about challenge {challenge_id}")
            logger.info(f"Message: {message}")
            
            # Send to miners
            responses = await self.dendrite.forward(
                axons=serving_axons,
                synapse=synapse,
                timeout=60
            )
            
            # Process responses
            miner_responses = {}
            for uid, response in zip(serving_uids, responses):
                if hasattr(response, 'response') and response.response:
                    miner_responses[uid] = response.response
                    if response.response.upper() == "OK":
                        logger.debug(f"Miner {uid} acknowledged challenge")
                else:
                    logger.warning(f"Miner {uid} did not respond")
            
            logger.info(f"Received {len(miner_responses)} responses from miners")
            return miner_responses
            
        except Exception as e:
            logger.error(f"Error notifying miners about challenge: {e}")
            return {}


    async def notify_miners_batch_complete(self, batch_id: str = None) -> Dict[int, str]:
        """Notify miners about batch completion using SimpleMessage synapse"""
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
            
            # Create simple custom synapse
            timestamp = datetime.now(timezone.utc).isoformat()
            message = f"BATCH_COMPLETE:{batch_id if batch_id else 'unknown'}:{timestamp}"
            
            synapse = SimpleMessage()
            synapse.message = message
            
            logger.info(f"Notifying {len(serving_axons)} miners about batch completion")
            logger.info(f"Message: {message}")
            
            # Send to miners
            responses = await self.dendrite.forward(
                axons=serving_axons,
                synapse=synapse,
                timeout=60
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