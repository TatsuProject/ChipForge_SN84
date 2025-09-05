#!/usr/bin/env python3
"""
ChipForge Subnet Miner
Receives challenge notifications and manages challenge downloads
"""

import os
import asyncio
import zipfile
import shutil
import requests
from datetime import datetime, timezone
from pathlib import Path
import logging
import traceback
import json
import argparse
from typing import Dict, Optional, Tuple
import bittensor as bt
from chipforge.protocol import ChallengeNotification, BatchEvaluationComplete

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChipForgeMiner:
    """ChipForge Subnet Miner"""
    
    def __init__(self, config):
        self.config = config
        self.wallet = bt.wallet(config=config)
        self.subtensor = bt.subtensor(config=config)
        self.metagraph = self.subtensor.metagraph(config.netuid)
        self.axon = bt.axon(wallet=self.wallet, config=config)
        
        # Challenge storage
        self.challenge_dir = Path('./downloaded_active_challenge')
        self.challenge_dir.mkdir(exist_ok=True)
        
        # State tracking
        self.current_challenge_id = None
        self.current_github_url = None
        
        logger.info(f"ChipForge Miner initialized")
        logger.info(f"Miner hotkey: {self.wallet.hotkey.ss58_address}")
        logger.info(f"Challenge directory: {self.challenge_dir.absolute()}")
        
        logger.info(f"Challenge directory: {self.challenge_dir.absolute()}")
    
        # Register custom synapses with axon
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        
        # Add the synapses to the axon's registry
        if hasattr(self.axon, 'forward_class_types'):
            self.axon.forward_class_types[ChallengeNotification.__name__] = ChallengeNotification
            self.axon.forward_class_types[BatchEvaluationComplete.__name__] = BatchEvaluationComplete

    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        """Generic forward handler that routes to specific handlers"""
        if isinstance(synapse, ChallengeNotification):
            return await self.challenge_notification_handler(synapse)
        elif isinstance(synapse, BatchEvaluationComplete):
            return await self.batch_evaluation_complete_handler(synapse)
        else:
            logger.warning(f"Unknown synapse type: {type(synapse)}")
            return synapse
    
    async def challenge_notification_handler(self, synapse: ChallengeNotification) -> ChallengeNotification:
        """Handle challenge notification from validator"""
        try:
            logger.info(f"Received challenge notification: {synapse.challenge_id}")
            logger.info(f"GitHub URL: {synapse.github_url}")
            logger.info(f"Message: {synapse.message}")
            
            # Update current challenge info
            self.current_challenge_id = synapse.challenge_id
            self.current_github_url = synapse.github_url
            
            # Download challenge if URL provided
            if synapse.github_url:
                success = await self.download_challenge(synapse.challenge_id, synapse.github_url)
                if success:
                    logger.info(f"Successfully downloaded challenge {synapse.challenge_id}")
                else:
                    logger.error(f"Failed to download challenge {synapse.challenge_id}")
            
            # Send acknowledgment
            synapse.response = "OK"
            return synapse
            
        except Exception as e:
            logger.error(f"Error handling challenge notification: {e}")
            logger.error(traceback.format_exc())
            synapse.response = f"ERROR: {str(e)}"
            return synapse
    
    async def batch_evaluation_complete_handler(self, synapse: BatchEvaluationComplete) -> BatchEvaluationComplete:
        """Handle batch evaluation complete notification from validator"""
        try:
            logger.info(f"Received batch evaluation complete notification")
            logger.info(f"Message: {synapse.message}")
            if synapse.batch_id:
                logger.info(f"Batch ID: {synapse.batch_id}")
            
            # Log the completion (miners don't need to take action)
            logger.info("Batch evaluation completed by validator")
            
            # Send acknowledgment
            synapse.response = "OK"
            return synapse
            
        except Exception as e:
            logger.error(f"Error handling batch completion notification: {e}")
            logger.error(traceback.format_exc())
            synapse.response = f"ERROR: {str(e)}"
            return synapse
    
    async def download_challenge(self, challenge_id: str, github_url: str) -> bool:
        """Download challenge from GitHub and extract to challenge directory"""
        try:
            # Create challenge-specific directory instead of clearing all
            challenge_specific_dir = self.challenge_dir / challenge_id
            
            # Check if already exists
            if challenge_specific_dir.exists():
                logger.info(f"Challenge {challenge_id} directory already exists, skipping download")
                return True
            
            challenge_specific_dir.mkdir(parents=True, exist_ok=True)
            
            # Parse GitHub URL to get download URL
            download_url = self.convert_github_url_to_download(github_url)
            if not download_url:
                logger.error(f"Could not convert GitHub URL to download URL: {github_url}")
                return False
            
            logger.info(f"Downloading challenge {challenge_id} from: {download_url}")
            
            # Download challenge
            response = requests.get(download_url, timeout=30)
            response.raise_for_status()
            
            # Save and extract ZIP
            zip_path = challenge_specific_dir / f"{challenge_id}.zip"
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extract ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(challenge_specific_dir)
            
            # Remove ZIP file
            zip_path.unlink()
            
            # Create metadata file
            metadata = {
                'challenge_id': challenge_id,
                'github_url': github_url,
                'download_url': download_url,
                'downloaded_at': datetime.now(timezone.utc).isoformat(),
                'miner_hotkey': self.wallet.hotkey.ss58_address
            }
            
            metadata_path = challenge_specific_dir / 'challenge_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Challenge {challenge_id} downloaded and extracted successfully")
            
            # List downloaded files
            files = list(challenge_specific_dir.rglob('*'))
            logger.info(f"Downloaded {len(files)} files/directories for {challenge_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading challenge {challenge_id}: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def convert_github_url_to_download(self, github_url: str) -> str:
        """Convert GitHub repository URL to ZIP download URL"""
        try:
            # Handle different GitHub URL formats
            if github_url.endswith('.git'):
                github_url = github_url[:-4]
            
            if '/tree/' in github_url:
                # URL points to specific branch/folder
                parts = github_url.split('/tree/')
                base_url = parts[0]
                branch_path = parts[1]
                
                if '/' in branch_path:
                    # Extract branch name (first part before /)
                    branch = branch_path.split('/')[0]
                else:
                    branch = branch_path
                
                download_url = f"{base_url}/archive/{branch}.zip"
            else:
                # URL points to main repository
                download_url = f"{github_url}/archive/main.zip"
            
            return download_url
            
        except Exception as e:
            logger.error(f"Error converting GitHub URL: {e}")
            return ""

    async def blacklist(self, synapse: bt.Synapse) -> Tuple[bool, str]:
        """Generic blacklist function"""
        return False, ""

    async def priority(self, synapse: bt.Synapse) -> float:
        """Generic priority function"""
        if isinstance(synapse, ChallengeNotification):
            return 1000.0
        elif isinstance(synapse, BatchEvaluationComplete):
            return 500.0
        return 0.0
    
    def get_challenge_files(self) -> Dict[str, Path]:
        """Get paths to current challenge files"""
        files = {}
        
        if not self.current_challenge_id:
            return files
        
        # Look in current challenge directory
        current_challenge_dir = self.challenge_dir / self.current_challenge_id
        if not current_challenge_dir.exists():
            return files
        
        # Look for common challenge files
        common_files = [
            'metadata.json',
            'specification.md', 
            'testbench.v',
            'constraints.json',
            'REFERENCE_IMPL.V'
        ]
        
        for filename in common_files:
            file_paths = list(current_challenge_dir.rglob(filename))
            if file_paths:
                files[filename] = file_paths[0]
        
        # Find all Verilog files
        verilog_files = list(current_challenge_dir.rglob('*.v')) + list(current_challenge_dir.rglob('*.sv'))
        if verilog_files:
            files['verilog_files'] = verilog_files
        
        return files

    async def check_active_challenge(self) -> Optional[Dict]:
        """Check if there's an active challenge"""
        try:
            # Use same API as validator
            url = f"{self.config.challenge_api_url}/api/v1/challenges/active"
            
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                challenge = response.json()
                if challenge and isinstance(challenge, dict) and 'challenge_id' in challenge:
                    return challenge
            return None
        except Exception as e:
            logger.error(f"Error checking active challenge: {e}")
            return None

    async def poll_for_challenges(self):
        """Periodically check for new challenges"""
        downloaded_challenges = set()  # Track downloaded challenges
        
        while True:
            try:
                challenge = await self.check_active_challenge()
                
                if challenge:
                    challenge_id = challenge['challenge_id']
                    github_url = challenge.get('github_url', '')
                    
                    # Check if this is a new challenge AND not already downloaded
                    if challenge_id != self.current_challenge_id or challenge_id not in downloaded_challenges:
                        logger.info(f"New challenge detected: {challenge_id}")
                        
                        # Update current state
                        self.current_challenge_id = challenge_id
                        self.current_github_url = github_url
                        
                        # Download challenge only if not already downloaded
                        if challenge_id not in downloaded_challenges and github_url:
                            success = await self.download_challenge(challenge_id, github_url)
                            if success:
                                downloaded_challenges.add(challenge_id)
                                logger.info(f"Auto-downloaded challenge {challenge_id}")
                            else:
                                logger.error(f"Failed to auto-download challenge {challenge_id}")
                        elif challenge_id in downloaded_challenges:
                            logger.info(f"Challenge {challenge_id} already downloaded, skipping")
                    
                    # Log remaining time (only occasionally to avoid spam)
                    if 'expires_at' in challenge:
                        try:
                            expires_at = datetime.fromisoformat(challenge['expires_at'].replace('Z', '+00:00'))
                            remaining = (expires_at - datetime.now(timezone.utc)).total_seconds()
                            if remaining > 0:
                                # Only log every 10 minutes to reduce spam
                                if int(remaining) % 600 < 60:  # Log when remaining time is multiple of 10 minutes
                                    logger.info(f"Challenge {challenge_id} expires in {remaining/3600:.1f} hours")
                            else:
                                logger.info(f"Challenge {challenge_id} has expired")
                                # Remove from current if expired
                                if self.current_challenge_id == challenge_id:
                                    self.current_challenge_id = None
                                    self.current_github_url = None
                        except Exception:
                            pass
                else:
                    # No active challenge
                    if self.current_challenge_id:
                        logger.info("No active challenge found")
                        self.current_challenge_id = None
                        self.current_github_url = None
            
            except Exception as e:
                logger.error(f"Error in challenge polling: {e}")
            
            # Check every 60 seconds
            await asyncio.sleep(60)
    
    def print_challenge_info(self):
        """Print information about current challenge"""
        if not self.current_challenge_id:
            logger.info("No active challenge")
            return
        
        logger.info(f"Current challenge: {self.current_challenge_id}")
        logger.info(f"GitHub URL: {self.current_github_url}")
        
        files = self.get_challenge_files()
        if files:
            logger.info("Challenge files:")
            for name, path in files.items():
                if name == 'verilog_files':
                    logger.info(f"  {name}: {len(path)} files")
                    for vfile in path:
                        logger.info(f"    {vfile.relative_to(self.challenge_dir)}")
                else:
                    logger.info(f"  {name}: {path.relative_to(self.challenge_dir)}")
        else:
            logger.info("No challenge files found")
    
    async def run(self):
        """Main miner loop"""
        logger.info("Starting ChipForge Miner")
        
        # Serve axon
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()
        
        logger.info(f"Axon serving on {self.axon.external_ip}:{self.axon.external_port}")
        
        # Start challenge polling task
        polling_task = asyncio.create_task(self.poll_for_challenges())
        
        try:
            # Main loop
            while True:
                try:
                    # Sync metagraph
                    self.metagraph.sync(subtensor=self.subtensor)
                    
                    # Print current status every 5 minutes
                    await asyncio.sleep(300)
                    self.print_challenge_info()
                    
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal")
                    break
                except Exception as e:
                    logger.error(f"Error in miner loop: {e}")
                    await asyncio.sleep(10)
        
        finally:
            polling_task.cancel()
            self.axon.stop()
            logger.info("ChipForge Miner stopped")


def get_config():
    """Get miner configuration"""
    
    parser = argparse.ArgumentParser(description="ChipForge Subnet Miner")
    
    # Add bittensor arguments
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    
    parser.add_argument("--netuid", type=int, required=True,
                       help="Subnet netuid")
    parser.add_argument("--challenge_api_url", type=str, 
                       default="http://localhost:8000",
                       help="Challenge server API URL")
    
    config = bt.config(parser)
    
    return config


async def main():
    """Main function"""
    try:
        config = get_config()
        miner = ChipForgeMiner(config)
        await miner.run()
        
    except KeyboardInterrupt:
        logger.info("Miner stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())