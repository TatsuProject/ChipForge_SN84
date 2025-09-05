#!/usr/bin/env python3
"""
ChipForge Solution Submission Script
Standalone script for submitting hardware design solutions to the challenge server
"""

import os
import sys
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import logging
import argparse
import requests
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
import binascii
import bittensor as bt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SolutionSubmitter:
    """Handles solution submission to ChipForge challenge server"""
    
    def __init__(self, config):
        self.config = config
        self.wallet = bt.wallet(config=config)
        self.api_url = config.api_url
        
        # Load private key for signatures
        self.miner_hotkey = self.wallet.hotkey.ss58_address
        self.private_key = self._load_private_key()
        
        logger.info(f"Solution Submitter initialized")
        logger.info(f"Miner hotkey: {self.miner_hotkey}")
        logger.info(f"Challenge API URL: {self.api_url}")
    
    def _load_private_key(self) -> ed25519.Ed25519PrivateKey:
        """Load private key directly from bittensor wallet file"""
        try:
            # Based on your wallet location and the fact that hotkey is "default"
            wallet_name = self.config.wallet.name  # Should be "miner"
            hotkey_name = self.config.wallet.hotkey  # Should be "default"
            
            # Construct path to hotkey file
            hotkey_path = Path.home() / "Documents/subnet_second/wallets" / wallet_name / "hotkeys" / hotkey_name
            
            logger.debug(f"Loading wallet from: {hotkey_path}")
            
            if not hotkey_path.exists():
                # Try alternative path structure
                alt_path = Path.home() / ".bittensor/wallets" / wallet_name / "hotkeys" / hotkey_name
                if alt_path.exists():
                    hotkey_path = alt_path
                    logger.debug(f"Using alternative path: {hotkey_path}")
                else:
                    raise FileNotFoundError(f"Hotkey file not found at {hotkey_path} or {alt_path}")
            
            # Load the wallet JSON
            with open(hotkey_path, 'r') as f:
                wallet_data = json.load(f)
            
            logger.debug(f"Loaded wallet data for: {wallet_data['ss58Address']}")
            
            # Extract the secret seed (this is the 32-byte private key seed)
            secret_seed = wallet_data['secretSeed']
            if secret_seed.startswith('0x'):
                secret_seed = secret_seed[2:]  # Remove 0x prefix
            
            # Convert hex to bytes
            private_key_bytes = bytes.fromhex(secret_seed)
            
            logger.debug(f"Secret seed: {secret_seed}")
            logger.debug(f"Private key bytes ({len(private_key_bytes)} bytes): {private_key_bytes.hex()}")
            
            # Create Ed25519 private key from the seed
            ed25519_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
            
            # Verify the public key matches the expected one
            public_key = ed25519_key.public_key()
            public_key_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
            
            expected_public_key = wallet_data['publicKey']
            if expected_public_key.startswith('0x'):
                expected_public_key = expected_public_key[2:]
            
            logger.debug(f"Generated public key: {public_key_bytes.hex()}")
            logger.debug(f"Expected public key:  {expected_public_key}")
            
            if public_key_bytes.hex() == expected_public_key:
                logger.debug("✅ SUCCESS: Public key matches wallet file!")
                return ed25519_key
            else:
                logger.debug("❌ Public key mismatch - trying alternative method")
                
                # Try using the full private key from the wallet (64 bytes, take first 32)
                full_private_key = wallet_data['privateKey']
                if full_private_key.startswith('0x'):
                    full_private_key = full_private_key[2:]
                
                # Take first 32 bytes as seed
                private_key_bytes = bytes.fromhex(full_private_key[:64])  # First 32 bytes
                
                logger.debug(f"Trying with first 32 bytes of privateKey: {private_key_bytes.hex()}")
                
                ed25519_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
                public_key = ed25519_key.public_key()
                public_key_bytes = public_key.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw
                )
                
                logger.debug(f"Alt method public key: {public_key_bytes.hex()}")
                
                if public_key_bytes.hex() == expected_public_key:
                    logger.debug("✅ SUCCESS: Public key matches with alternative method!")
                    return ed25519_key
                else:
                    raise ValueError(f"Neither method produced the correct public key. Expected: {expected_public_key}")
            
        except FileNotFoundError as e:
            logger.debug(f"Wallet file not found: {e}")
            logger.debug("Falling back to bittensor wallet loading...")
            return self._load_private_key_fallback()
        except Exception as e:
            logger.debug(f"Error loading from wallet file: {e}")
            logger.debug("Falling back to bittensor wallet loading...")
            return self._load_private_key_fallback()

    def _load_private_key_fallback(self) -> ed25519.Ed25519PrivateKey:
        """Fallback method using bittensor wallet directly"""
        try:
            # Your original working method that at least creates a valid key
            private_key_data = self.wallet.hotkey.private_key
            
            if isinstance(private_key_data, list):
                private_key_bytes = bytes(private_key_data)
            else:
                private_key_bytes = private_key_data
            
            # Take only first 32 bytes for Ed25519
            if len(private_key_bytes) > 32:
                private_key_bytes = private_key_bytes[:32]
            
            logger.debug(f"Fallback: Using bittensor private_key ({len(private_key_bytes)} bytes): {private_key_bytes.hex()}")
            
            return ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
            
        except Exception as e:
            logger.error(f"Fallback method also failed: {e}")
            raise
    
    def create_signature(self, message: str) -> str:
        """Create signature for message using bittensor wallet (native method)"""
        try:
            # Use bittensor's native signing method - sign the raw message, not a hash
            signature_bytes = self.wallet.hotkey.sign(data=message)
            signature_hex = signature_bytes.hex()
            
            logger.debug(f"Signing with bittensor wallet (native method):")
            logger.debug(f"  Message: {message}")
            logger.debug(f"  Message length: {len(message)}")
            logger.debug(f"  Signature: {signature_hex}")
            
            return signature_hex
            
        except Exception as e:
            logger.error(f"Error creating signature: {e}")
            raise
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def validate_solution_zip(self, zip_path: Path) -> bool:
        """Validate that the ZIP file exists and is valid"""
        try:
            if not zip_path.exists():
                logger.error(f"ZIP file not found: {zip_path}")
                return False
            
            if not zip_path.suffix.lower() == '.zip':
                logger.error(f"File must be a ZIP file: {zip_path}")
                return False
            
            # Test if ZIP is valid
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                file_list = zipf.namelist()
                logger.info(f"ZIP contains {len(file_list)} files")
                for filename in file_list[:5]:  # Show first 5 files
                    logger.debug(f"  {filename}")
                if len(file_list) > 5:
                    logger.debug(f"  ... and {len(file_list) - 5} more files")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating ZIP file: {e}")
            return False
    
    def get_active_challenge(self) -> dict:
        """Get active challenge information"""
        try:
            url = f"{self.api_url}/api/v1/challenges/active"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                if not response.json():
                    logger.info(f"Currently, No challenge is active!")
                    return None
                challenge = response.json()
                logger.info(f"Active challenge: {challenge['challenge_id']}")
                return challenge
            else:
                logger.error(f"No active challenge found: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting active challenge: {e}")
            return None
    
    def generate_submission_id(self, challenge_id: str) -> dict:
        """Generate submission ID for the challenge"""
        try:
            url = f"{self.api_url}/api/v1/challenges/{challenge_id}/generate-submission-id"
            
            data = {
                'hotkey': self.miner_hotkey
            }
            
            response = requests.post(url, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Generated submission ID: {result['submission_id']}")
                return result
            else:
                logger.error(f"Failed to generate submission ID: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating submission ID: {e}")
            return None
    
    def submit_solution(self, challenge_id: str, submission_data: dict, solution_zip_path: Path) -> bool:
        """Submit solution to challenge server"""
        try:
            url = f"{self.api_url}/api/v1/challenges/{challenge_id}/submit"
            
            # Calculate file hash
            file_hash = self.calculate_file_hash(solution_zip_path)
            
            # Create timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Convert timestamp to Unix format for signature
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            timestamp_unix = str(int(dt.timestamp()))
            
            # Create signature message using Unix timestamp
            message = f"{challenge_id}{submission_data['challenge_nonce']}{file_hash}{submission_data['submission_id']}{timestamp_unix}"
            
            # DEBUG LOGGING
            logger.debug(f"CLIENT DEBUG:")
            logger.debug(f"  challenge_id: {challenge_id}")
            logger.debug(f"  challenge_nonce: {submission_data['challenge_nonce']}")
            logger.debug(f"  file_hash: {file_hash}")
            logger.debug(f"  submission_id: {submission_data['submission_id']}")
            logger.debug(f"  timestamp_unix: {timestamp_unix}")
            logger.debug(f"  full_message: {message}")
            
            # Hash the message
            message_hash = hashlib.sha256(message.encode('utf-8')).digest()
            logger.debug(f"  message_hash: {message_hash.hex()}")
            
            signature = self.create_signature(message)
            logger.debug(f"  signature: {signature}")
            
            # Prepare form data (still use ISO timestamp for API)
            data = {
                'hotkey': self.miner_hotkey,
                'submission_id': submission_data['submission_id'],
                'signature': signature,
                'timestamp': timestamp,  # Keep ISO format for API
                'metadata': json.dumps({
                    'solution_type': 'verilog_design',
                    'submitted_by': self.miner_hotkey,
                    'submission_tool': 'chipforge_submitter'
                })
            }
            
            # Prepare file
            files = {
                'file': (solution_zip_path.name, open(solution_zip_path, 'rb'), 'application/zip')
            }
            
            logger.info(f"Submitting solution: {solution_zip_path.name}")
            logger.info(f"File size: {solution_zip_path.stat().st_size} bytes")
            logger.debug(f"File hash: {file_hash}")
            
            response = requests.post(url, data=data, files=files, timeout=120)
            
            # Close file
            files['file'][1].close()
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Solution submitted successfully!")
                logger.info(f"Submission ID: {result.get('submission_id')}")
                logger.info(f"Status: {result.get('status')}")
                return True
            else:
                logger.error(f"Failed to submit solution: {response.status_code}")
                logger.error(f"Error: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error submitting solution: {e}")
            return False
    
    def check_submission_status(self, challenge_id: str, submission_id: str) -> dict:
        """Check submission status"""
        try:
            url = f"{self.api_url}/api/v1/challenges/{challenge_id}/submissions/{submission_id}/status"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                status = response.json()
                logger.info(f"Submission status: {status.get('status')}")
                return status
            else:
                logger.error(f"Failed to get submission status: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error checking submission status: {e}")
            return None
    
    def get_miner_submissions(self, challenge_id: str) -> list:
        """Get all submissions for this miner"""
        try:
            url = f"{self.api_url}/api/v1/challenges/{challenge_id}/submissions/hotkey/{self.miner_hotkey}"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                submissions = response.json()
                logger.info(f"Found {len(submissions.get('submissions', []))} previous submissions")
                return submissions.get('submissions', [])
            else:
                logger.debug(f"No previous submissions found: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting miner submissions: {e}")
            return []


def main():
    """Main submission function"""
    parser = argparse.ArgumentParser(description="Submit ChipForge hardware design solution")
    
    # Bittensor arguments
    parser.add_argument("--wallet.name", type=str, required=True, help="Wallet name")
    parser.add_argument("--wallet.hotkey", type=str, required=True, help="Wallet hotkey")
    
    # Challenge server arguments
    parser.add_argument("--api_url", type=str, default="http://localhost:8000",
                       help="Challenge server API URL")
    
    # Solution arguments
    parser.add_argument("--solution_zip", type=str, required=True,
                       help="Path to solution ZIP file")
    parser.add_argument("--challenge_id", type=str,
                       help="Challenge ID (if not provided, will use active challenge)")
    
    # Optional arguments
    parser.add_argument("--check_status", action='store_true',
                       help="Check status of previous submissions")
    parser.add_argument("--dry_run", action='store_true',
                       help="Validate ZIP but don't submit")
    
    args = parser.parse_args()
    
    try:
        # Create config object
        config = type('Config', (), {
            'wallet': type('Wallet', (), {
                'name': getattr(args, 'wallet.name'),
                'hotkey': getattr(args, 'wallet.hotkey')
            })(),
            'api_url': args.api_url
        })()
        
        # Initialize submitter
        submitter = SolutionSubmitter(config)
        
        # Get challenge info
        if args.challenge_id:
            challenge_id = args.challenge_id
            logger.info(f"Using specified challenge: {challenge_id}")
        else:
            challenge = submitter.get_active_challenge()
            if not challenge:
                logger.error("No active challenge found and no challenge ID specified")
                sys.exit(1)
            challenge_id = challenge['challenge_id']
        
        # Check previous submissions if requested
        if args.check_status:
            submissions = submitter.get_miner_submissions(challenge_id)
            if submissions:
                logger.info("Previous submissions:")
                for sub in submissions:
                    logger.info(f"  ID: {sub.get('submission_id')} - Status: {sub.get('status')} - Score: {sub.get('score', 'N/A')}")
            else:
                logger.info("No previous submissions found")
        
        # Validate solution ZIP
        solution_zip = Path(args.solution_zip)
        if not submitter.validate_solution_zip(solution_zip):
            logger.error("Solution ZIP validation failed")
            sys.exit(1)
        
        if args.dry_run:
            logger.info(f"Dry run completed. ZIP validated: {solution_zip}")
            sys.exit(0)
        
        # Generate submission ID
        submission_data = submitter.generate_submission_id(challenge_id)
        if not submission_data:
            logger.error("Failed to generate submission ID")
            sys.exit(1)
        
        # Submit solution
        success = submitter.submit_solution(challenge_id, submission_data, solution_zip)
        if success:
            logger.info("Solution submitted successfully!")
            
            # Check status
            status = submitter.check_submission_status(challenge_id, submission_data['submission_id'])
            if status:
                logger.info(f"Final status: {status}")
        else:
            logger.error("Failed to submit solution")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Submission cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()