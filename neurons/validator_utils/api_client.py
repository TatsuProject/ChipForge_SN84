#!/usr/bin/env python3
"""
API Client for ChipForge Validator
Handles all API communications with challenge server and EDA server
"""

import asyncio
import aiohttp
import aiofiles
import logging
import tempfile
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from cryptography.hazmat.primitives.asymmetric import ed25519
import zipfile
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class APIClient:
    """Handles API communications for the validator"""
    
    def __init__(self, config, wallet, session: aiohttp.ClientSession):
        self.config = config
        self.wallet = wallet
        self.session = session
        
        # Challenge server configuration
        self.api_url = getattr(config, 'challenge_api_url', 'http://localhost:8000')
        self.validator_secret = getattr(config, 'validator_secret_key', '')
        
        # EDA Server configuration
        self.eda_server_url = os.getenv("EDA_SERVER_URL", "http://localhost:8080")
        self.use_dummy_evaluation = os.getenv("USE_DUMMY_EVALUATION", "false").lower() == "true"
        
        # Validator authentication
        self.validator_hotkey = self.wallet.hotkey.ss58_address
        
        # Directories
        self.base_dir = Path('./validator_data')
        self.submissions_dir = self.base_dir / 'submissions'
        self.submissions_dir.mkdir(parents=True, exist_ok=True)
    
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
                        logger.info(f"Found current batch: {batch['batch_id']} with {batch.get('available_submissions', 0)} submissions")
                        return batch
                    else:
                        logger.info(f"{batch}")
                else:
                    logger.debug(f"No current batch: {response.status}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting current batch: {e}")
            return None
    
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
    
    async def evaluate_submissions_with_eda_server(self, challenge_id: str, submissions: Dict[str, bytes]) -> Dict[str, Dict]:
        """Send submissions to EDA server for evaluation with test cases"""
        logger.info(f"Evaluating {len(submissions)} submissions with EDA server using test cases")
        
        # Fallback to dummy evaluation if configured
        if self.use_dummy_evaluation:
            return await self._dummy_evaluate_submissions(submissions)
        
        # Get test case files
        evaluator_zip_path = self.get_testcase_files(challenge_id)
            
        if not evaluator_zip_path.exists():
            logger.error(f"Evaluator zip file not found: {evaluator_zip_path}")
            return await self._dummy_evaluate_submissions(submissions)
        
        logger.info(f"Using test case files:")
        logger.info(f" Validator's testcases Zip: {evaluator_zip_path}")
        
        evaluations = {}
        
        for submission_id, submission_data in submissions.items():
            try:
                logger.info(f"Evaluating submission {submission_id} with EDA server and test cases")
                
                # Create temporary files for submission and test cases
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as design_temp:
                    design_temp.write(submission_data)
                    design_temp.flush()
                    
                    # Create timeout for each individual submission
                    timeout = aiohttp.ClientTimeout(total=900)  # 15 minutes per submission
                    
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        # Prepare multipart form data
                        form_data = aiohttp.FormData()
                        
                        # Add design zip
                        with open(design_temp.name, 'rb') as design_file:
                            form_data.add_field('design_zip', design_file.read(), 
                                              filename=f'{submission_id}.zip',
                                              content_type='application/zip')
                        
                        # Add evaluator zip file
                        with open(evaluator_zip_path, 'rb') as evaluator_zip_file:
                            form_data.add_field('evaluator_zip', evaluator_zip_file.read(),
                                              filename=f'{challenge_id}_validator.zip',
                                              content_type='application/zip')
                        
                        logger.info(f"Sending evaluation request for {submission_id}:")
                        logger.info(f"  Design zip size: {len(submission_data)} bytes")
                        logger.info(f"  Validaotr's testcases zip size: {evaluator_zip_path.stat().st_size} bytes")
                        
                        try:
                            async with session.post(
                                f"{self.eda_server_url}/evaluate",
                                data=form_data,
                            ) as response:
                                logger.info(f"EDA server response status for {submission_id}: {response.status}")
                                
                                if response.status == 200:
                                    result = await response.json()
                                    logger.info(f"Successfully evaluated {submission_id} with EDA server")
                                    logger.info(f"EDA response: {result}")
                                    
                                    # Transform EDA server response to expected format
                                    evaluations[submission_id] = self._transform_eda_response(result, submission_id)
                                    
                                else:
                                    error_text = await response.text()
                                    logger.error(f"EDA server error for {submission_id}: {response.status} - {error_text}")
                                    # Use fallback evaluation
                                    evaluations[submission_id] = self._generate_fallback_evaluation(submission_id)
                                    
                        except asyncio.TimeoutError:
                            logger.error(f"Timeout evaluating {submission_id} with EDA server")
                            evaluations[submission_id] = self._generate_fallback_evaluation(submission_id)
                        except Exception as eval_error:
                            logger.error(f"Exception during EDA evaluation for {submission_id}: {eval_error}")
                            evaluations[submission_id] = self._generate_fallback_evaluation(submission_id)
                    
                    # Clean up temporary design file
                    os.unlink(design_temp.name)
                    
            except Exception as e:
                logger.error(f"Error evaluating submission {submission_id}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Use fallback evaluation
                evaluations[submission_id] = self._generate_fallback_evaluation(submission_id)
        
        logger.info(f"EDA server evaluation completed for {len(evaluations)} submissions")
        return evaluations

    def _transform_eda_response(self, eda_result: Dict, submission_id: str) -> Dict:
        """Transform EDA server response to expected format"""
        # Extract the final score from the new response format
        final_score = eda_result.get('final_score', {})
        
        # Extract functionality score from verilator results
        verilator_results = eda_result.get('verilator_results', {})
        verilator_success = verilator_results.get('success', False)
        functionality_score = 0.0
        
        if verilator_success:
            verilator_inner_results = verilator_results.get('results', {})
            functionality_score = verilator_inner_results.get('functionality_score', 0.0)
        
        # Check if the submission passed the testbench (based on functional gate)
        passed_testbench = final_score.get('functional_gate', False) and functionality_score > 0
        
        return {
            'overall_score': final_score.get('overall', 0.0),
            'functionality_score': final_score.get('func_score', 0.0),
            'area_score': final_score.get('area_score', 0.0),
            'delay_score': final_score.get('perf_score', 0.0),  # Using perf_score as delay_score
            'power_score': 0.0,  # Power score not provided in new format
            'passed_testbench': passed_testbench,
            'evaluation_notes': f"EDA evaluation for {submission_id} - Functionality: {functionality_score:.2f}, Overall: {final_score.get('overall', 0.0):.2f}"
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
                'overall_score': 0.0,
                'functionality_score': 0.0,
                'area_score': 0.0,
                'delay_score': 0.0,
                'power_score': 0.0,
                'passed_testbench': random.choice([True, True, True, False]),
                'evaluation_notes': f"ERROR! Dummy evaluation for {submission_id}, There is an error in evaluation pipeline"
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

    async def download_test_cases(self, challenge_id: str) -> bool:
        """Download and extract test cases for a challenge"""
        try:
            logger.info(f"Downloading test cases for challenge {challenge_id}")
            
            url = f"{self.api_url}/api/v1/challenges/{challenge_id}/test_cases/download"
            
            # Create signature for authentication
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            message = f"{self.validator_hotkey}{timestamp}"
            signature = self.create_signature(message)
            
            headers = {
                'X-Validator-Secret': self.validator_secret
            }
            
            params = {
                'validator_hotkey': self.validator_hotkey,
                'signature': signature,
                'timestamp': timestamp
            }
            
            logger.info(f"Requesting test cases from: {url}")
            
            async with self.session.get(url, headers=headers, params=params, timeout=60) as response:
                if response.status == 200:
                    content = await response.read()
                    logger.info(f"Downloaded test cases: {len(content)} bytes")
                    
                    # Create test cases directory
                    testcases_dir = self.base_dir / 'testcases'
                    testcases_dir.mkdir(exist_ok=True)
                    
                    # Save the zip file
                    zip_path = testcases_dir / f"{challenge_id}_validator.zip"
                    async with aiofiles.open(zip_path, 'wb') as f:
                        await f.write(content)
                    
                    return True
                    
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to download test cases: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error downloading test cases for {challenge_id}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def get_testcase_files(self, challenge_id: str) -> tuple:
        """Get test case files for a challenge"""
        evaluator_zip_path = self.base_dir / 'testcases' / f"{challenge_id}_validator.zip"
        return evaluator_zip_path

    def check_testcase_files_exist(self, challenge_id: str) -> bool:
        """Check if all required test case files exist for a challenge"""
        try:
            evaluator_zip_path = self.base_dir / 'testcases' / f"{challenge_id}_validator.zip"
            
            if not evaluator_zip_path.exists():
                logger.warning(f"Missing test case file: {file_path}")
                return False
            
            logger.debug(f"All test case files exist for challenge {challenge_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking test case files for {challenge_id}: {e}")
            return False