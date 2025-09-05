#!/bin/bash
# submit_solution.sh - Example solution submission

source .env

python python_scripts/submit_solution.py \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $MINER_HOTKEY \
    --api_url $CHALLENGE_API_URL \
    --solution_zip $FILE_TO_SUBMIT \
    --check_status