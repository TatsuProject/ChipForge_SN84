#!/bin/bash  
# start_miner.sh

source .env

python neurons/miner.py \
    --netuid $NETUID \
    --subtensor.network $SUBTENSOR_NETWORK \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $MINER_HOTKEY \
    --challenge_api_url $CHALLENGE_API_URL \
    --logging.debug