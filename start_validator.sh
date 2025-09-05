#!/bin/bash
# start_validator.sh

source .env

python neurons/validator.py \
    --netuid $NETUID \
    --subtensor.network $SUBTENSOR_NETWORK \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $VALIDATOR_HOTKEY \
    --challenge_api_url $CHALLENGE_API_URL \
    --validator_secret_key $VALIDATOR_SECRET_KEY \
    --logging.debug