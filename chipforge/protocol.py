import bittensor as bt
from typing import Optional

class SimpleMessage(bt.Synapse):
    """Simple message synapse for validator-miner communication"""
    message: str = ""
    response: str = ""