import typing
import bittensor as bt

class SimpleMessage(bt.Synapse):
    """Simple message synapse for validator-miner communication"""
    
    # Set the synapse name explicitly
    class Config:
        """Pydantic config"""
        arbitrary_types_allowed = True
    
    # Your custom fields
    message: str = ""
    response: str = ""
    
    def deserialize(self) -> "SimpleMessage":
        """Custom deserialize method"""
        return self