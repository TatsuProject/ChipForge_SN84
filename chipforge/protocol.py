import typing
import pydantic
import bittensor as bt
from typing import Optional

class ChallengeNotification(bt.Synapse):
    """
    Synapse for notifying miners about new active challenges
    """
    challenge_id: str = pydantic.Field(
        ...,
        title="Challenge ID",
        description="The ID of the active challenge"
    )
    
    github_url: str = pydantic.Field(
        ..., 
        title="GitHub URL",
        description="URL to the challenge repository on GitHub"
    )
    
    message: str = pydantic.Field(
        default="",
        title="Message", 
        description="Challenge notification message"
    )
    
    timestamp: str = pydantic.Field(
        ...,
        title="Timestamp",
        description="ISO timestamp of the notification"
    )
    
    # Response field for miner acknowledgment
    response: str = pydantic.Field(
        default="",
        title="Response",
        description="Miner response to the notification"
    )

    def deserialize(self) -> dict:
        """
        Deserialize the synapse for the miner.
        """
        return {
            "challenge_id": self.challenge_id,
            "github_url": self.github_url,
            "message": self.message,
            "timestamp": self.timestamp
        }

class BatchEvaluationComplete(bt.Synapse):
    """
    Synapse for notifying miners that current batch evaluation is complete
    """
    message: str = pydantic.Field(
        default="current batch of submission is evaluated",
        title="Message",
        description="Batch evaluation completion message"
    )
    
    batch_id: Optional[str] = pydantic.Field(
        default=None,
        title="Batch ID", 
        description="ID of the completed batch"
    )
    
    timestamp: str = pydantic.Field(
        ...,
        title="Timestamp",
        description="ISO timestamp of the notification"
    )
    
    # Response field for miner acknowledgment
    response: str = pydantic.Field(
        default="",
        title="Response",
        description="Miner response to the notification"
    )

    def deserialize(self) -> dict:
        """
        Deserialize the synapse for the miner.
        """
        return {
            "message": self.message,
            "batch_id": self.batch_id,
            "timestamp": self.timestamp
        }