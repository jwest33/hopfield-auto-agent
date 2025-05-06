from typing import Any, Dict, List
import numpy as np

class Message:
    def __init__(self, sender: str, symbol: int, data: Dict[str, Any]):
        self.sender = sender
        self.symbol = symbol
        self.data = data

class DefaultComm:
    def __init__(self):
        self.handlers = []
        self.mailbox: List[Message] = []

    def broadcast(self, msg: Message):
        # send to all handlers except sender
        for h in self.handlers:
            if h.agent_id != msg.sender:
                h.receive(msg)

    def register(self, handler):
        self.handlers.append(handler)
        
    def receive_all(self) -> List[Message]:
        """Retrieve and clear all messages from the mailbox."""
        msgs = self.mailbox[:]
        self.mailbox.clear()
        return msgs

# VQ placeholder
class VQ:
    def __init__(self, n_clusters: int, dim: int):
        self.codebook_ = np.random.randn(n_clusters, dim)

    def encode(self, seq: np.ndarray) -> int:
        # assign to nearest centroid
        dists = np.linalg.norm(self.codebook_ - seq, axis=1)
        return int(np.argmin(dists))

default_comm = DefaultComm()
