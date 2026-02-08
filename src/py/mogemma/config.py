from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for Gemma 3 Embedding generation."""
    
    model_path: Path
    """Path to the local Gemma 3 model weights."""
    
    device: str = "cpu"
    """Execution device (e.g., 'cpu', 'gpu')."""
    
    max_sequence_length: int = 512
    """Maximum input sequence length."""
    
    batch_size: int = 1
    """Inference batch size."""
    
    metadata: dict[str, str] = field(default_factory=dict)
    """Additional model metadata."""

    def __post_init__(self) -> None:
        # Check if it's a local path or a HF model ID
        # Local paths typically have a '/' or exist
        if "/" in str(self.model_path) or self.model_path.exists():
            return
        
        # If it doesn't exist and isn't a likely model ID, warn
        if not self.model_path.exists():
            # For now, we allow it to support HF Hub IDs
            pass
