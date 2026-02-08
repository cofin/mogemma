from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for Gemma 3 Embedding generation."""
    
    model_path: Union[Path, str]
    """Path to the local Gemma 3 model weights or HF Hub ID."""
    
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
        model_path_str = str(self.model_path)
        if "/" in model_path_str:
            return
        
        # If it's a Path object, we can check exists
        if isinstance(self.model_path, Path) and self.model_path.exists():
            return

@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for Gemma 3 Text generation."""
    
    model_path: Union[Path, str]
    """Path to the local Gemma 3 model weights or HF Hub ID."""
    
    device: str = "cpu"
    """Execution device (e.g., 'cpu', 'gpu')."""
    
    max_sequence_length: int = 512
    """Maximum input sequence length."""
    
    max_new_tokens: int = 128
    """Maximum number of tokens to generate."""
    
    temperature: float = 1.0
    """Sampling temperature."""
    
    top_k: int = 50
    """Top-k sampling parameter."""
    
    top_p: float = 1.0
    """Top-p (nucleus) sampling parameter."""
    
    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")
            
        model_path_str = str(self.model_path)
        if "/" in model_path_str:
            return
        if isinstance(self.model_path, Path) and self.model_path.exists():
            return