import os
from typing import Union, List, Any, Optional
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

# Import the Mojo native module
try:
    from . import _core
except ImportError as e:
    # Allow fallback for development/testing if .so is missing
    _core = None

from .config import EmbeddingConfig

class EmbeddingModel:
    """Python wrapper for the Mojo-powered Gemma 3 Embedding engine."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        
        # Initialize Tokenizer
        # Note: We use the local path if provided, else HF Hub
        self._tokenizer = AutoTokenizer.from_pretrained(str(config.model_path))
        
        # Initialize Mojo core
        if _core is not None:
            try:
                self._llm = _core.init_model(str(config.model_path))
            except Exception as e:
                print(f"Mojo loading failed (using dummy mode): {e}")
                self._llm = None
        else:
            print("Mojo core not found (using dummy mode)")
            self._llm = None

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for the given text."""
        if isinstance(text, str):
            text = [text]
        
        # 1. Tokenize input
        encoded = self._tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=self.config.max_sequence_length,
            return_tensors="np"
        )
        tokens = encoded["input_ids"].astype(np.int32)
        
        # 2. Inference via Mojo
        if _core is not None:
            # Pass the LLM instance and the token array to Mojo
            embeddings = _core.generate_embeddings(self._llm, tokens)
            return embeddings
        
        # 3. Dummy fallback for development
        return np.random.rand(len(text), 768).astype(np.float32)

    @property
    def tokenizer(self) -> Any:
        """Access to the underlying tokenizer."""
        return self._tokenizer
