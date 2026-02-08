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

from .config import EmbeddingConfig, GenerationConfig

class EmbeddingModel:
    """Python wrapper for the Mojo-powered Gemma 3 Embedding engine."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        
        # Initialize Tokenizer
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

class GemmaModel:
    """Python wrapper for the Mojo-powered Gemma 3 Text generation engine."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        
        # Initialize Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(str(config.model_path))
        
        # Initialize Mojo core
        if _core is not None:
            try:
                self._llm = _core.init_model(str(config.model_path))
            except Exception as e:
                print(f"Mojo loading failed (using dummy mode): {e}")
                self._llm = None
        else:
            self._llm = None

    def generate(self, prompt: str) -> str:
        """Generate text from the given prompt."""
        # 1. Tokenize input
        encoded = self._tokenizer(
            prompt, 
            padding=True, 
            truncation=True, 
            max_length=self.config.max_sequence_length,
            return_tensors="np"
        )
        tokens = encoded["input_ids"].astype(np.int32)
        
        # 2. Inference via Mojo
        if _core is not None:
            # Pass config parameters to Mojo generate_text
            new_tokens = _core.generate_text(
                self._llm, 
                tokens, 
                self.config.max_new_tokens,
                self.config.temperature,
                self.config.top_k,
                self.config.top_p
            )
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # 3. Dummy fallback
        return "Mojo dummy response"

    @property
    def tokenizer(self) -> Any:
        """Access to the underlying tokenizer."""
        return self._tokenizer