"""Local LLM inference via llama-cpp-python.

Provides offline LLM capabilities using Mistral 7B Q4 GGUF model.
Designed to run on phone/MacBook with 8GB+ RAM.

Usage:
    from backend.llm import get_local_llm

    llm = get_local_llm()
    if llm.is_available:
        response = llm.generate("What is 2+2?")
"""
import os
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Default model paths to check
DEFAULT_MODEL_PATHS = [
    "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    os.path.expanduser("~/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
    "/opt/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
]


class LocalLLM:
    """Wrapper for local Mistral 7B inference via llama-cpp-python."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize LocalLLM.

        Args:
            model_path: Path to GGUF model file. If None, checks:
                1. LOCAL_LLM_PATH environment variable
                2. Default model paths
        """
        self.model_path = self._resolve_model_path(model_path)
        self._llm = None
        self._loaded = False

    def _resolve_model_path(self, model_path: Optional[str]) -> Optional[str]:
        """Find model file from various sources."""
        # Check explicit path
        if model_path and os.path.exists(model_path):
            return model_path

        # Check environment variable
        env_path = os.getenv("LOCAL_LLM_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        # Check default locations
        for path in DEFAULT_MODEL_PATHS:
            if os.path.exists(path):
                return path

        return None

    @property
    def is_available(self) -> bool:
        """Check if model file exists."""
        return self.model_path is not None and os.path.exists(self.model_path)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded in memory."""
        return self._loaded and self._llm is not None

    def load(self) -> bool:
        """Load model into memory.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if self._loaded:
            return True

        if not self.is_available:
            logger.warning(f"Model not available. Check LOCAL_LLM_PATH or download model.")
            return False

        try:
            # Import here to avoid dependency if not using offline mode
            from llama_cpp import Llama

            logger.info(f"Loading local LLM from: {self.model_path}")
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=4096,           # Context window
                n_threads=4,          # CPU threads
                n_gpu_layers=0,       # CPU-only for compatibility
                verbose=False,
            )
            self._loaded = True
            logger.info("Local LLM loaded successfully")
            return True

        except ImportError:
            logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
            return False
        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            return False

    def unload(self):
        """Unload model from memory."""
        if self._llm:
            del self._llm
            self._llm = None
            self._loaded = False
            logger.info("Local LLM unloaded")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate response from prompt.

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            system_prompt: Optional system prompt

        Returns:
            Generated text response
        """
        if not self.load():
            return "Local LLM not available. Please check model installation."

        # Build Mistral instruction format
        if system_prompt:
            full_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
        else:
            full_prompt = f"<s>[INST] {prompt} [/INST]"

        try:
            response = self._llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "[INST]"],
                echo=False,
            )

            text = response["choices"][0]["text"].strip()
            return text

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating response: {str(e)}"

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ):
        """Stream generation token by token.

        Yields:
            Token strings as they're generated
        """
        if not self.load():
            yield "Local LLM not available."
            return

        # Build Mistral instruction format
        if system_prompt:
            full_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
        else:
            full_prompt = f"<s>[INST] {prompt} [/INST]"

        try:
            for output in self._llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "[INST]"],
                echo=False,
                stream=True,
            ):
                token = output["choices"][0]["text"]
                yield token

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\nError: {str(e)}"

    def get_status(self) -> dict:
        """Get current status of the local LLM."""
        return {
            "available": self.is_available,
            "loaded": self.is_loaded,
            "model_path": self.model_path,
            "model_name": Path(self.model_path).name if self.model_path else None,
        }


# Singleton instance
_local_llm: Optional[LocalLLM] = None


def get_local_llm() -> LocalLLM:
    """Get singleton LocalLLM instance."""
    global _local_llm
    if _local_llm is None:
        _local_llm = LocalLLM()
    return _local_llm
