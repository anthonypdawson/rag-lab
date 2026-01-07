"""
Ollama Integration Module (RAG-Lab)

This module provides functionality to connect to and interact with Ollama models
for processing multimodal content (text, audio, video).
"""

import asyncio
from typing import Dict, List, Optional, Generator, Any
import logging
import ollama
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaProcessor:
    """Main class for processing content through Ollama models"""
    
    def __init__(self, 
                 model_name: str = "llama2", 
                 host: str = "http://localhost:11434",
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None):
        """
        Initialize Ollama processor
        Args:
            model_name (str): Name of the Ollama model to use
            host (str): Ollama server host URL
            temperature (float): Temperature for text generation
            max_tokens (Optional[int]): Maximum tokens per request
        """
        self.model_name = model_name
        self.host = host
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = ollama.Client(host=host)
        self._verify_model()
    
    def _verify_model(self) -> None:
        """Verify that the specified model is available"""
        try:
            models = self.client.list()
            available_models = [model.model for model in models.models]
            if self.model_name not in available_models:
                logger.warning(f"Model '{self.model_name}' not found in available models: {available_models}")
                logger.info(f"You can download the model by running: ollama pull {self.model_name}")
                raise ValueError(f"Model '{self.model_name}' not available. Available models: {available_models}")
            logger.info(f"Successfully connected to Ollama model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            logger.info("Make sure Ollama is running (ollama serve) and the model is pulled")
            raise
    
    def list_available_models(self) -> List[Dict]:
        """List all available models"""
        try:
            models = self.client.list()
            return [{'name': model.model} for model in models.models]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def process_text(self, 
                     text: str, 
                     prompt_template: str = None, 
                     system_prompt: str = None) -> str:
        """
        Process a single text chunk through Ollama
        Args:
            text (str): Text content to process
            prompt_template (str): Template for the prompt (should contain {text} placeholder)
            system_prompt (str): System prompt to set context
        Returns:
            str: Processed text response from the model
        """
        if not text.strip():
            return ""
        if prompt_template is None:
            prompt_template = "Please analyze and summarize the following text:\n\n{text}"
        formatted_prompt = prompt_template.format(text=text)
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt or 'You are a helpful assistant analyzing content.'},
                    {'role': 'user', 'content': formatted_prompt}
                ],
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens
                } if self.max_tokens else {'temperature': self.temperature}
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error processing text with Ollama: {e}")
            raise
    
    def process_text_streaming(self, 
                              text: str, 
                              prompt_template: str = None, 
                              system_prompt: str = None) -> Generator[str, None, None]:
        """
        Process text with streaming response
        Args:
            text (str): Text content to process
            prompt_template (str): Template for the prompt
            system_prompt (str): System prompt to set context
        Yields:
            str: Streaming response chunks from the model
        """
        if not text.strip():
            return
        if prompt_template is None:
            prompt_template = "Please analyze and summarize the following text:\n\n{text}"
        formatted_prompt = prompt_template.format(text=text)
        try:
            stream = self.client.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt or 'You are a helpful assistant analyzing content.'},
                    {'role': 'user', 'content': formatted_prompt}
                ],
                stream=True,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens
                } if self.max_tokens else {'temperature': self.temperature}
            )
            for chunk in stream:
                yield chunk['message']['content']
        except Exception as e:
            logger.error(f"Error processing text with Ollama (streaming): {e}")
            raise
