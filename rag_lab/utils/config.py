"""
Configuration management for RAG-Lab

Handles loading and managing configuration from various sources.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

DEFAULT_CONFIG = {
    'ollama': {
        'model': 'llama3.2:latest',
        'host': 'http://localhost:11434',
        'temperature': 0.7
    },
    'processing': {
        'chunk_size': 4000,
        'chunk_overlap': 200,
        'output_format': 'markdown',
        'save_chunks': False,
        'processing_mode': 'summary'
    },
    'output': {
        'directory': 'output',
        'create_report': True
    },
    'logging': {
        'level': 'info',
        'verbose': False
    },
    'rag': {
        'persist_directory': "vector_db",
        'collection_name': "multimodal_collection",
        'embedding_model': "sentence-transformers/all-MiniLM-L6-v2",
        'context_chunk_count': 10
    }
}

class Config:
    """Configuration management class (Singleton)"""
    _instance = None

    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
            cls._instance._init_path = config_path
        return cls._instance

    def __init__(self, config_path: str = None):
        if getattr(self, '_initialized', False):
            return
        self.config_path = config_path or getattr(self, '_init_path', None) or 'config/config.yml'
        self.config = DEFAULT_CONFIG.copy()
        self.load_config()
        self._initialized = True
    
    def load_config(self):
        config_file = Path(self.config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    self._merge_config(self.config, loaded_config)
                print(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                print(f"Error loading config: {e}")
        else:
            self.save_config()
            print(f"Default configuration created at {self.config_path}")
    
    def save_config(self):
        try:
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def _merge_config(self, base: Dict, update: Dict):
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, path: str, default=None):
        keys = path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, path: str, value: Any):
        keys = path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
