"""
Configuration management for BERT News Recommender
"""
import os
import yaml
from dataclasses import dataclass, asdict
from typing import Optional, List
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    vocab_size: int = 30522
    d_model: int = 768
    n_layers: int = 12
    heads: int = 12
    dropout: float = 0.1
    seq_len: int = 128
    num_classes: int = 4


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    save_steps: int = 1000
    eval_steps: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0


@dataclass
class DataConfig:
    """Data configuration parameters"""
    dataset_name: str = "ag_news"
    train_split: str = "train"
    test_split: str = "test"
    max_seq_length: int = 128
    tokenizer_name: str = "bert-base-uncased"


@dataclass
class APIConfig:
    """API configuration"""
    news_api_key: str = ""
    wandb_api_key: str = ""
    wandb_project: str = "bert-news-recommender"


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    type: str = "local"  # local, docker, heroku
    port: int = 8501
    host: str = "0.0.0.0"


@dataclass
class PathConfig:
    """File path configuration"""
    model_path: str = "./best_ft_model.pth"
    pretrained_model_path: str = "./bert_model_wiki103.pth"
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    log_dir: str = "./logs"


@dataclass
class AppConfig:
    """Application configuration"""
    debug: bool = False
    log_level: str = "INFO"
    device: str = "auto"  # auto, cpu, cuda
    seed: int = 42


class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "configs/config.yaml"
        
        # Initialize with defaults
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.api = APIConfig()
        self.deployment = DeploymentConfig()
        self.paths = PathConfig()
        self.app = AppConfig()
        
        # Load from file if exists
        if Path(self.config_path).exists():
            self.load_from_yaml()
        
        # Override with environment variables
        self._load_from_env()
    
    def load_from_yaml(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            if 'model' in config_dict:
                self.model = ModelConfig(**config_dict['model'])
            if 'training' in config_dict:
                self.training = TrainingConfig(**config_dict['training'])
            if 'data' in config_dict:
                self.data = DataConfig(**config_dict['data'])
            if 'api' in config_dict:
                self.api = APIConfig(**config_dict['api'])
            if 'deployment' in config_dict:
                self.deployment = DeploymentConfig(**config_dict['deployment'])
            if 'paths' in config_dict:
                self.paths = PathConfig(**config_dict['paths'])
            if 'app' in config_dict:
                self.app = AppConfig(**config_dict['app'])
                
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # API keys
        if os.getenv("NEWS_API_KEY"):
            self.api.news_api_key = os.getenv("NEWS_API_KEY")
        if os.getenv("WANDB_API_KEY"):
            self.api.wandb_api_key = os.getenv("WANDB_API_KEY")
        if os.getenv("WANDB_PROJECT"):
            self.api.wandb_project = os.getenv("WANDB_PROJECT")
        
        # Deployment
        if os.getenv("DEPLOYMENT_TYPE"):
            self.deployment.type = os.getenv("DEPLOYMENT_TYPE")
        if os.getenv("PORT"):
            self.deployment.port = int(os.getenv("PORT"))
        
        # Paths
        if os.getenv("MODEL_PATH"):
            self.paths.model_path = os.getenv("MODEL_PATH")
        
        # App
        if os.getenv("DEBUG"):
            self.app.debug = os.getenv("DEBUG").lower() == "true"
        if os.getenv("LOG_LEVEL"):
            self.app.log_level = os.getenv("LOG_LEVEL")
    
    def save_to_yaml(self):
        """Save current configuration to YAML file"""
        config_dict = {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'api': asdict(self.api),
            'deployment': asdict(self.deployment),
            'paths': asdict(self.paths),
            'app': asdict(self.app)
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_device(self) -> str:
        """Get the appropriate device for computation"""
        if self.app.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.app.device
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check required API keys
        if not self.api.news_api_key:
            errors.append("NEWS_API_KEY is required")
        
        # Check model parameters
        if self.model.d_model % self.model.heads != 0:
            errors.append("d_model must be divisible by heads")
        
        # Check file paths
        if not Path(self.paths.model_path).exists():
            errors.append(f"Model file not found: {self.paths.model_path}")
        
        return errors
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"""
ConfigManager:
  Model: {self.model}
  Training: {self.training}
  Data: {self.data}
  API: {self.api}
  Paths: {self.paths}
  App: {self.app}
"""


# Global configuration instance
config = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global configuration instance"""
    return config


def set_config_path(path: str):
    """Set configuration file path and reload"""
    global config
    config = ConfigManager(path)
