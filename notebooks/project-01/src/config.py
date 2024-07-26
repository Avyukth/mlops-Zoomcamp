import toml
import os
from pathlib import Path
from dotenv import load_dotenv

class Config:
    def __init__(self, config_path="config/config.toml", env_path=".env"):
        self.config_path = Path(__file__).resolve().parent.parent / config_path
        self.env_path = Path(__file__).resolve().parent.parent / env_path
        self.load_config()

    def load_config(self):
        print(f"Attempting to load config from: {self.config_path.resolve()}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path.resolve()}")
        
        # Load .env file
        load_dotenv(self.env_path)
        
        # Load TOML config
        with open(self.config_path, 'r') as config_file:
            self._config_data = toml.load(config_file)

        # Merge environment variables into config
        for key, value in os.environ.items():
            self._config_data[key] = value

        # Set attributes
        for key, value in self._config_data.items():
            setattr(self, key.upper(), value)
        
        print("Config loaded successfully")

    def __getitem__(self, key):
        return self._config_data.get(key, os.getenv(key))

    def __str__(self):
        config_str = "Loaded Configuration:\n"
        for key, value in self._config_data.items():
            if not key.startswith('_'):  # Exclude private attributes
                config_str += f"{key.upper()}: {value}\n"
        return config_str

# Create a global instance of Config
config = Config()
