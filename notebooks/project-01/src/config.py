import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = Path(__file__).resolve().parent.parent / config_path
        self.load_config()

    def load_config(self):
        print(f"Attempting to load config from: {self.config_path.resolve()}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path.resolve()}")
        
        with open(self.config_path, 'r') as config_file:
            self._config_data = yaml.safe_load(config_file)
        
        for key, value in self._config_data.items():
            setattr(self, key.upper(), value)
        print("Config loaded successfully")

    def __getitem__(self, key):
        return self._config_data[key]
    
    def __str__(self):
        config_str = "Loaded Configuration:\n"
        for key, value in self._config_data.items():
            config_str += f"{key.upper()}: {value}\n"
        return config_str

# Create a global instance of Config
config = Config()
