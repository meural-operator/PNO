import yaml
import argparse
from typing import Dict, Any

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args_and_config():
    """
    Parses CLI arguments strictly for '--config' and returns the YAML config 
    merged with any critical runtime overrides if needed later.
    """
    parser = argparse.ArgumentParser(description="PAdic Neural Operator Runner")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint.pth to resume training from")
    
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    
    # Store the path in config for replication purposes
    config['config_path'] = args.config
    config['resume'] = args.resume
    return config
