"""Class for configs, in another file to avoid circular imports"""
from typing import Any, List, Dict

class AllConfigs():
    """Class containing all configs"""
    def __init__(self, configs: List["Config"]):
        self.configs = configs

    def get_dict(self) -> Dict[str, "Config"]:
        """return config"""
        config_dict = {}
        for config in self.configs:
            config_dict[config.name] = config
        return config_dict

    def create_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Create config given the config dict of the yaml"""
        all_configs = self.get_dict()
        for config in config_dict:
            if config not in all_configs:
                raise ValueError(f"Warning parameter {config} does not exist")
            assert isinstance(config_dict[config], all_configs[config].type)

        for config_key, config_val in all_configs.items():
            if config_key not in config_dict:
                config_dict[config_key] = config_val.default_val
        return config_dict


class Config():
    """Config class containing info for a parameter"""
    def __init__(self, name:str, config_type:Any, config_help:str, default_val:Any):
        self.name = name
        self.type = config_type
        self.help = config_help
        self.default_val = default_val
