"""Create dict containing all the configs"""
import yaml

from rlpackage.environment import all_configs_env
from rlpackage.replay_buffer import all_configs_replay_buffer
from rlpackage.loggers import all_configs_logger
from rlpackage.policy import all_configs_policy
from rlpackage.config_base import Config, AllConfigs

all_configs_main = AllConfigs([
    Config(name="training",
           config_type=bool,
           config_help="To toggle training or testing",
           default_val=True,
           ),
    Config(name="training_steps",
           config_type=int,
           config_help="The number of training steps",
           default_val=100000,
           ),
    Config(name="t_checkpoint",
           config_type=int,
           config_help="The frequence of checkpoints",
           default_val=500,
           ),
])

def load_config(yaml_path:str):
    """return dict containing the configs"""

    with open(yaml_path, 'r', encoding="utf-8") as file:
        configs = yaml.safe_load(file)

    all_configs_name = {
        'env': all_configs_env,
        'replay_buffer': all_configs_replay_buffer,
        'logger': all_configs_logger,
        'policy': all_configs_policy,
        'main': all_configs_main,
    }
    for param_key in configs:
        if param_key not in all_configs_name:
            raise ValueError(f"{param_key} is not the right name for the class of parameters")
        configs[param_key] = all_configs_name[param_key].create_config(configs[param_key])
    return configs
