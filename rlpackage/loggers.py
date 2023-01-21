"""Class for the logger"""
from typing import Dict, Any
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
# pylint: disable=wrong-import-position
from torch.utils.tensorboard import SummaryWriter
import wandb
from rlpackage.config_base import AllConfigs, Config
# pylint: enable=wrong-import-position

class LoggerWrapper():
    """General Wrapper for the loggers"""
    def __init__(self, name:str):
        raise NotImplementedError()
    def log_step(self, step:int, info:Dict[str, Any], testing:bool):
        """log step given the info"""
        raise NotImplementedError()
    def close_logger(self):
        """Execute commands when finishing"""
        pass

class WandbWrapper(LoggerWrapper):
    """Tensorboard wrapper"""
    def __init__(self, name:str, configs:Dict[str, Any]):
        wandb.init(project="RlPackage", name=name, config=configs)
    def log_step(self, step:int, info:Dict[str, Any], testing:bool):
        is_testing_prefix = "Test" if testing else "Train"
        info_copy = {}
        for info_key, info_val in info.items():
            info_copy[f"{is_testing_prefix}_{info_key}"] = info_val
        wandb.log(info_copy, step=step)
    def close_logger(self):
        wandb.finish(0)
class TensorboardWrapper(LoggerWrapper):
    """Tensorboard wrapper"""
    def __init__(self, name:str):
        self.writer = SummaryWriter("Tensorboard/"+name)
    def log_step(self, step:int, info:Dict[str, Any], testing:bool):
        for metric, metric_val in info.items():
            is_testing_prefix = "Test" if testing else "Train"
            self.writer.add_scalar(f"{is_testing_prefix}/{metric}", metric_val, step)
    def close_logger(self):
        self.writer.flush()
        self.writer.close()

def create_logger(config:Dict[str, Any]) -> LoggerWrapper:
    """Create the logger"""
    logger_config = config["logger"]
    if logger_config["logger"] == "wandb":
        logger = WandbWrapper(logger_config["name"], config) # type: LoggerWrapper
    elif logger_config["logger"] == "tensorboard":
        logger = TensorboardWrapper(logger_config["name"])
    else:
        raise ValueError("logger not recognized")
    return logger

all_configs_logger = AllConfigs([
    Config(name="logger",
           config_type=str,
           config_help="The logger to use",
           default_val="wandb"),
    Config(name="name",
           config_type=str,
           config_help="The name of the run",
           default_val="debug"),
])