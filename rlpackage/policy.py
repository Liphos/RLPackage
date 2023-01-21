"""Define policies"""
from typing import Dict, Any
from abc import ABC, abstractmethod
import torch
from rlpackage.environment import EnvInfo
from rlpackage.replay_buffer import Sample
from rlpackage.config_base import AllConfigs, Config

def freeze(model:torch.nn.Module):
    """Freeze Model parameters

    Args:
        model (torch.nn.Module): the model to freeze
    """

    for param in model.parameters():
        param.requires_grad = False

def load_policy(env_info:EnvInfo, config:Dict[str, Any]) -> "Policy":
    """Load the policy given the name"""
    if config["algo"] == "RandomPolicy":
        return RandomPolicy(env_info, config)
    elif config["algo"] == "":
        raise NotImplementedError("DQN not implemented yet")
    else:
        raise ValueError("This algo is not implemented.")

class Policy(ABC):
    """Base policy"""
    def __init__(self,
                 env_info:EnvInfo,
                 config:Dict[str, Any]):

        self.env_info = env_info
        self.action_space = env_info.action_space
        self.observation_space = env_info.observation_space
        self.num_agents = env_info.num_agents
        self.batch_size = config["batch_size"]

    @abstractmethod
    def act(self, observation:torch.Tensor, deterministic:bool=False):
        """function that given a state returns the action"""
        pass

    @abstractmethod
    def train(self, sample:Sample) -> Dict[str, Any]:
        """Sample from the replay buffer to train the policy"""
        pass

    def get_mem_req(self) -> Dict:
        """Get mem req"""
        return {}

class RandomPolicy(Policy):
    """Random Agent"""
    def act(self, observation:torch.Tensor, deterministic:bool=False) -> Dict[str, Any]:
        if self.num_agents==1:
            return {"action":self.action_space.sample()}

        return {"action": [self.action_space.sample() for _ in range(self.num_agents)]}

    def train(self, sample:Sample) -> Dict[str, Any]:
        """There is no need to train"""
        return {}

all_configs_policy = AllConfigs([
    Config(name="algo",
           config_type=str,
           config_help="The algo to use",
           default_val="RandomPolicy"),
    Config(name="batch_size",
           config_type=int,
           config_help="The batch_size to use",
           default_val=64),

])