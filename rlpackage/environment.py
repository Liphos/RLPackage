"""Define env utils functions"""
from typing import Set, Dict, Any
import gym
from gym.spaces import Discrete, Box
import numpy as np
from rlpackage.config_base import AllConfigs, Config
UNITY_ENVS: Set[str] = set()

class GymWrapper(gym.Wrapper):
    """Wrapper to encapsulate the informations"""
    def reset(self):
        obs = self.env.reset()
        return [obs]

    def step(self, action:np.ndarray):
        obs, rew, done, info = self.env.step(action)
        return [obs], [rew], [done], info

def create_env(config:Dict[str, Any]) -> gym.Env:
    """Create the environment given the name and the number of instances"""
    env_name = config["env_name"]
    n_envs = config["num_agents"]
    if env_name not in UNITY_ENVS:
        if n_envs == 1:
            env = gym.make(env_name)
        else:
            env = gym.vector.AsyncVectorEnv([lambda:gym.make(env_name) for _ in range(n_envs)])
    else:
        raise NotImplementedError("Unity environment not supported yet")
    return env

class EnvInfo():
    """
    To store the information about the environment
    """
    def __init__(self,
                 action_space: gym.Space,
                 observation_space: gym.Space,
                 async_env:bool=False,
                 ):

        if not isinstance(observation_space, Box):
            raise NotImplementedError("Observation space outside of Box is not supported yet")

        if not async_env:
            self.observation_space = observation_space
            self.action_space = action_space
            self.num_agents = 1
        else:
            self.observation_space = Box(high=observation_space.high[0],
                                         low=observation_space.low[0],
                                         shape=observation_space.shape[1:]
                                         )
            self.action_space = action_space[0]
            self.num_agents = observation_space.shape[0]

        if isinstance(self.action_space, Discrete):
            self.act_dim = ()
        elif isinstance(self.action_space, Box):
            self.act_dim = self.action_space.shape
        else:
            raise TypeError("action_space don't have a supported type")

        self.obs_dim = self.observation_space.shape

        self.async_env = async_env

    @staticmethod
    def from_env(env:gym.Env):
        """Create envInfo object from env"""
        return EnvInfo(env.action_space,
                       env.observation_space,
                       async_env=isinstance(env, gym.vector.VectorEnv),
                       )

all_configs_env = AllConfigs([
    Config(name="env_name",
           config_type=str,
           config_help="Name of the environment to initialize",
           default_val="CartPole-v1",
           ),
    Config(name="num_agents",
           config_type=int,
           config_help="Number of envs to launch",
           default_val=1,
           ),
])
