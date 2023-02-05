"""File containing all datatypes used to store the episode data"""
from typing import Union, Tuple, Dict, Optional, Any
from abc import ABC, abstractmethod
import numpy as np
import torch
from rlpackage.environment import EnvInfo
from rlpackage.config_base import AllConfigs, Config

def create_replay_buffer(env_info: EnvInfo, config: Dict[str, int], pol_mem_req: Dict[str, Any]):
    """Return Buffer"""
    if pol_mem_req["type"] == "torch":
        return TorchReplayBuffer(env_info, config, pol_mem_req)
    elif pol_mem_req["type"] == "array":
        return ArrayReplayBuffer(env_info, config, pol_mem_req)
    else:
        raise ValueError(pol_mem_req["type"] + " is not a correct buffer")

class Sample(ABC):
    """Base class for sample"""
    @abstractmethod
    def __init__(self, replay_buffer:"ReplayBuffer"):
        pass
    @abstractmethod
    def to_torch_tensor(self, device:str):
        """transform to torch tensor"""
        pass

class ArraySample(Sample):
    """Base class for an array sample"""
    def __init__(self, replay_buffer:"ArrayReplayBuffer",
                 indicies: Union[np.ndarray, Tuple[int, int]]):
        if isinstance(indicies, np.ndarray):
            self.obs = replay_buffer.obs_arr[indicies]
            self.act = replay_buffer.act_arr[indicies]
            self.rew = replay_buffer.rew_arr[indicies]
            self.done = replay_buffer.done_arr[indicies]
            self.next_obs = replay_buffer.next_obs_arr[indicies]

            self.act_log_prob = None
            if replay_buffer.act_log_prob_arr is not None:
                self.act_log_prob = replay_buffer.act_log_prob_arr[indicies]
        elif isinstance(indicies, tuple):
            self.obs = replay_buffer.obs_arr[indicies[0]: indicies[1]]
            self.act = replay_buffer.act_arr[indicies[0]: indicies[1]]
            self.rew = replay_buffer.rew_arr[indicies[0]: indicies[1]]
            self.done = replay_buffer.done_arr[indicies[0]: indicies[1]]
            self.next_obs = replay_buffer.next_obs_arr[indicies[0]: indicies[1]]

            self.act_log_prob = None
            if replay_buffer.act_log_prob_arr is not None:
                self.act_log_prob = replay_buffer.act_log_prob_arr[indicies[0]: indicies[1]]
        else:
            raise TypeError("The indicies for the samples do not have the good type.")

    def to_torch_tensor(self, device:str):
        """Transform sample numpy to torch"""
        self.obs = torch.as_tensor(self.obs, dtype=torch.float32, device=device)
        self.act = torch.as_tensor(self.act, dtype=torch.float32, device=device)
        self.rew = torch.as_tensor(self.rew, dtype=torch.float32, device=device)
        self.done = torch.as_tensor(self.done, dtype=torch.long, device=device)
        self.next_obs = torch.as_tensor(self.next_obs, dtype=torch.float32, device=device)
        if self.act_log_prob is not None:
            self.act_log_prob = torch.as_tensor(self.act_log_prob,
                                                dtype=torch.float32,
                                                device=device)

class TorchSample(Sample):
    """Sample from the gpu on policy buffer"""
    def __init__(self, replay_buffer:"TorchReplayBuffer"):
        self.obs = replay_buffer.obs_arr
        self.act = replay_buffer.act_arr
        self.rew = replay_buffer.rew_arr
        self.done = replay_buffer.done_arr
        self.next_obs = replay_buffer.next_obs
        self.next_done = replay_buffer.next_done

        self.act_log_prob = None
        if replay_buffer.act_log_prob_arr is not None:
            self.act_log_prob = replay_buffer.act_log_prob_arr

    def to_torch_tensor(self, device: str):
        pass

class EpisodeUnit():
    """
    Type to store an episode
    """
    def __init__(self):
        self.obs_arr = []
        self.act_arr = []
        self.rew_arr = []
        self.done_arr = []
        self.next_obs_arr = []

        #Optional storage
        self.act_log_prob = []

    def append(self, obs, act, rew, done, next_obs, act_log_prob=None):
        """Append set of info to the EpisodeUnit"""
        self.obs_arr.append(obs)
        self.act_arr.append(act)
        self.rew_arr.append(rew)
        self.done_arr.append(done)
        self.next_obs_arr.append(next_obs)

        if act_log_prob is not None:
            self.act_log_prob.append(act_log_prob)

    def to_numpy(self):
        """Convert to numpy"""
        self.obs_arr = np.asarray(self.obs_arr, dtype=np.float32)
        self.act_arr = np.asarray(self.act_arr, dtype=np.float32)
        self.rew_arr = np.asarray(self.rew_arr, dtype=np.float32)
        self.done_arr = np.asarray(self.done_arr, dtype=np.float32)
        self.next_obs_arr = np.asarray(self.next_obs_arr, dtype=np.float32)

        self.act_log_prob = np.asarray(self.act_log_prob, dtype=np.float32)

class ReplayBuffer(ABC):
    """Base class for replay buffer"""
    @abstractmethod
    def __init__(self,
                 env_info:EnvInfo,
                 config:Dict[str, int],
                 pol_mem_req:Dict[str, Any]):
        pass
    @abstractmethod
    def store(self, obs: np.ndarray,
              act: np.ndarray,
              rew: Union[float, np.ndarray],
              done: Union[bool, np.ndarray],
              next_obs: np.ndarray,
              act_log_prob: Optional[torch.Tensor]=None) -> None:
        """store elements"""

    @abstractmethod
    def sample(self):
        """sample elements from buffer"""

    @abstractmethod
    def reset(self) -> None:
        """reset buffer"""

    @abstractmethod
    def train_cond(self):
        """Return true if there is enough data for training"""


class TorchReplayBuffer(ReplayBuffer):
    """Small buffer set on selected device. Meant for ON-Policy"""
    def __init__(self, env_info: EnvInfo, config: Dict[str, int], pol_mem_req: Dict[str, Any]):
        #Load pointers and arrays dim
        self.max_size = config["max_size"]
        self.size = 0
        self.act_dim = env_info.act_dim
        self.obs_dim = env_info.obs_dim
        self.num_agents = env_info.num_agents
        self.device = torch.device(config["device"])

        #Create arrays
        self.obs_arr = torch.zeros((self.max_size, self.num_agents) + self.obs_dim, device=self.device)
        self.act_arr = torch.zeros((self.max_size, self.num_agents) + self.act_dim, device=self.device)
        self.rew_arr = torch.zeros([self.max_size, self.num_agents], device=self.device)
        self.done_arr = torch.zeros([self.max_size, self.num_agents], dtype=torch.long, device=self.device)
        self.act_log_prob_arr = None
        if "act_log_prob" in pol_mem_req:
            self.act_log_prob_arr = torch.zeros([self.max_size, self.num_agents], device=self.device)

        self.next_obs = torch.as_tensor((self.num_agents,) + self.obs_dim, device=self.device)
        self.next_done = torch.as_tensor((self.num_agents,), dtype=torch.long, device=self.device)

    def store(self, obs: np.ndarray,
              act: Union[int, np.ndarray],
              rew: Union[float, np.ndarray],
              done: Union[bool, np.ndarray],
              next_obs: np.ndarray,
              act_log_prob: Optional[torch.Tensor] = None) -> None:
        if self.size == 0:
            self.obs_arr[self.size] = torch.as_tensor(obs, device=self.device)
            self.done_arr[self.size] = torch.zeros((self.num_agents,), device=self.device)
        else:
            self.obs_arr[self.size] = self.next_obs
            self.done_arr[self.size] = self.next_done

        self.act_arr[self.size] = torch.as_tensor(act, device=self.device)
        if act_log_prob is not None:
            self.act_log_prob_arr[self.size] = act_log_prob
        self.rew_arr[self.size] = torch.as_tensor(rew, device=self.device)

        self.next_obs = torch.as_tensor(next_obs, device=self.device)
        self.next_done = torch.as_tensor(done, dtype=torch.long, device=self.device)

        self.size +=1

    def sample(self):
        sample = TorchSample(self)
        self.reset() #We reset the buffer after each training step
        return sample

    def reset(self):
        #We just need to reset the pointer
        self.size = 0

    def train_cond(self):
        """Return true if there is enough data for training"""
        return self.size == self.max_size

class ArrayReplayBuffer(ReplayBuffer):
    """Array Replay Buffer class"""
    def __init__(self,
                 env_info:EnvInfo,
                 config:Dict[str, int],
                 pol_mem_req:Dict[str, Any]):

        #Load pointers and arrays dim
        self.max_size = config["max_size"]
        self.size, self.ptr = 0, 0
        self.act_dim = env_info.act_dim
        self.obs_dim = env_info.obs_dim
        self.num_agents = env_info.num_agents

        #Create arrays
        self.obs_arr = np.zeros((self.max_size, ) + self.obs_dim)
        self.act_arr = np.zeros((self.max_size, ) + self.act_dim)
        self.rew_arr = np.zeros((self.max_size, ))
        self.done_arr = np.zeros((self.max_size, ))
        self.next_obs_arr = np.zeros((self.max_size, ) + self.obs_dim)
        self.act_log_prob_arr = None
        if "act_log_prob" in pol_mem_req:
            self.act_log_prob_arr = np.zeros((self.max_size, ) + self.act_dim)

        self.sample_size = pol_mem_req["batch_size"]

        #Create EpisodeUnit
        self.episode_units = [EpisodeUnit() for _ in range(self.num_agents)]

    def store(self, obs: np.ndarray,
              act: np.ndarray,
              rew: Union[float, np.ndarray],
              done: Union[bool, np.ndarray],
              next_obs: np.ndarray,
              act_log_prob: Optional[torch.Tensor]=None) -> None:
        """Store the set of information inside the Episode Unit"""
        #Encapsulate the results for an env with only 1 agent
        if self.num_agents == 1:
            if act_log_prob is not None:
                self.episode_units[0].append(obs, act, rew, done, next_obs, act_log_prob)
            else:
                self.episode_units[0].append(obs, act, rew, done, next_obs)
            if done:
                self.store_episode(self.episode_units[0])
                self.episode_units[0] = EpisodeUnit()
        else:
            assert isinstance(rew, np.ndarray)
            assert isinstance(done, np.ndarray)
            for incr_agent in range(self.num_agents):
                if act_log_prob is not None:
                    self.episode_units[incr_agent].append(obs[incr_agent],
                                                        act[incr_agent],
                                                        rew[incr_agent],
                                                        done[incr_agent],
                                                        next_obs[incr_agent],
                                                        act_log_prob[incr_agent])
                else:
                    self.episode_units[incr_agent].append(obs[incr_agent],
                                                        act[incr_agent],
                                                        rew[incr_agent],
                                                        done[incr_agent],
                                                        next_obs[incr_agent])
                if done[incr_agent]:
                    self.store_episode(self.episode_units[incr_agent])
                    self.episode_units[incr_agent] = EpisodeUnit()

    def store_episode(self, episode:EpisodeUnit) -> None:
        """Store the information of an episode inside the replay_buffer"""
        episode.to_numpy()
        if self.ptr + len(episode.obs_arr)>self.max_size:
            next_idx = len(episode.obs_arr) - (self.max_size-self.ptr)
            self.obs_arr[self.ptr: self.max_size] = episode.obs_arr[:self.max_size-self.ptr]
            self.act_arr[self.ptr: self.max_size] = episode.act_arr[:self.max_size-self.ptr]
            self.rew_arr[self.ptr: self.max_size] = episode.rew_arr[:self.max_size-self.ptr]
            self.done_arr[self.ptr: self.max_size] = episode.done_arr[:self.max_size-self.ptr]
            self.next_obs_arr[self.ptr: self.max_size] = episode.next_obs_arr[:self.max_size-self.ptr]

            self.obs_arr[:next_idx] = episode.obs_arr[self.max_size-self.ptr:]
            self.act_arr[:next_idx] = episode.act_arr[self.max_size-self.ptr:]
            self.rew_arr[:next_idx] = episode.rew_arr[self.max_size-self.ptr:]
            self.done_arr[:next_idx] = episode.done_arr[self.max_size-self.ptr:]
            self.next_obs_arr[:next_idx] = episode.next_obs_arr[self.max_size-self.ptr:]

            if len(episode.act_log_prob)>0:
                if self.act_log_prob_arr is None:
                    raise ValueError("Tried to store act log prob but it was not provided during init")

                self.act_log_prob_arr[self.ptr: self.max_size] = episode.act_log_prob[:self.max_size-self.ptr]
                self.act_log_prob_arr[:next_idx] = episode.act_log_prob[self.max_size-self.ptr:]

            self.size = self.max_size
            self.ptr = next_idx

        else:
            next_idx = self.ptr + len(episode.obs_arr)
            self.obs_arr[self.ptr: next_idx] = episode.obs_arr
            self.act_arr[self.ptr: next_idx] = episode.act_arr
            self.rew_arr[self.ptr: next_idx] = episode.rew_arr
            self.done_arr[self.ptr: next_idx] = episode.done_arr
            self.next_obs_arr[self.ptr: next_idx] = episode.next_obs_arr

            if len(episode.act_log_prob)>0:
                if self.act_log_prob_arr is None:
                    raise ValueError("Tried to store act log prob but it was not provided during init")

                self.act_log_prob_arr[self.ptr: next_idx] = episode.act_log_prob


            self.size += len(episode.obs_arr)
            self.ptr += len(episode.obs_arr)

            self.size = np.minimum(self.size, self.max_size)
            self.ptr = self.ptr % self.max_size

    def sample(self) -> ArraySample:
        """Return a Sample object of size 'sample_size' containing random information"""
        indicies = np.random.choice(self.size, size=self.sample_size)
        if self.size < self.sample_size:
            print("WARNING: try to sample with a sample size higher than amount of saved info")
        else:
            indicies = indicies[:self.sample_size]
        return ArraySample(self, indicies)

    def reset(self) -> None:
        """Reset the replay_buffer and episode_units"""
        self.size = 0
        self.ptr = 0
        self.episode_units = [EpisodeUnit() for _ in range(self.num_agents)]

    def train_cond(self):
        """Return true if there is enough data for training"""
        return self.sample_size<=self.size

all_configs_replay_buffer = AllConfigs([
    Config(name="max_size",
           config_type=int,
           config_help="Maximum size of the replay_buffer",
           default_val=10000)
])
