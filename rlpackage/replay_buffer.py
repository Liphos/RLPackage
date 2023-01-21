"""File containing all datatypes used to store the episode data"""
from typing import Union, Tuple, Dict, Optional, Any
import numpy as np
from rlpackage.environment import EnvInfo
from rlpackage.config_base import AllConfigs, Config

class Sample():
    """Base class for a sample"""
    def __init__(self, replay_buffer:"ReplayBuffer", indicies: Union[np.ndarray, Tuple[int, int]]):
        if isinstance(indicies, np.ndarray):
            self.obs = replay_buffer.obs_arr[indicies]
            self.act = replay_buffer.act_arr[indicies]
            self.rew = replay_buffer.rew_arr[indicies]
            self.done = replay_buffer.done_arr[indicies]
            self.next_obs = replay_buffer.next_obs_arr[indicies]

            if replay_buffer.act_log_prob_arr is not None:
                self.act_log_prob = replay_buffer.act_log_prob_arr[indicies]
        elif isinstance(indicies, tuple):
            self.obs = replay_buffer.obs_arr[indicies[0]: indicies[1]]
            self.act = replay_buffer.act_arr[indicies[0]: indicies[1]]
            self.rew = replay_buffer.rew_arr[indicies[0]: indicies[1]]
            self.done = replay_buffer.done_arr[indicies[0]: indicies[1]]
            self.next_obs = replay_buffer.next_obs_arr[indicies[0]: indicies[1]]

            if replay_buffer.act_log_prob_arr is not None:
                self.act_log_prob = replay_buffer.act_log_prob_arr[indicies[0]: indicies[1]]
        else:
            raise TypeError("The indicies for the samples do not have the good type.")

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

class ReplayBuffer():
    """Replay Buffer class"""
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
        if "action_log_prob" in pol_mem_req:
            self.act_log_prob_arr = np.zeros((self.max_size, ) + self.act_dim)

        #Create EpisodeUnit
        self.episode_units = [EpisodeUnit() for _ in range(self.num_agents)]

    def store(self, obs: np.ndarray,
              act: np.ndarray,
              rew: Union[float, np.ndarray],
              done: Union[bool, np.ndarray],
              next_obs: np.ndarray,
              act_log_prob: Optional[np.ndarray]=None) -> None:
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

    def sample(self, sample_size:int) -> Sample:
        """Return a Sample object of size 'sample_size' containing random information"""
        indicies = np.arange(self.size)
        np.random.shuffle(indicies)
        if self.size < sample_size:
            print("WARNING: try to sample with a sample size higher than amount of saved info")
        else:
            indicies = indicies[:sample_size]
        return Sample(self, indicies)

    def reset(self) -> None:
        """Reset the replay_buffer and episode_units"""
        self.size = 0
        self.ptr = 0
        self.episode_units = [EpisodeUnit() for _ in range(self.num_agents)]

all_configs_replay_buffer = AllConfigs([
    Config(name="max_size",
           config_type=int,
           config_help="Maximum size of the replay_buffer",
           default_val=10000)
])
