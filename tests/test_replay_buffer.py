"""Unitary test for the replay buffer"""
from typing import Dict, Union, Set
import random
import numpy as np
import pytest
import gym
from rlpackage.environment import EnvInfo
from rlpackage.replay_buffer import create_replay_buffer



@pytest.mark.parametrize("env, config, pol_mem_req",
                         [(gym.make("CartPole-v1"), {"max_size":11}, {"type":"array", "batch_size": 16}),
                          (gym.make("LunarLanderContinuous-v2"), {"max_size":11}, {"type":"array", "batch_size": 16}),
                          (gym.vector.AsyncVectorEnv([
                              lambda: gym.make("Pendulum-v1", g=9.81) for _ in range(10)]), {"max_size":11}, {"type":"array", "batch_size": 16}),
                          ])
def test_create_replay_buffer(env:gym.Env,
                              config:Dict[str, Union[bool, int, str]],
                              pol_mem_req:Set):
    """Test replay buffer functions"""
    env_info = EnvInfo.from_env(env)
    replay_buffer = create_replay_buffer(env_info=env_info,
                                      config=config,
                                      pol_mem_req=pol_mem_req)
    assert replay_buffer.size == 0
    assert replay_buffer.ptr == 0

    obs = env.observation_space.sample()
    act = env.action_space.sample()
    if env_info.num_agents == 1:
        rew = random.random()
        done = False
    else:
        rew = np.array([random.random() for _ in range(env_info.num_agents)])
        done = np.array([False for _ in range(env_info.num_agents)])
    replay_buffer.store(obs, act, rew, done, obs)
    assert replay_buffer.size == 0
    assert replay_buffer.ptr == 0

    obs = env.observation_space.sample()
    act = env.action_space.sample()
    if env_info.num_agents == 1:
        rew = random.random()
        done = True
    else:
        rew = np.array([random.random() for _ in range(env_info.num_agents)])
        done = np.array([True for _ in range(env_info.num_agents)])
    replay_buffer.store(obs, act, rew, done, obs)
    assert replay_buffer.size == min(2 * env_info.num_agents, replay_buffer.max_size)
    assert replay_buffer.ptr == (2 * env_info.num_agents) % replay_buffer.max_size
    _ = replay_buffer.sample()
    replay_buffer.reset()
    assert replay_buffer.size == 0
    assert replay_buffer.ptr == 0
    assert len(replay_buffer.episode_units[0].obs_arr) == 0