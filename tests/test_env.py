"""Unitary tests of environment utilities"""
from typing import Tuple
import pytest
import gym
from rlpackage.environment import create_env, EnvInfo

@pytest.mark.parametrize("env_name, n_envs", [("CartPole-v1", 1),
                                              ("LunarLanderContinuous-v2", 10)])
def test_create_env(env_name:str, n_envs:int):
    """Test the creation of the environment"""
    config = {"env_name": env_name, "n_envs": n_envs}
    env = create_env(config)
    assert isinstance(env, gym.Env) if n_envs == 1 else isinstance(env, gym.vector.VectorEnv)
    obs = env.reset()
    obs, _ , _, _ = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape


@pytest.mark.parametrize("env, obs_dim, act_dim",
                         [(gym.make("CartPole-v1"), (4,), ()),
                          (gym.make("LunarLanderContinuous-v2"), (8,), (2,)),
                          (gym.vector.AsyncVectorEnv([
                              lambda: gym.make("CartPole-v1"),
                              lambda: gym.make("CartPole-v1")]), (4,), ()),
                          (gym.vector.AsyncVectorEnv([
                              lambda: gym.make("Pendulum-v1", g=9.81) for _ in range(10)]), (3,), (1,)),
                          ])
def test_envinfo(env:gym.Env, obs_dim:Tuple, act_dim:Tuple):
    """Test creation of EnvInfo object"""
    env.reset()
    env_info = EnvInfo.from_env(env)
    assert env_info.obs_dim == obs_dim
    assert env_info.act_dim == act_dim

    if env_info.num_agents == 1:
        assert env.step(env_info.action_space.sample())
        obs= env.reset()
        assert obs.shape == env_info.observation_space.shape
    else:
        assert env.step([env_info.action_space.sample() for _ in range(env_info.num_agents)])
        obs = env.reset()
        assert obs[0].shape == env_info.observation_space.shape