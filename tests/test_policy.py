"""Utilitaries tests for policy"""
from typing import Dict, Any
import gym
import pytest
from rlpackage.environment import EnvInfo
from rlpackage.policy import RandomPolicy

@pytest.mark.parametrize("env, config", [(gym.make("CartPole-v1"), {"batch_size":10}),
                                         (gym.vector.AsyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(5)]),
                                                                   {"batch_size":10})])
def test_policy(env:gym.Env, config:Dict[str, Any]):
    """test goven policy"""
    env_info = EnvInfo.from_env(env)
    policy = RandomPolicy(env_info, config=config)
    obs = env.reset()
    action = policy.act(obs, deterministic=False)
    obs, _, _, _ = env.step(action["action"])
