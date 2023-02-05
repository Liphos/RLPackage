"""Main file to run the training process"""
import argparse
import cProfile
import pstats
from typing import Any, Dict

import torch

from rlpackage.configs import load_config
from rlpackage.environment import EnvInfo, create_env
from rlpackage.loggers import LoggerWrapper, create_logger
from rlpackage.policy import Policy, load_policy
from rlpackage.replay_buffer import create_replay_buffer


def training_loop(config:Dict[str, Any]) -> None:
    """Training loop"""
    #Create the env and utilities to train
    env = create_env(config["env"])
    env_info = EnvInfo.from_env(env)
    policy = load_policy(env_info, config["policy"])
    policy_mem_req = policy.get_mem_req() #Return the information needed for the replay buffer
    replay_buffer = create_replay_buffer(env_info, config["replay_buffer"], policy_mem_req)
    logger = create_logger(config)

    obs = env.reset()
    #Main loop
    for step in range(config["main"]["training_steps"]):
        with torch.no_grad():
            action_dict = policy.act(obs)
        next_obs, rew, done, info = env.step(action_dict["action"])
        if "act_log_prob" in action_dict:
            replay_buffer.store(obs, action_dict["action"], rew, done, next_obs, action_dict["act_log_prob"])
        else:
            replay_buffer.store(obs, action_dict["action"], rew, done, next_obs)
        if replay_buffer.train_cond():
            sample = replay_buffer.sample()
            info_policy = policy.train(sample)
            info_policy["replay_buffer"] = replay_buffer.size
            logger.log_step(step, info_policy, testing=False)

        obs = next_obs
        if isinstance(done, bool) and done:
            obs = env.reset()
        if step >0 and step % config["main"]["t_checkpoint"] == 0:
            cumul_reward = test_episode(policy, logger, config["env"], step)
            print(f"step: {step}, reward: {cumul_reward}")
    logger.close_logger()

def test_episode(policy: Policy,
                 logger:LoggerWrapper,
                 config:Dict[str, Any],
                 step:int) -> float:
    """Test episode during training"""
    config["num_agents"] = 1 #Only env to make it lighter
    env = create_env(config)
    obs = env.reset()
    policy.set_eval_mode()
    done = False
    cumul_reward = 0
    while not done:
        action_dict = policy.act(obs, deterministic=True)
        next_obs, rew, done, info = env.step(action_dict["action"])
        cumul_reward += rew
        obs = next_obs

    logger.log_step(step=step, info={"reward": cumul_reward}, testing=True)
    policy.set_train_mode()
    return cumul_reward

def testing_loop(config: Dict[str, Any]):
    """Testing loop"""
    raise NotImplementedError()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="path to config file")
    args = parser.parse_args()
    _all_configs = load_config(args.config) #From yaml or argparse
    print("=======================================================================================")

    if _all_configs["main"]["device"] != "cpu":
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(torch.device(_all_configs["main"]["device"]))))
    else:
        print("Device set to : cpu")

    print("=======================================================================================")

    if _all_configs["main"]["training"]:
        with cProfile.Profile() as pr:
            training_loop(_all_configs)
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.dump_stats(filename='profiling_train.prof')
    else:
        testing_loop(_all_configs)
    print("Finished")
