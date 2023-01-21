"""Main file to run the training process"""
from typing import Any, Dict
import argparse
from rlpackage.policy import load_policy
from rlpackage.environment import create_env, EnvInfo
from rlpackage.replay_buffer import ReplayBuffer
from rlpackage.configs import load_config
from rlpackage.loggers import create_logger

def training_loop(config:Dict[str, Any]) -> None:
    """Training loop"""
    #Create the env and utilities to train
    env = create_env(config["env"])
    env_info = EnvInfo.from_env(env)
    policy = load_policy(env_info, config["policy"])
    policy_mem_req = policy.get_mem_req() #Return the information needed for the replay buffer
    replay_buffer = ReplayBuffer(env_info, config["replay_buffer"], policy_mem_req)
    logger = create_logger(config)

    obs = env.reset()
    #Main loop
    for step in range(config["main"]["training_steps"]):
        action_dict = policy.act(obs)
        next_obs, rew, done, info = env.step(action_dict["action"])
        replay_buffer.store(obs, action_dict["action"], rew, done, next_obs)
        if replay_buffer.size > policy.batch_size:
            sample = replay_buffer.sample(policy.batch_size)
            info_policy = policy.train(sample)
            info_policy["replay_buffer"] = replay_buffer.size
            logger.log_step(step, info_policy, testing=False)
        obs = next_obs
        if isinstance(done, bool) and done:
            obs = env.reset()
        if step >0 and step % config["main"]["t_checkpoint"] == 0:
            print(f"step: {step}")
    logger.close_logger()

def testing_loop(config: Dict[str, Any]):
    """Testing loop"""
    raise NotImplementedError()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="path to config file")
    args = parser.parse_args()
    _all_configs = load_config(args.config) #From yaml or argparse

    if _all_configs["main"]["training"]:
        training_loop(_all_configs)
    else:
        testing_loop(_all_configs)
    print("Finished")
