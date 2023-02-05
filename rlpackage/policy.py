"""Define policies"""
from typing import Dict, Any, Tuple, Union
from abc import ABC, abstractmethod
from copy import deepcopy
import torch
from torch.distributions.categorical import Categorical
import numpy as np
import gym.spaces
from rlpackage.environment import EnvInfo
from rlpackage.replay_buffer import ArraySample, TorchSample, Sample
from rlpackage.config_base import AllConfigs, Config
from rlpackage.model import MLP

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
    elif config["algo"] == "DQN":
        return DQN(env_info, config)
    elif config["algo"] == "PPODiscrete":
        return PPODiscrete(env_info, config)
    else:
        raise ValueError("This algo is not implemented.")

def compute_gae(sample:TorchSample, critic:torch.nn.Module, gamma:float
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the gae for the given sample"""
    returns = torch.zeros_like(sample.rew).to(sample.rew.device)
    for step in reversed(range(len(sample.rew))):
        if step == len(sample.rew) - 1:
            nextnonterminal = 1.0 - sample.next_done
            next_return = critic(sample.next_obs).squeeze(-1)
        else:
            nextnonterminal = 1.0 - sample.done[step + 1]
            next_return = returns[step + 1]
        returns[step] = sample.rew[step] + gamma * nextnonterminal * next_return
    advantages = returns - critic(sample.obs).squeeze(-1)
    return returns, advantages

class Policy(ABC):
    """Base policy"""
    def __init__(self,
                 env_info:EnvInfo,
                 config:Dict[str, Any]):

        self.action_space = env_info.action_space
        self.observation_space = env_info.observation_space
        self.num_agents = env_info.num_agents
        self.batch_size = config["batch_size"]

    @abstractmethod
    def act(self, observation:torch.Tensor, deterministic:bool=False):
        """function that given a state returns the action"""

    @abstractmethod
    def train(self, sample:Union[TorchSample, ArraySample]) -> Dict[str, Any]:
        """Sample from the replay buffer to train the policy"""

    def get_mem_req(self) -> Dict:
        """Get mem req"""
        return {"type": "array", "batch_size": self.batch_size}

    def set_train_mode(self) -> None:
        """Set models to train mode"""
        pass

    def set_eval_mode(self) -> None:
        """Set models to eval mode"""
        pass

class RandomPolicy(Policy):
    """Random Agent"""
    def act(self, observation:torch.Tensor, deterministic:bool=False) -> Dict[str, Any]:
        if self.num_agents==1:
            return {"action":self.action_space.sample()}

        return {"action": [self.action_space.sample() for _ in range(self.num_agents)]}

    def train(self, sample:Sample) -> Dict[str, Any]:
        """There is no need to train"""
        return {}

class PPODiscrete(Policy):
    """PPO implementation"""
    def __init__(self, env_info: EnvInfo, config: Dict[str, Any]):
        super().__init__(env_info, config)
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise NotImplementedError("PPODiscrete only handles discrete prediction for now")

        self.device = config["device"]
        self.actor = MLP(self.observation_space.shape[0], self.action_space.n, embed_size=config["embed_size"]).to(self.device) #type:MLP
        self.critic = MLP(self.observation_space.shape[0], 1, embed_size=config["embed_size"]).to(self.device) #type:MLP
        self.actor: MLP #It is not recognized as from MLP class due to the to function for some reason
        self.critic: MLP #It is not recognized as from MLP class due to the to function for some reason
        self.actor.train()
        self.critic.train()
        self.optimizer = torch.optim.Adam([{"params": self.critic.parameters()},
                                           {"params": self.actor.parameters()}],
                                          lr=config["lr"], weight_decay=config["weight_decay"])
        self.loss_fn = torch.nn.MSELoss()
        self.gamma = config["gamma"]
        self.clip_coef = config["clip_coef"]
        self.num_updates = config["num_updates"]

    def act(self, observation:torch.Tensor, deterministic:bool=False):
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if observation.shape[:] == self.observation_space.shape:
            obs = torch.unsqueeze(obs, dim=0)
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(probs.probs, dim=-1)
        else:
            actions = probs.sample()
        if observation.shape[:] == self.observation_space.shape:
            return {"action": actions[0].detach().cpu().numpy(), "act_log_prob": probs.log_prob(actions)[0]}
        return {"action": actions.detach().cpu().numpy(), "act_log_prob": probs.log_prob(actions)}

    def get_val_prob(self, observation:torch.Tensor, actions:torch.Tensor):
        """Returns the val and log prob for an observation and action"""
        logits = self.actor(observation)
        probs = Categorical(logits=logits)
        return self.critic(observation), probs.log_prob(actions), probs.entropy()

    def train(self, sample):
        #Compute gae for the given sample
        #compute the loss
        #Do the optimizer step
        with torch.no_grad():
            returns, advantages = compute_gae(sample, self.critic, self.gamma)

        #flatten the sample
        sample.obs = sample.obs.reshape((-1,) + self.observation_space.shape)
        sample.act = sample.act.reshape((-1))
        sample.act_log_prob = sample.act_log_prob.reshape(-1)
        f_advantages = advantages.reshape(-1)
        f_returns = returns.reshape(-1)

        for _ in range(self.num_updates):
            new_values, new_log_probs, entropy = self.get_val_prob(sample.obs, sample.act)
            logratio = new_log_probs - sample.act_log_prob
            ratio = logratio.exp()

            # Policy loss
            pg_loss1 = -f_advantages * ratio
            pg_loss2 = -f_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            v_loss = 0.5 * ((new_values - f_returns) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss - 0.01 * entropy_loss + v_loss * 0.5

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {"loss": loss, "gamma": self.gamma}


    def train_mode(self):
        """Set models to train mode"""
        self.actor.train()
        self.critic.train()

    def eval_mode(self):
        """Set models to eval mode"""
        self.actor.eval()
        self.critic.eval()

    def get_mem_req(self) -> Dict:
        return {"type": "torch", "batch_size": self.batch_size, "act_log_prob": None}


class DQN(Policy):
    """DQN implementation"""
    def __init__(self, env_info:EnvInfo, config:Dict[str, Any]):
        super().__init__(env_info=env_info, config=config)
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise ValueError("DQN only handles discretes prediction")
        self.device = config["device"]
        self.critic = MLP(self.observation_space.shape[0], self.action_space.n, embed_size=config["embed_size"]).to(self.device)
        self.critic: MLP # We precise the type to not have typing error
        self.target_critic = deepcopy(self.critic)
        freeze(self.target_critic)

        self.critic.train()
        self.optimizer = torch.optim.Adam(self.critic.parameters(),
                                          lr=config["lr"],
                                          weight_decay=config["weight_decay"])
        self.loss_fn = torch.nn.MSELoss()
        self.epsilon = config["epsilon"]
        self.gamma = config["gamma"]
        self.target_freq = config["target_freq"]

        self.train_step = 0

    def act(self, observation:torch.Tensor, deterministic:bool=False):
        #If we tackle multiple environments, we return one action per env
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if observation.shape[:] == self.observation_space.shape:
            obs = torch.unsqueeze(obs, dim=0)
        actions = torch.argmax(self.critic(obs), dim=-1).detach().cpu().numpy()
        mask = (np.random.rand(obs.shape[0]) < self.epsilon ) & (not deterministic)
        actions = np.where(mask, self.action_space.sample(), actions)
        if observation.shape[:] == self.observation_space.shape:
            return {"action": actions[0]}
        return {"action": actions}

    def train(self, sample:ArraySample):
        sample.to_torch_tensor(self.device)
        obs, act, rew, done, next_obs = sample.obs, sample.act, sample.rew, sample.done, sample.next_obs
        act = act.to(torch.long) #Conver to long
        self.optimizer.zero_grad()

        q = self.critic(obs)
        q_a = torch.gather(q, 1, torch.unsqueeze(act, dim=-1))[:, 0]
        loss = self.loss_fn(q_a, rew + self.gamma * (1-done) * torch.max(self.target_critic(next_obs), dim=-1)[0])

        loss.backward()
        self.optimizer.step()

        self.train_step +=1

        if self.train_step % self.target_freq == 0:
            for target_param, param in zip(self.target_critic.parameters(),
                                           self.critic.parameters()):
                target_param.data.copy_(param.data)

        return {"loss": loss, "epsilon": self.epsilon}

    def train_mode(self):
        """Set models to train mode"""
        self.critic.train()
        self.target_critic.train()

    def eval_mode(self):
        """Set models to eval mode"""
        self.target_critic.eval()
        self.critic.eval()

    def get_mem_req(self) -> Dict:
        return {"type": "array", "batch_size": self.batch_size}

all_configs_policy = AllConfigs([
    Config(name="algo",
           config_type=str,
           config_help="The algo to use",
           default_val="RandomPolicy"),
    Config(name="embed_size",
           config_type=int,
           config_help="The embedding_size for the hidden layers",
           default_val=64),
    Config(name="batch_size",
           config_type=int,
           config_help="The batch_size to use",
           default_val=64),
    Config(name="gamma",
           config_type=float,
           config_help="gamma",
           default_val=0.99),
    Config(name="epsilon",
           config_type=float,
           config_help="Parameter for epsilon greedy exploration",
           default_val=0.2),
    Config(name="lr",
           config_type=float,
           config_help="The learning rate",
           default_val=1e-3),
    Config(name="weight_decay",
           config_type=float,
           config_help="The weight decay for adam",
           default_val=1e-3),
    Config(name="target_freq",
           config_type=float,
           config_help="The frequence of setting the target parameters",
           default_val=100),

    Config(name="clip_coef",
           config_type=float,
           config_help="To clip the policy loss",
           default_val=0.2),
    Config(name="num_updates",
           config_type=int,
           config_help="Num updates to do with the same batch of data",
           default_val=4),
    Config(name="num_updates",
           config_type=int,
           config_help="Num updates to do with the same batch of data",
           default_val=4),
])
