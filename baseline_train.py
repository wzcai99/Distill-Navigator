from environment import make_envs
from common import get_config,space2shape
from rl_base.representation import Baseline_CNN
from rl_base.policy import ActorCriticPolicy as Baseline_Policy
from rl_base.baseline_agent import Baseline_Agent
import argparse
import torch
import os
os.environ['DISPLAY'] = ":1"
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id",type=str,default="CoG-Navigation")
    parser.add_argument("--seed",type=int,default=7820)
    parser.add_argument("--config_path",type=str,default="./config/baseline_config.yaml")
    args = parser.parse_known_args()[0]
    return args
if __name__ == "__main__":
    args = get_args()
    config = get_config(args.config_path)
    envs = make_envs(args.env_id,args.seed,config)
    observation_space = envs.observation_space
    action_space = envs.action_space
    representation = Baseline_CNN(space2shape(observation_space),
                                  None,
                                  torch.nn.init.orthogonal_,
                                  torch.nn.LeakyReLU,
                                  config.device)
    policy = Baseline_Policy(action_space,
                            representation,
                            config.actor_hidden_size,
                            config.critic_hidden_size,
                            initialize=torch.nn.init.orthogonal_,
                            activation=torch.nn.Tanh,
                            device = config.device)
    optimizer = torch.optim.Adam(policy.parameters(),config.learning_rate,eps=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1.0,end_factor=0.25,total_iters=int(config.training_steps*config.nepoch*config.nminibatch/config.nsteps))
    agent = Baseline_Agent(config,envs,policy,optimizer,lr_scheduler,config.device)
    agent.train(config.training_steps)


