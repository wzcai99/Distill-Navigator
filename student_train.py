from environment import make_envs
from common import get_config,space2shape
from rl_base.representation import Student_Encoder,Teacher_Encoder
from rl_base.policy import ActorCriticPolicy as Policy
from rl_base.student_agent import Student_Agent
import argparse
import torch
import os
os.environ['DISPLAY'] = ":1"
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id",type=str,default="CoG-Navigation")
    parser.add_argument("--seed",type=int,default=7820)
    parser.add_argument("--config_path",type=str,default="./config/student_config.yaml")
    args = parser.parse_known_args()[0]
    return args
if __name__ == "__main__":
    args = get_args()
    config = get_config(args.config_path)
    envs = make_envs(args.env_id,args.seed,config)
    observation_space = envs.observation_space
    action_space = envs.action_space
    representation = Student_Encoder(space2shape(observation_space),
                                     None,
                                     torch.nn.init.orthogonal_,
                                     torch.nn.LeakyReLU,
                                     config.device,
                                     config.teacher_model)
    student_policy = Policy(action_space,
                            representation,
                            config.actor_hidden_size,
                            config.critic_hidden_size,
                            initialize=torch.nn.init.orthogonal_,
                            activation=torch.nn.Tanh,
                            device = config.device)
    teacher_representation = Teacher_Encoder(space2shape(observation_space),
                                             None,
                                             torch.nn.init.orthogonal_,
                                             torch.nn.LeakyReLU,
                                             config.device,)
    teacher_policy = Policy(action_space,
                            teacher_representation,
                            config.actor_hidden_size,
                            config.critic_hidden_size,
                            initialize=torch.nn.init.orthogonal_,
                            activation=torch.nn.Tanh,
                            device = config.device)
    teacher_policy.load_state_dict(torch.load(config.teacher_model))
    optimizer = torch.optim.Adam(student_policy.parameters(),config.learning_rate,eps=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1.0,end_factor=0.25,total_iters=int(config.training_steps*config.nepoch*config.nminibatch/config.nsteps))
    agent = Student_Agent(config,envs,student_policy,teacher_policy,optimizer,lr_scheduler,config.device)
    agent.train(config.training_steps)


