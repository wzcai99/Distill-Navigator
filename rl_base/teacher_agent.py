from gym.spaces import Space,Box,Discrete,Dict
from argparse import Namespace
from mpi4py import MPI
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch 
import torch.nn as nn
import torch.nn.functional as F
import cv2
from common import *
from environment import *
from .teacher_learner import Teacher_Learner

class Teacher_Agent:
    def __init__(self,
                 config:Namespace,
                 envs:VecEnv,
                 policy:nn.Module,
                 optimizer:torch.optim.Optimizer,
                 scheduler:Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device:Optional[Union[int,str,torch.device]] = None):
        self.comm = MPI.COMM_WORLD
        self.nenvs = envs.num_envs
        self.nsteps = config.nsteps
        self.nminibatch = config.nminibatch
        self.nepoch = config.nepoch
        self.policy = policy
        self.device = device
        self.envs = envs
        
        self.gamma = config.gamma
        self.lam = config.lam
        self.use_obsnorm = config.use_obsnorm
        self.use_rewnorm = config.use_rewnorm
        self.obsnorm_range = config.obsnorm_range
        self.rewnorm_range = config.rewnorm_range
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {"old_logp":()}
        self.writer = SummaryWriter(config.logdir)
        self.memory = DummyOnPolicyBuffer(self.observation_space,
                                    self.action_space,
                                    self.representation_info_shape,
                                    self.auxiliary_info_shape,
                                    self.nenvs,
                                    self.nsteps,
                                    self.nminibatch,
                                    self.gamma,
                                    self.lam)
        self.learner = Teacher_Learner(self.policy,
                                       optimizer,
                                       scheduler,
                                       self.writer,
                                       config.device,
                                       config.modeldir,
                                       config.vf_coef,
                                       config.ent_coef,
                                       config.clip_range)
        self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space),comm=self.comm,use_mpi=False)
        self.ret_rms = RunningMeanStd(shape=(),comm=self.comm,use_mpi=False)
        
        self.logdir = config.logdir
        self.modeldir = config.modeldir
        create_directory(config.logdir)
        create_directory(config.modeldir)
        
    def save_model(self):
        self.learner.save_model()
    def load_model(self,path):
        self.learner.load_model(path)
    
    def _process_observation(self,observations):
        if self.use_obsnorm:
            if isinstance(self.observation_space,gym.spaces.Dict):
                for key in self.observation_space.spaces.keys():
                    observations[key] = np.clip((observations[key] - self.obs_rms.mean[key])/(self.obs_rms.std[key]+EPS),-self.obsnorm_range,self.obsnorm_range)
            else:
                observations = np.clip((observations - self.obs_rms.mean)/(self.obs_rms.std+EPS),-self.obsnorm_range,self.obsnorm_range)
            return observations
        return observations
    def _process_reward(self,rewards):
        if self.use_rewnorm:
            std = np.clip(self.ret_rms.std,0.1,100)
            return np.clip(rewards / std,-self.rewnorm_range,self.rewnorm_range)
        return rewards
    def _action(self,obs):
        states,dists,vs = self.policy(obs)
        acts = dists.stochastic_sample()
        logps = dists.log_prob(acts)
        for key in states.keys():
            states[key] = states[key].detach().cpu().numpy()
        vs = vs.detach().cpu().numpy()
        acts = acts.detach().cpu().numpy()
        logps = logps.detach().cpu().numpy()
        return states,acts,vs,logps
    
    def train(self,train_steps=10000):
        episodes = np.zeros((self.nenvs,),np.int32)
        scores = np.zeros((self.nenvs,),np.float32)
        returns = np.zeros((self.nenvs,),np.float32)
        obs = self.envs.reset()        
        for step in tqdm(range(train_steps)):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            states,acts,rets,logps = self._action(obs)
            next_obs,rewards,dones,infos = self.envs.step(acts)
            self.memory.store(obs,acts,self._process_reward(rewards),rets,dones,states,{"old_logp":logps})
            if self.memory.full:
                _,_,vals,_ = self._action(self._process_observation(next_obs))
                for i in range(self.nenvs):
                    self.memory.finish_path(vals[i],i)
                for _ in range(self.nminibatch*self.nepoch):
                    obs_batch,act_batch,ret_batch,adv_batch,_,aux_batch = self.memory.sample()
                    self.learner.update(obs_batch,act_batch,ret_batch,adv_batch,aux_batch['old_logp'])
                self.memory.clear()
            scores += rewards
            returns = self.gamma*returns + rewards
            obs = next_obs
            for i in range(self.nenvs):
                if dones[i] == True:
                    self.ret_rms.update(returns[i:i+1])
                    self.memory.finish_path(0,i)
                    self.writer.add_scalars("length",{"env-%d"%i:infos[i]['length']},step)
                    self.writer.add_scalars("activation",{"env-%d"%i:infos[i]['activation']},step)
                    self.writer.add_scalars("returns-step",{"env-%d"%i:scores[i]},step)
                    self.writer.add_scalars("collision",{"env-%d"%i:infos[i]['collision']},step)
                    scores[i] = 0
                    returns[i] = 0
                    episodes[i] += 1
            if step % 10000 == 0 or step == train_steps - 1:
                self.save_model()
                np.save(self.modeldir + "/obs_rms.npy",{'mean':self.obs_rms.mean,'std':self.obs_rms.std,'count':self.obs_rms.count})
                                    
    def test(self,test_episodes,model_name):
        self.load_model(model_name)
        activation = []
        collision = []
        time = []
        episode = 0 
        obs = self.envs.reset()   
        while episode < test_episodes:
            obs = self._process_observation(obs)
            states,acts,rets,logps = self._action(obs)
            next_obs,rewards,dones,infos = self.envs.step(acts)
            for i in range(self.nenvs):
                if dones[i] == True:
                    episode += 1
                    activation.append(infos[i]['activation'])
                    collision.append(infos[i]['collision'])
                    time.append(infos[i]['length'])
                    print("Episode:%d,Activation:%f,Collision:%f,Time:%f"%(episode,infos[i]['activation'],infos[i]['collision'],infos[i]['length']))
            obs = next_obs
        print("Testing for %d episodes, Evaluation Metrics: Activation:%f,Collision:%f,Time:%f"%(episode,np.mean(activation),np.mean(collision),np.mean(time)))

        