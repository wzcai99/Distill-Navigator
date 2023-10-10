import numpy as np
from common import discount_cumsum,EPS
from gym import Space
from abc import ABC,abstractmethod
from typing import Optional,Union,Sequence
from common.common_tools import space2shape

def create_memory(shape:Optional[Union[tuple,dict]],nenvs:int,nsize:int):
    if shape == None:
        return None
    elif isinstance(shape,dict):
        memory = {}
        for key,value in zip(shape.keys(),shape.values()):
            if value is None: #save an object type
                memory[key] = np.zeros([nenvs,nsize],dtype=object)
            else:
                memory[key] = np.zeros([nenvs,nsize]+list(value),dtype=np.float32)
        return memory
    elif isinstance(shape,tuple):
        return np.zeros([nenvs,nsize]+list(shape),np.float32)
    else:
        raise NotImplementedError

def store_element(data:Optional[Union[np.ndarray,dict,float]],memory:Union[dict,np.ndarray],ptr:int):
    if data is None:
        return
    elif isinstance(data,dict):
        for key,value in zip(data.keys(),data.values()):
            memory[key][:,ptr] = data[key]
    else:
        memory[:,ptr] = data

def sample_batch(memory:Optional[Union[np.ndarray,dict]],index:np.ndarray):
    if memory is None:
        return None
    elif isinstance(memory,dict):
        batch = {}
        for key,value in zip(memory.keys(),memory.values()):
            batch[key] = value[index]
        return batch
    else:
        return memory[index]

class Buffer(ABC):
    def __init__(self,
                 observation_space:Space,
                 action_space:Space,
                 representation_info_shape:Optional[dict],
                 auxiliary_info_shape:Optional[dict]):
        self.observation_space = observation_space
        self.action_space = action_space
        self.representation_shape = representation_info_shape
        self.auxiliary_shape = auxiliary_info_shape
        self.size,self.ptr = 0,0
    @abstractmethod
    def store(self,*args):
        raise NotImplementedError
    @abstractmethod
    def clear(self,*args):
        raise NotImplementedError
    @abstractmethod
    def sample(self,*args):
        raise NotImplementedError
    
class DummyOnPolicyBuffer(Buffer):
    def __init__(self,
                 observation_space:Space,
                 action_space:Space,
                 representation_shape:Optional[dict],
                 auxiliary_shape:Optional[dict],
                 nenvs:int,
                 nsize:int,
                 nminibatch:int,
                 gamma:float=0.99,
                 lam:float=0.95):
        super(DummyOnPolicyBuffer,self).__init__(observation_space,
                                                 action_space,
                                                 representation_shape,
                                                 auxiliary_shape)
        self.nenvs,self.nsize,self.nminibatch= nenvs,nsize,nminibatch
        self.gamma,self.lam = gamma,lam
        self.start_ids = np.zeros(self.nenvs,np.int64)
        self.observations = create_memory(space2shape(self.observation_space),self.nenvs,self.nsize)
        self.actions = create_memory(space2shape(self.action_space),self.nenvs,self.nsize)
        self.representation_infos = create_memory(self.representation_shape,self.nenvs,self.nsize)
        self.auxiliary_infos = create_memory(self.auxiliary_shape,self.nenvs,self.nsize)
        self.rewards = create_memory((),self.nenvs,self.nsize)
        self.returns = create_memory((),self.nenvs,self.nsize)
        self.terminals = create_memory((),self.nenvs,self.nsize)
        self.advantages = create_memory((),self.nenvs,self.nsize)
    @property
    def full(self):
        return self.size >= self.nsize
    def clear(self):
        self.ptr,self.size = 0,0
        self.observations = create_memory(space2shape(self.observation_space),self.nenvs,self.nsize)
        self.actions = create_memory(space2shape(self.action_space),self.nenvs,self.nsize)
        self.representation_infos = create_memory(self.representation_shape,self.nenvs,self.nsize)
        self.auxiliary_infos = create_memory(self.auxiliary_shape,self.nenvs,self.nsize)
        self.rewards = create_memory((),self.nenvs,self.nsize)
        self.returns = create_memory((),self.nenvs,self.nsize)
        self.advantages = create_memory((),self.nenvs,self.nsize)
        
    def store(self,obs,acts,rews,rets,terminals,rep_info,aux_info):
        store_element(obs,self.observations,self.ptr)
        store_element(acts,self.actions,self.ptr)
        store_element(rews,self.rewards,self.ptr)
        store_element(rets,self.returns,self.ptr)
        store_element(terminals,self.terminals,self.ptr)
        store_element(rep_info,self.representation_infos,self.ptr)
        store_element(aux_info,self.auxiliary_infos,self.ptr)
        self.ptr = (self.ptr+1)%self.nsize
        self.size = min(self.size+1,self.nsize)
    def finish_path(self,val,i):
        if self.full:
            path_slice = np.arange(self.start_ids[i],self.nsize).astype(np.int32)
        else:
            path_slice = np.arange(self.start_ids[i],self.ptr).astype(np.int32)
        rewards = np.append(np.array(self.rewards[i,path_slice]),[val],axis=0)
        critics = np.append(np.array(self.returns[i,path_slice]),[val],axis=0)
        returns = discount_cumsum(rewards,self.gamma)[:-1]
        deltas = rewards[:-1] + self.gamma * critics[1:] - critics[:-1]
        advantages = discount_cumsum(deltas,self.gamma*self.lam)
        self.returns[i,path_slice] = returns
        self.advantages[i,path_slice] = advantages
        self.start_ids[i] = self.ptr
    def sample(self):
        assert self.full, "Not enough transitions for on-policy buffer to random sample"
       
        env_choices = np.random.choice(self.nenvs,self.nenvs*self.nsize//self.nminibatch)
        step_choices = np.random.choice(self.nsize,self.nenvs*self.nsize//self.nminibatch)
        
        obs_batch = sample_batch(self.observations,[env_choices,step_choices])
        act_batch = sample_batch(self.actions,[env_choices,step_choices])
        ret_batch = sample_batch(self.returns,[env_choices,step_choices])
        adv_batch = sample_batch(self.advantages,[env_choices,step_choices])
        rep_batch = sample_batch(self.representation_infos,[env_choices,step_choices])
        aux_batch = sample_batch(self.auxiliary_infos,[env_choices,step_choices])
        adv_batch = (adv_batch - np.mean(self.advantages))/(np.std(self.advantages)+EPS)  
                          
        return obs_batch,act_batch,ret_batch,adv_batch,rep_batch,aux_batch
    

class DummyOffPolicyBuffer(Buffer):
    def __init__(self,
                 observation_space:Space,
                 action_space:Space,
                 representation_shape:Optional[dict],
                 auxiliary_shape:Optional[dict],
                 nenvs:int,
                 nsize:int,
                 batchsize:int):
        super(DummyOffPolicyBuffer,self).__init__(observation_space,
                                                  action_space,
                                                  representation_shape,
                                                  auxiliary_shape)
        self.nenvs,self.nsize,self.batchsize = nenvs,nsize,batchsize
        self.observations = create_memory(space2shape(self.observation_space),self.nenvs,self.nsize)
        self.next_observations = create_memory(space2shape(self.observation_space),self.nenvs,self.nsize)
        self.actions = create_memory(space2shape(self.action_space),self.nenvs,self.nsize)
        self.representation_infos = create_memory(self.representation_shape,self.nenvs,self.nsize)
        self.auxiliary_infos = create_memory(self.auxiliary_shape,self.nenvs,self.nsize)
        self.rewards = create_memory((),self.nenvs,self.nsize)
        self.terminals = create_memory((),self.nenvs,self.nsize)
    
    def clear(self):
        self.observations = create_memory(space2shape(self.observation_space),self.nenvs,self.nsize)
        self.next_observations = create_memory(space2shape(self.observation_space),self.nenvs,self.nsize)
        self.actions = create_memory(space2shape(self.action_space),self.nenvs,self.nsize)
        self.representation_infos = create_memory(self.representation_shape,self.nenvs,self.nsize)
        self.auxiliary_infos = create_memory(self.auxiliary_shape,self.nenvs,self.nsize)
        self.rewards = create_memory((),self.nenvs,self.nsize)
        self.terminals = create_memory((),self.nenvs,self.nsize)
    
    def store(self,obs,acts,rews,terminals,next_obs,rep_info,aux_info):
        store_element(obs,self.observations,self.ptr)
        store_element(acts,self.actions,self.ptr)
        store_element(rews,self.rewards,self.ptr)
        store_element(terminals,self.terminals,self.ptr)
        store_element(next_obs,self.next_observations,self.ptr)
        store_element(rep_info,self.representation_infos,self.ptr)
        store_element(aux_info,self.auxiliary_infos,self.ptr)
        self.ptr = (self.ptr + 1)%self.nsize
        self.size = min(self.size+1,self.nsize)
    
    def sample(self):
        env_choices = np.random.choice(self.nenvs,self.batchsize)
        step_choices = np.random.choice(self.size,self.batchsize)
        obs_batch = sample_batch(self.observations,[env_choices,step_choices])
        act_batch = sample_batch(self.actions,[env_choices,step_choices])
        rew_batch = sample_batch(self.rewards,[env_choices,step_choices])
        terminal_batch = sample_batch(self.terminals,[env_choices,step_choices])
        next_batch = sample_batch(self.next_observations,[env_choices,step_choices])
        rep_batch = sample_batch(self.representation_infos,[env_choices,step_choices])
        aux_batch = sample_batch(self.auxiliary_infos,[env_choices,step_choices])
        return obs_batch,act_batch,rew_batch,terminal_batch,next_batch,rep_batch,aux_batch
