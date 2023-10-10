import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Optional, Sequence,Union
from torch.utils.tensorboard import SummaryWriter

class Teacher_Learner:
    def __init__(self,
                 policy:nn.Module,
                 optimizer:torch.optim.Optimizer,
                 scheduler:Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 summary_writer:Optional[SummaryWriter] = None,
                 device: Optional[Union[int,str,torch.device]] = None,
                 modeldir: str = "./",
                 vf_coef: float = 0.25,
                 ent_coef: float = 0.005,
                 clip_range: float = 0.25):
        self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = summary_writer
        self.device = device
        self.modeldir = modeldir
        self.iterations = 0
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_range = clip_range
        self.aux_iterations = 0
        
    def save_model(self):
        time_string = time.asctime()
        time_string = time_string.replace(" ","")
        model_path = self.modeldir + "model-%s-%s.pth"%(time.asctime(),str(self.iterations))
        torch.save(self.policy.state_dict(),model_path)
    def load_model(self,path):
        model_path = self.modeldir + path
        self.policy.load_state_dict(torch.load(model_path))
        
    def update(self,obs_batch,act_batch,ret_batch,adv_batch,old_logp):
        self.iterations += 1
        act_batch = torch.as_tensor(act_batch,device=self.device)
        ret_batch = torch.as_tensor(ret_batch,device=self.device)
        adv_batch = torch.as_tensor(adv_batch,device=self.device)
        old_logp_batch = torch.as_tensor(old_logp,device=self.device)
        
        outputs,a_dist,v_pred = self.policy(obs_batch)
        log_prob = a_dist.log_prob(act_batch)
        # ppo-clip core implementations 
        ratio = (log_prob - old_logp_batch).exp().float()
        surrogate1 = ratio.clamp(1.0-self.clip_range,1.0+self.clip_range)*adv_batch
        surrogate2 = adv_batch * ratio
        a_loss = -torch.minimum(surrogate1,surrogate2).mean()
        c_loss = F.mse_loss(v_pred,ret_batch)
        e_loss = a_dist.entropy().mean()
        f_loss = outputs['loss']
        # teacher loss
        loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss + 0.1*f_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.policy.representation.soft_update()
        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        cr = ((ratio<1-self.clip_range).sum()+(ratio>1+self.clip_range).sum())/ratio.shape[0]
        self.writer.add_scalar("actor-loss",a_loss.item(),self.iterations)
        self.writer.add_scalar("critic-loss",c_loss.item(),self.iterations)
        self.writer.add_scalar("infonce-loss",f_loss.item(),self.iterations)
        self.writer.add_scalar("entropy",e_loss.item(),self.iterations)
        self.writer.add_scalar("learning_rate",lr,self.iterations)
        self.writer.add_scalar("predict_value",v_pred.mean().item(),self.iterations)
        self.writer.add_scalar("clip_ratio",cr,self.iterations)