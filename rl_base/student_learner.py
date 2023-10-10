import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Optional, Sequence,Union
from torch.utils.tensorboard import SummaryWriter

class Student_Learner:
    def __init__(self,
                 policy:nn.Module,
                 policy_teacher:nn.Module,
                 optimizer:torch.optim.Optimizer,
                 scheduler:Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 summary_writer:Optional[SummaryWriter] = None,
                 device: Optional[Union[int,str,torch.device]] = None,
                 modeldir: str = "./"):
        self.policy = policy
        self.policy_teacher = policy_teacher
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = summary_writer
        self.device = device
        self.modeldir = modeldir
        self.iterations = 0
        self.aux_iterations = 0
    def save_model(self):
        time_string = time.asctime()
        time_string = time_string.replace(" ","")
        model_path = self.modeldir + "model-%s-%s.pth"%(time.asctime(),str(self.iterations))
        torch.save(self.policy.state_dict(),model_path)
    def load_model(self,path):
        model_path = self.modeldir + path
        self.policy.load_state_dict(torch.load(model_path))
    def update(self,obs_batch):
        self.iterations += 1
        outputs,a_dist,v_pred = self.policy(obs_batch)
        _,t_dist,_ = self.policy_teacher(obs_batch)
        kl_loss = a_dist.kl_divergence(t_dist).mean()
        d_loss = outputs['loss']
        loss = 0.7*kl_loss + 0.3*d_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.policy_teacher.zero_grad()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.writer.add_scalar("d-loss",d_loss.item(),self.iterations)
        self.writer.add_scalar("kl-loss",kl_loss.item(),self.iterations)
        self.writer.add_scalar("learning_rate",lr,self.iterations)
       
      
        