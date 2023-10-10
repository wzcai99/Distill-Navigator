from .layers import *
from .distributions import *
from gym.spaces import Space,Box
class ActorNet(nn.Module):
    def __init__(self,
                 state_dim:int,
                 action_dim:int,
                 hidden_sizes: Sequence[int],
                 normalize:Optional[ModuleType] = None,
                 initialize:Optional[Callable[...,torch.Tensor]] = None,
                 activation:Optional[ModuleType] = None,
                 device:Optional[Union[str,int,torch.device]] = None):
        super(ActorNet,self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp,input_shape = mlp_block(input_shape[0],h,normalize,activation,initialize,device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0],action_dim,None,None,initialize,device)[0])
        self.mu = nn.Sequential(*layers)     
        self.logstd = nn.Parameter(-0.5*torch.ones((action_dim,),device=device))
        self.dist = DiagGaussianDistribution(action_dim)
    def forward(self,x:torch.Tensor):
        self.dist.set_param(self.mu(x),self.logstd.exp())
        return self.dist        

class CriticNet(nn.Module):
    def __init__(self,
                 state_dim:int,
                 hidden_sizes: Sequence[int],
                 normalize:Optional[ModuleType] = None,
                 initialize:Optional[Callable[...,torch.Tensor]] = None,
                 activation:Optional[ModuleType] = None,
                 device:Optional[Union[str,int,torch.device]] = None):
        super(CriticNet,self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp,input_shape = mlp_block(input_shape[0],h,normalize,activation,initialize,device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0],1,None,None,None,device)[0])
        self.model = nn.Sequential(*layers)
    def forward(self,x:torch.Tensor):
        return self.model(x)[:,0]

class ActorCriticPolicy(nn.Module):
    def __init__(self,
                 action_space:Space,
                 representation:ModuleType,
                 actor_hidden_size:Sequence[int] = None,
                 critic_hidden_size:Sequence[int] = None,
                 normalize:Optional[ModuleType] = None,
                 initialize:Optional[Callable[...,torch.Tensor]] = None,
                 activation:Optional[ModuleType] = None,
                 device:Optional[Union[str,int,torch.device]] = None):
        assert isinstance(action_space,Box)
        super(ActorCriticPolicy,self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0],self.action_dim,actor_hidden_size,
                              normalize,initialize,activation,device)
        self.critic = CriticNet(representation.output_shapes['state'][0],critic_hidden_size,
                                normalize,initialize,activation,device)

    def forward(self,observation:Union[np.ndarray,dict]):  
        outputs = self.representation(observation)
        a = self.actor(outputs['state'])
        v = self.critic(outputs['state'])
        return outputs,a,v
    
class ActorPolicy(nn.Module):
    def __init__(self,
                 action_space:Space,
                 representation:ModuleType,
                 actor_hidden_size:Sequence[int] = None,
                 normalize:Optional[ModuleType] = None,
                 initialize:Optional[Callable[...,torch.Tensor]] = None,
                 activation:Optional[ModuleType] = None,
                 device:Optional[Union[str,int,torch.device]] = None,
                 fixed_std:bool = True):
        assert isinstance(action_space,Box)
        super(ActorPolicy,self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0],self.action_dim,actor_hidden_size,
                              normalize,initialize,activation,device)

    def forward(self,observation:Union[np.ndarray,dict]): 
        outputs = self.representation(observation)
        a = self.actor(outputs['state'])
        return outputs,a
