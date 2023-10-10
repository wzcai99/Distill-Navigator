from .layers import *
import torchvision.transforms as T
class Baseline_CNN(nn.Module):
    def __init__(self,
                 input_shape:Sequence[int],
                 normalize:Optional[ModuleType] = None,
                 initialize:Optional[Callable[...,torch.Tensor]] = None,
                 activation:Optional[ModuleType] = None,
                 device:Optional[Union[str,int,torch.device]]=None):
        super(Baseline_CNN,self).__init__()
        self.input_shape = input_shape
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.output_shapes = {'state':(2048+128,)}
        self.semantic_encoder = self._semantic_encoder()
        self.laser_encoder = self._laser_encoder()
    def _semantic_encoder(self):
        input_shape = (self.input_shape['topdown'][2],self.input_shape['topdown'][0],self.input_shape['topdown'][1])
        cnn1,input_shape = cnn_block(input_shape,8,8,4,None,self.activation,self.initialize,self.device)
        cnn2,input_shape = cnn_block(input_shape,16,6,2,None,self.activation,self.initialize,self.device)
        cnn3,input_shape = cnn_block(input_shape,32,4,2,None,self.activation,self.initialize,self.device)
        return nn.Sequential(*(cnn1+cnn2+cnn3))
    def _laser_encoder(self):
        mlp1,_ = mlp_block(self.input_shape['laser'][0],64,self.normalize,self.activation,self.initialize,self.device)
        mlp2,_ = mlp_block(64,128,self.normalize,self.activation,self.initialize,self.device)
        return nn.Sequential(*(mlp1+mlp2))
    def forward(self,observations:dict):
        tensor_point = torch.as_tensor(observations['laser'],dtype=torch.float32,device=self.device)
        tensor_semantic = torch.as_tensor(np.transpose(observations['topdown'],(0,3,1,2)),dtype=torch.float32,device=self.device)
        point_feature = self.laser_encoder(tensor_point)
        semantic_feature = self.semantic_encoder(tensor_semantic).permute(0,2,3,1).flatten(start_dim=1,end_dim=-1)
        state = torch.cat((semantic_feature,point_feature),dim=-1)
        return {'state':state}

class Teacher_Encoder(nn.Module):
    def __init__(self,
                 input_shape:Sequence[int],
                 normalize:Optional[ModuleType] = None,
                 initialize:Optional[Callable[...,torch.Tensor]] = None,
                 activation:Optional[ModuleType] = None,
                 device:Optional[Union[str,int,torch.device]]=None):
        super(Teacher_Encoder,self).__init__()
        self.input_shape = input_shape
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.output_shapes = {'state':(2048+128,),'loss':()}
        self.teacher_cnn = self._teacher_cnn()
        self.momentum_cnn = self._momentum_cnn()
        self.laser_mlp = self._laser_mlp()
        self.color_jitter = T.ColorJitter((0.5,1.5),(0.5,1.5),(0.5,1.5),(-0.2,0.2))
        self.size_scale = T.RandomAffine(degrees=0,scale=(1,1.2))
        self.transform = nn.Sequential(self.color_jitter,self.size_scale)
        for ep,tp in zip(self.momentum_cnn.parameters(),self.teacher_cnn.parameters()):
            ep.data.copy_(tp.data)
    def _teacher_cnn(self):
        input_shape = (self.input_shape['label_topdown'][2],self.input_shape['label_topdown'][0],self.input_shape['label_topdown'][1])
        cnn1,input_shape = cnn_block(input_shape,8,8,4,None,self.activation,self.initialize,self.device)
        cnn2,input_shape = cnn_block(input_shape,16,6,2,None,self.activation,self.initialize,self.device)
        cnn3,input_shape = cnn_block(input_shape,32,4,2,None,self.activation,self.initialize,self.device)
        return nn.Sequential(*(cnn1+cnn2+cnn3))
    def _laser_mlp(self):
        mlp1,_ = mlp_block(self.input_shape['laser'][0],128,self.normalize,self.activation,self.initialize,self.device)
        mlp2,_ = mlp_block(128,128,self.normalize,self.activation,self.initialize,self.device)
        return nn.Sequential(*(mlp1+mlp2))
    def _momentum_cnn(self):
        input_shape = (self.input_shape['label_topdown'][2],self.input_shape['label_topdown'][0],self.input_shape['label_topdown'][1])
        cnn1,input_shape = cnn_block(input_shape,8,8,4,None,self.activation,self.initialize,self.device)
        cnn2,input_shape = cnn_block(input_shape,16,6,2,None,self.activation,self.initialize,self.device)
        cnn3,input_shape = cnn_block(input_shape,32,4,2,None,self.activation,self.initialize,self.device)
        return nn.Sequential(*(cnn1+cnn2+cnn3))
    def soft_update(self):
        for ep,tp in zip(self.momentum_cnn.parameters(),self.teacher_cnn.parameters()):
            ep.data.copy_(0.995*ep.data + 0.005*tp.data)
    def forward(self,observations:dict):
        tensor_map = torch.as_tensor(np.transpose(observations['label_topdown'],(0,3,1,2)),dtype=torch.float32,device=self.device)
        tensor_noise_map = torch.as_tensor(np.transpose(observations['topdown'],(0,3,1,2)),dtype=torch.float32,device=self.device)
        tensor_laser = torch.as_tensor(observations['laser'],dtype=torch.float32,device=self.device)
        
        map_feature = self.teacher_cnn(tensor_map).permute(0,2,3,1).flatten(start_dim=1,end_dim=-1)
        laser_feature = self.laser_mlp(tensor_laser)
        augment_mapA = tensor_map
        augment_mapB = self.transform(tensor_map)
        augment_mapC = tensor_noise_map
        feature_augA = self.teacher_cnn(augment_mapA).permute(0,2,3,1).flatten(start_dim=1,end_dim=-1) # B * 2048
        feature_augB = self.momentum_cnn(augment_mapB).detach().permute(0,2,3,1).flatten(start_dim=1,end_dim=-1) # B * 2048
        feature_augC = self.momentum_cnn(augment_mapC).detach().permute(0,2,3,1).flatten(start_dim=1,end_dim=-1)
        l_pos = torch.bmm(feature_augA.view(-1,1,2048),feature_augB.view(-1,2048,1)).view(-1,1) # N*1
        l_neg = torch.mm(feature_augA,feature_augC.permute(1,0)) # N*N
        logits = torch.cat((l_pos,l_neg),dim=-1)
        labels = torch.zeros(logits.shape[0],dtype=torch.int64,device=self.device)
        loss = F.cross_entropy(logits/0.25,labels)
        return {"state":torch.concat((map_feature,laser_feature),dim=-1),"loss":loss}

# Policy Distill + Feature Distill
class Student_Encoder(nn.Module):
    def __init__(self,
                 input_shape:Sequence[int],
                 normalize:Optional[ModuleType] = None,
                 initialize:Optional[Callable[...,torch.Tensor]] = None,
                 activation:Optional[ModuleType] = None,
                 device:Optional[Union[str,int,torch.device]]=None,
                 teacher_model:str=None):
        super(Student_Encoder,self).__init__()
        self.input_shape = input_shape
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.output_shapes = {'state':(2048+128,),'loss':()}
        self.student_cnn = self._student_cnn()
        self.student_mlp = self._student_mlp()
        self.teacher_cnn = self._teacher_cnn()
        self.transform_layer = self._transfrom_mlp()
        self.teacher_params = torch.load(teacher_model)
        self.teacher_cnn_params = []
        for key in self.teacher_params.keys():
            if 'teacher_cnn' in key:
                self.teacher_cnn_params.append(self.teacher_params[key])
        for ep,tp in zip(self.student_cnn.parameters(),self.teacher_cnn_params):
            ep.data.copy_(tp)
    
    def _teacher_cnn(self):
        input_shape = (self.input_shape['label_topdown'][2],self.input_shape['label_topdown'][0],self.input_shape['label_topdown'][1])
        cnn1,input_shape = cnn_block(input_shape,8,8,4,None,self.activation,self.initialize,self.device)
        cnn2,input_shape = cnn_block(input_shape,16,6,2,None,self.activation,self.initialize,self.device)
        cnn3,input_shape = cnn_block(input_shape,32,4,2,None,self.activation,self.initialize,self.device)
        return nn.Sequential(*(cnn1+cnn2+cnn3))
    # Fixed CNN and Fixed MLP
    def _student_cnn(self):
        input_shape = (self.input_shape['label_topdown'][2],self.input_shape['label_topdown'][0],self.input_shape['label_topdown'][1])
        cnn1,input_shape = cnn_block(input_shape,8,8,4,None,self.activation,self.initialize,self.device)
        cnn2,input_shape = cnn_block(input_shape,16,6,2,None,self.activation,self.initialize,self.device)
        cnn3,input_shape = cnn_block(input_shape,32,4,2,None,self.activation,self.initialize,self.device)
        return nn.Sequential(*(cnn1+cnn2+cnn3))
    
    def _student_mlp(self):
        mlp1,_ = mlp_block(self.input_shape['laser'][0],128,self.normalize,self.activation,self.initialize,self.device)
        mlp2,_ = mlp_block(128,128,self.normalize,self.activation,self.initialize,self.device)
        return nn.Sequential(*(mlp1+mlp2))
    
    def _transfrom_mlp(self):    
        mlp1,_ = mlp_block(2048+128,2048,self.normalize,self.activation,self.initialize,self.device)
        mlp2,_ = mlp_block(2048,2048,None,None,self.initialize,self.device)
        return nn.Sequential(*(mlp1+mlp2))
        
    def forward(self,observations:dict):
        tensor_map = torch.as_tensor(np.transpose(observations['topdown'],(0,3,1,2)),dtype=torch.float32,device=self.device)
        tensor_laser = torch.as_tensor(observations['laser'],dtype=torch.float32,device=self.device)
        tensor_label = torch.as_tensor(np.transpose(observations['label_topdown'],(0,3,1,2)),dtype=torch.float32,device=self.device)
        
        label_feature = self.teacher_cnn(tensor_label).detach().permute(0,2,3,1).flatten(start_dim=1,end_dim=-1)
        map_feature = self.student_cnn(tensor_map).permute(0,2,3,1).flatten(start_dim=1,end_dim=-1)
        laser_feature = self.student_mlp(tensor_laser)
        transform_feature = self.transform_layer(torch.concat((map_feature,laser_feature),dim=-1))
        
        loss = -((F.normalize(transform_feature,dim=-1) * F.normalize(label_feature,dim=-1)).sum(-1)).mean()
        return {"state":torch.concat((transform_feature,laser_feature),dim=-1),'loss':loss}