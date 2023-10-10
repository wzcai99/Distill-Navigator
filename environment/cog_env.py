
from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
from gym.spaces import Box,Discrete,Dict,Space
import gym
import numpy as np
import cv2
ENVIRONMENT_IDS = ['CoG-Navigation']
MARGIN = 0
# Provide a Top-Down Map as Input
class CoG_Mapper(object):
    def __init__(self):
        pass
    def ruma_map(self,target_loc,enemy_loc,goal_loc):
        ruma_map = 255*np.ones((448,808,3),np.uint8)
        #B1,B3,B5,B7
        ruma_map = cv2.rectangle(ruma_map,pt1=(0,100-MARGIN),pt2=(100,120+MARGIN),color=(128,128,128),thickness=-1)
        ruma_map = cv2.rectangle(ruma_map,pt1=(808-170-MARGIN,0),pt2=(808-150+MARGIN,100),color=(128,128,128),thickness=-1)
        ruma_map = cv2.rectangle(ruma_map,pt1=(150-MARGIN,448-100),pt2=(170+MARGIN,448),color=(128,128,128),thickness=-1)
        ruma_map = cv2.rectangle(ruma_map,pt1=(808-100,448-100-20-MARGIN),pt2=(808,448-100+MARGIN),color=(128,128,128),thickness=-1)
        #B4,B6
        ruma_map = cv2.rectangle(ruma_map,pt1=(808-354-100,93-MARGIN),pt2=(808-354,93+20+MARGIN),color=(128,128,128),thickness=-1)
        ruma_map = cv2.rectangle(ruma_map,pt1=(808-354-100,448-93-20-MARGIN),pt2=(808-354,448-93+MARGIN),color=(128,128,128),thickness=-1)
        #B2,B8
        ruma_map = cv2.rectangle(ruma_map,pt1=(150,214-MARGIN),pt2=(150+80,234+MARGIN),color=(128,128,128),thickness=-1)
        ruma_map = cv2.rectangle(ruma_map,pt1=(808-150-80,214-MARGIN),pt2=(808-150,234+MARGIN),color=(128,128,128),thickness=-1)
        #B9
        ruma_map = cv2.rectangle(ruma_map,pt1=(404-25-MARGIN,224-25-MARGIN),pt2=(404+25+MARGIN,224+25+MARGIN),color=(128,128,128),thickness=-1)
        #Target Loc
        for loc in target_loc:
            ruma_map = cv2.rectangle(ruma_map,pt1=(int(loc[0]*100-25-MARGIN),int(448-loc[1]*100-25-MARGIN)),pt2=(int(loc[0]*100+25+MARGIN),int(448-loc[1]*100+25+MARGIN)),color=(128,128,128),thickness=-1)
        ruma_map = cv2.rectangle(ruma_map,pt1=(int(enemy_loc[0]*100-25-MARGIN),int(448-enemy_loc[1]*100-25-MARGIN)),pt2=(int(enemy_loc[0]*100+25+MARGIN),int(448-enemy_loc[1]*100+25+MARGIN)),color=(205,90,106),thickness=-1)
        ruma_map = cv2.rectangle(ruma_map,pt1=(int(goal_loc[0]*100-25-MARGIN),int(448-goal_loc[1]*100-25-MARGIN)),pt2=(int(goal_loc[0]*100+25+MARGIN),int(448-goal_loc[1]*100+25+MARGIN)),color=(106,106,255),thickness=-1)
        return ruma_map
    def reset(self,marker_info,enemy_pose,goal_loc):
        marker_pose = marker_info[:,0:2]
        self.dense_map = self.ruma_map(marker_pose,enemy_pose,goal_loc)
    def step(self,marker_info,enemy_pose,goal_loc):
        marker_pose = marker_info[:,0:2]
        self.dense_map = self.ruma_map(marker_pose,enemy_pose,goal_loc)

class Cog_Env(gym.Env):
    def __init__(self,env_id,seed_num,config):
        super(Cog_Env,self).__init__()
        assert env_id in ENVIRONMENT_IDS
        self.seed_num = seed_num
        self.env_id = env_id
        self.worker_id = np.random.randint(0,1024)
        self.time_scale = config.time_scale
        self.env_sim_name = config.sim_path
        self.timelimit = config.timelimit
        self.topdown_size = config.topdown_size
        # Noise Settings
        self.uniform_laser_noise_scale = config.uniform_laser_noise_scale
        self.episode_pose_noise_scale = config.episode_pose_noise_scale
        self.step_pose_noise_scale = config.step_pose_noise_scale
        self.step_angle_noise_scale = config.step_angle_noise_scale
        self.resize_scale = config.resize_scale
        self.speed_scale = config.speed_scale
        self.laser_observation_space = Box(0,10,(60,))
        self.pose_observation_space = Box(-10,10,shape=(4,))
        self.goal_observation_space = Box(0,10,(4,))
        self.topdown_observation_space = Box(0,1,(self.resize_scale,self.resize_scale,3))
        self.laser_map_space = Box(0,1,(self.resize_scale,self.resize_scale,3))
        self.laser_point_space = Box(-10,10,(60,2,))
        self.action_observation_space = Box(-2,2,(4,))
        self.shift_observation_space = Box(-10,10,(2,))
        self.action_space = Box(-1,1,(4,))
        self.observation_space = Dict({'laser':self.laser_observation_space,
                                       'goal':self.goal_observation_space,
                                       'pose':self.pose_observation_space,
                                       'topdown':self.topdown_observation_space,
                                       'label_topdown':self.topdown_observation_space,
                                       'laser_map':self.laser_map_space,
                                       'laser_point':self.laser_point_space,
                                       'action':self.action_space,
                                       'shift':self.shift_observation_space})
        self.mapper = CoG_Mapper()
        self.performance = []
        self.seeds = []
        self.env = CogEnvDecoder(worker_id=self.worker_id,env_name=self.env_sim_name,time_scale=self.time_scale,no_graphics=True)
        
    def _rescale_angle(self,angle):
        if angle < -np.pi:
            return angle + 2*np.pi
        elif angle > np.pi:
            return angle - 2*np.pi
        return angle
    
    def _vector_angle(self,vector_start=np.zeros((2,),np.float32),vector_end=np.zeros((2,),np.float32)):
        direction_vector = np.array(vector_end) - np.array(vector_start)
        direction_vector = direction_vector/np.linalg.norm(direction_vector)
        angle = np.arctan(direction_vector[1]/(np.sign(direction_vector[0])*np.clip(np.abs(direction_vector[0]),1e-5,1)))
        if direction_vector[0] < 0 and direction_vector[1] < 0:
            angle = angle - np.pi
        elif direction_vector[0] < 0 and direction_vector[1] >0:
            angle = angle + np.pi
        return angle
    
    def _distance(self,poseA,poseB):
        return np.sqrt(np.sum(np.square(poseA-poseB)))
    def _angle_distance(self,angleA,angleB):
        return min(np.abs(angleA-angleB),2*np.pi-np.abs(angleA-angleB))
    def _generate_agent_pose(self,vector):
        return np.array(vector[0][0:2])
    def _generate_agent_angle(self,vector):
        angle = self._rescale_angle(vector[0][2])
        return np.array([np.cos(angle),np.sin(angle)],np.float32)    
    def _generate_agent_info(self,vector):
        return np.array([vector[1][0],vector[1][1]],np.float32)
    def _generate_enemy_acti(self,vector):
        return np.array(vector[2])
    def _generate_enemy_pose(self,vector):
        return np.array(vector[3][0:2])
    def _generate_enemy_angle(self,vector):
        angle = self._rescale_angle(vector[3][2])
        return np.array([np.cos(angle),np.sin(angle)],np.float32)
    def _generate_enemy_info(self,vector):
        return np.array([vector[4][0],vector[4][1]],np.float32)
    def _generate_marker_info(self,vector):
        return np.array(vector[5:10],np.float32)
    
    def _noisy_pose(self):
        noisy_pose = self.agent_pose + self.episode_pose_noise + np.random.uniform(-self.step_pose_noise_scale,self.step_pose_noise_scale,2)
        return noisy_pose
    def _noisy_angle(self):
        noisy_angle = self.agent_angle + np.random.uniform(-self.step_angle_noise_scale,self.step_angle_noise_scale,2)
        return noisy_angle
    def _noisy_laser(self):
        noise_laser = self.agent_laser + np.random.uniform(-self.uniform_laser_noise_scale,self.uniform_laser_noise_scale,self.laser_observation_space.shape[0])
        return noise_laser
    def _generate_laser_observation(self):
        return self._noisy_laser()
    def _generate_pose_observation(self):
        return np.concatenate((self._noisy_pose(),self._noisy_angle()),axis=0)
    def _generate_goal_observation(self):
        marker_activation = self.marker_info[:,2].astype(np.int32)
        marker_position = self.marker_info[:,0:2]
        enemy_activation = self.enemy_acti
        enemy_position = self.enemy_pose
        enemy_angle = self.enemy_angle
        if enemy_activation:
            return np.concatenate((enemy_position,enemy_angle),axis=0)
        else:
            marker_index = np.sum(marker_activation)
            marker_position = marker_position[marker_index]
            marker_angle = self._vector_angle(vector_start=marker_position,vector_end=self.pose_observation[0:2])
            marker_angle = np.array([np.cos(marker_angle),np.sin(marker_angle)],np.float32)
            return np.concatenate((marker_position,marker_angle),axis=0)
    
    def _generate_laser_map(self,noisy_laser):
        # -135 degree to 135 degree, 60 data
        degree_unit = 270 / 60
        laser_map = np.ones((self.resize_scale,self.resize_scale,3),np.float32)
        cv2.rectangle(laser_map,pt1=(self.resize_scale//2-3,self.resize_scale//2-3),pt2=(self.resize_scale//2+3,self.resize_scale//2+3),color=(0,255,0),thickness=-1)
        for i in range(60):
            degree = -135 + i*degree_unit
            laser_length = noisy_laser[i] * (self.resize_scale/2)#noisy_laser[i] / 2.0 * (self.topdown_size/2)
            pixel_x = int(self.resize_scale / 2 - laser_length * np.cos((degree+90)/180*np.pi))
            pixel_y = int(self.resize_scale / 2 - laser_length * np.sin((degree+90)/180*np.pi))
            if pixel_x >= 5 and pixel_x < self.resize_scale - 5 and pixel_y >= 5 and pixel_y <= self.resize_scale - 5: 
                laser_map[pixel_y,pixel_x] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        laser_map = cv2.erode(laser_map,kernel,iterations = 3)
        return laser_map
    
    def _generate_laser_point(self,noisy_laser):
        # -135 degree to 135 degree, 60 data
        degree_unit = 270 / 60
        laser_point = np.zeros((60,2))
        for i in range(60):
            degree = -135 + i*degree_unit
            laser_length = noisy_laser[i] 
            x = - laser_length * np.cos((degree+90)/180*np.pi)
            y = - laser_length * np.sin((degree+90)/180*np.pi)
            laser_point[i] = np.array([x/2.,y/2.],np.float32)
        return laser_point
    
    def _generate_topdown_map(self,noisy_position,noisy_rotation,goal_position,goal_rotation):
        true_agent_x = int(self.agent_pose[0]*100)
        true_agent_y = int((4.48-self.agent_pose[1])*100)
        agent_x = int(noisy_position[0]*100)
        agent_y = int((4.48-noisy_position[1])*100)
        goal_x = int(goal_position[0]*100)
        goal_y = int((4.48-goal_position[1])*100)
        agent_rad_angle = self._vector_angle(vector_end = noisy_rotation)
        agent_deg_angle = np.rad2deg(-agent_rad_angle+np.pi/2)
        scene_map = self.mapper.dense_map.copy()
        scene_map = cv2.line(scene_map,pt1=(agent_x,agent_y),pt2=(int(agent_x + 50*noisy_rotation[0]),int(agent_y - 50*noisy_rotation[1])),color=(255,0,0),thickness=8)
        if self.env_id == 'CoG-Navigation':
            scene_map = cv2.line(scene_map,pt1=(goal_x,goal_y),pt2=(int(goal_x),int(goal_y - 50*1)),color=(0,0,255),thickness=8)
            scene_map = cv2.line(scene_map,pt1=(goal_x,goal_y),pt2=(int(goal_x),int(goal_y + 50*1)),color=(0,0,255),thickness=8)
            scene_map = cv2.line(scene_map,pt1=(goal_x,goal_y),pt2=(int(goal_x - 50*1),int(goal_y)),color=(0,0,255),thickness=8)
            scene_map = cv2.line(scene_map,pt1=(goal_x,goal_y),pt2=(int(goal_x + 50*1),int(goal_y)),color=(0,0,255),thickness=8)
        else:
            raise NotImplementedError        
        scene_map = cv2.line(scene_map,pt1=(agent_x,agent_y),pt2=(int(goal_x),int(goal_y)),color=(0,255,0),thickness=5)
        #Transform
        corner_points = np.array([[0,0,1],[0,448,1],[808,0,1],[808,448,1]])
        rot_matrix = cv2.getRotationMatrix2D((404,224),agent_deg_angle,scale=1)
        transform_corner_points = np.matmul(rot_matrix,corner_points.transpose()).transpose()
        new_width = np.max(transform_corner_points[:,0]) - np.min(transform_corner_points[:,0])
        new_height = np.max(transform_corner_points[:,1]) - np.min(transform_corner_points[:,1])
        shift_x = -np.min(transform_corner_points[:,0])
        shift_y = -np.min(transform_corner_points[:,1])
        rot_matrix[0,2] += shift_x
        rot_matrix[1,2] += shift_y
        transform_topdown = cv2.warpAffine(scene_map,rot_matrix,dsize=(int(new_width),int(new_height)))
        transform_true_center = np.matmul(rot_matrix,np.array([true_agent_x,true_agent_y,1]).transpose()).transpose()
        transform_agent_center = np.matmul(rot_matrix,np.array([agent_x,agent_y,1]).transpose()).transpose()
        start_x = max(transform_agent_center[0] - self.topdown_size//2,0) 
        start_y = max(transform_agent_center[1] - self.topdown_size//2,0)
        end_x = min(transform_agent_center[0] + self.topdown_size//2, new_width)
        end_y = min(transform_agent_center[1] + self.topdown_size//2, new_height)
        x_pl = np.abs(transform_agent_center[0] - self.topdown_size//2 - start_x)
        x_pr = np.abs(transform_agent_center[0] + self.topdown_size//2 - end_x)
        y_pt = np.abs(transform_agent_center[1] - self.topdown_size//2 - start_y)
        y_pb = np.abs(transform_agent_center[1] + self.topdown_size//2 - end_y)
        crop_topdown = transform_topdown[int(start_y):int(end_y),int(start_x):int(end_x)]
        crop_topdown = np.pad(crop_topdown,((int(y_pt),int(y_pb)),(int(x_pl),int(x_pr)),(0,0)),mode='constant',constant_values=0)
        crop_topdown = cv2.resize(crop_topdown,(self.resize_scale,self.resize_scale))
        crop_topdown = crop_topdown/255.0
        shift = (transform_true_center - transform_agent_center)/(self.topdown_size/2)
        return crop_topdown,shift
    
    def _navigation_arrival(self):
        return self.last_activation_num < self.current_activation_num
    def _navigation_reward_function(self):
        time_penalty = -0.01
        collision_penalty = -0.05*(self.current_collision > self.last_collision)
        angle_reward = 1.0*(self.last_angle_distance - self.current_angle_distance)*(1-self._navigation_arrival())
        distance_reward = 1.0*(self.last_euclidean_distance - self.current_euclidean_distance)*(1-self._navigation_arrival())
        activate_reward = 4.0 * (self._navigation_arrival())
        return collision_penalty + distance_reward + angle_reward + time_penalty + activate_reward
    def _navigation_done(self):
        done = (self.episode_length >= self.timelimit or self.current_activation_num==5)
        return done
    def _navigation_info(self):
        return {'collision':self.current_collision,
                'activation':self.current_activation_num,
                'length':self.episode_length*0.04}
    def _navigation_action_transform(self,action):
        clip_action = np.clip(action,-1,1)
        transform_action = np.zeros((4,),np.float32)
        transform_action[0] = clip_action[0] * self.speed_scale
        transform_action[1] = clip_action[1] * self.speed_scale
        transform_action[2] = clip_action[2] * np.pi/4
        return transform_action

    def reset(self):
        #self.make_environment()
        self.episode_length = 0
        self.episode_score = 0
        
        obs = self.env.reset()
        self.agent_image = obs['color_image']
        self.agent_laser = obs['laser'][0:60]
        self.agent_pose = self._generate_agent_pose(obs['vector'])
        self.agent_angle = self._generate_agent_angle(obs['vector'])
        self.agent_info = self._generate_agent_info(obs['vector'])
        self.enemy_acti = self._generate_enemy_acti(obs['vector'])
        self.enemy_pose = self._generate_enemy_pose(obs['vector'])
        self.enemy_angle = self._generate_enemy_angle(obs['vector'])
        self.enemy_info = self._generate_enemy_info(obs['vector'])
        self.marker_info = self._generate_marker_info(obs['vector'])
        
        self.episode_pose_noise = np.random.uniform(-self.episode_pose_noise_scale,self.episode_pose_noise_scale,2)
        self.laser_observation = self._generate_laser_observation()
        self.pose_observation = self._generate_pose_observation()
        self.goal_observation = self._generate_goal_observation()
        self.laser_map_observation = self._generate_laser_map(self.laser_observation)
        self.laser_point_observation = self._generate_laser_point(self.laser_observation)
        
        self.mapper.reset(self.marker_info,self.enemy_pose,self.goal_observation)
        self.topdown_observation,self.shift_observation = self._generate_topdown_map(self.pose_observation[0:2],
                                                                                     self.pose_observation[2:4],
                                                                                     self.goal_observation[0:2],
                                                                                     self.goal_observation[2:4])
        self.label_topdown_observation,_ = self._generate_topdown_map(self.agent_pose,self.agent_angle,self.goal_observation[0:2],self.goal_observation[2:4])
        self.last_laser_observation = self.laser_observation
        self.last_topdown_observation = self.topdown_observation
        self.last_activation_num = 0
        self.last_collision = obs['vector'][-1][1]
        self.last_euclidean_distance = self._distance(self.agent_pose,self.goal_observation[0:2])
        self.last_angle_distance = self._angle_distance(self._vector_angle(vector_end=self.agent_angle),self._vector_angle(vector_end=self.goal_observation[2:4])+np.pi)
        self.last_pose = self.agent_pose
        self.last_angle = self.agent_angle
        self.last_agent_info = self.agent_info
        self.last_enemy_info = self.enemy_info
        self.last_action = np.zeros((4,),np.float32)
        self.last_reward = np.zeros((1,),np.float32)
        return {'laser':self.laser_observation/2.0,
                'goal':self.goal_observation,
                'pose':self.pose_observation,
                'topdown':self.topdown_observation,
                'action':self.last_action,
                'label_topdown':self.label_topdown_observation,
                'laser_map':self.laser_map_observation,
                'laser_point':self.laser_point_observation,
                'shift':self.shift_observation}
    
    def step(self,action):
        if self.env_id == "CoG-Navigation":
            obs,_,_,_ = self.env.step(self._navigation_action_transform(action))
        else:
            raise NotImplementedError
        self.agent_image = obs['color_image']
        self.agent_laser = obs['laser'][0:60]
        self.agent_pose = self._generate_agent_pose(obs['vector'])
        self.agent_angle = self._generate_agent_angle(obs['vector'])
        self.agent_info = self._generate_agent_info(obs['vector'])
        self.enemy_acti = self._generate_enemy_acti(obs['vector'])
        self.enemy_pose = self._generate_enemy_pose(obs['vector'])
        self.enemy_angle = self._generate_enemy_angle(obs['vector'])
        self.enemy_info = self._generate_enemy_info(obs['vector'])
        self.marker_info = self._generate_marker_info(obs['vector'])
        
        self.laser_observation = self._generate_laser_observation()
        self.pose_observation = self._generate_pose_observation()
        self.goal_observation = self._generate_goal_observation()
        self.laser_map_observation = self._generate_laser_map(self.laser_observation)
        self.laser_point_observation = self._generate_laser_point(self.laser_observation)
        
        self.mapper.step(self.marker_info,self.enemy_pose,self.goal_observation)
        self.topdown_observation,self.shift_observation = self._generate_topdown_map(self.pose_observation[0:2],
                                                                                    self.pose_observation[2:4],
                                                                                    self.goal_observation[0:2],
                                                                                    self.goal_observation[2:4])
        self.label_topdown_observation,_ = self._generate_topdown_map(self.agent_pose,self.agent_angle,self.goal_observation[0:2],self.goal_observation[2:4])
        self.current_collision = obs['vector'][10][1]
        self.current_euclidean_distance = self._distance(self.agent_pose,self.goal_observation[0:2])
        self.current_activation_num = np.sum(self.marker_info[:,2])
        self.current_angle_distance = self._angle_distance(self._vector_angle(vector_end=self.agent_angle),self._vector_angle(vector_end=self.goal_observation[2:4])+np.pi)
        self.episode_length += 1
         
        if self.env_id == "CoG-Navigation":
            reward = self._navigation_reward_function()
            done = self._navigation_done()
            info = self._navigation_info()
        else:
            raise NotImplementedError
        self.episode_score += reward
        if done:
            print("activation:{},collision:{}".format(self.current_activation_num,self.current_collision))
            if self.seed_num in self.seeds:
                index = self.seeds.index(self.seed_num)
                self.performance[index] = self.episode_score
            else:
                self.performance.append(self.episode_score)
                self.seeds.append(self.seed_num)
            self.episode_score = 0
        self.last_action = action
        self.last_reward = [reward]
        self.last_laser_observation = self.laser_observation
        self.last_topdown_observation = self.topdown_observation
        self.last_collision = self.current_collision
        self.last_euclidean_distance = self.current_euclidean_distance
        self.last_activation_num = self.current_activation_num
        self.last_angle_distance = self.current_angle_distance
        self.last_pose = self.agent_pose
        self.last_angle = self.agent_angle
        self.last_agent_info = self.agent_info
        self.last_enemy_info = self.enemy_info
        return {'laser':self.laser_observation/2.0,
                'goal':self.goal_observation,
                'pose':self.pose_observation,
                'topdown':self.topdown_observation,
                'action':self.last_action,
                'label_topdown':self.label_topdown_observation,
                'laser_map':self.laser_map_observation,
                'laser_point':self.laser_point_observation,
                'shift':self.shift_observation},reward,done,info