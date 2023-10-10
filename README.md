# Robust Navigation with Cross-Modal Fusion and Knowledge Transfer
Recently, learning-based approaches show promising results in navigation tasks. However, the poor generalization capability and the simulation-reality gap prevent a wide range of applications. We consider the problem of improving the gen- eralization of mobile robots and achieving sim-to-real transfer for navigation skills. To that end, we propose a cross-modal fusion method and a knowledge transfer framework for better generalization. This is realized by a teacher-student distillation architecture. The teacher learns a discriminative representation and the near-perfect policy in an ideal environment. By imitat- ing the behavior and representation of the teacher, the student is able to align the features from noisy multi-modal input and reduce the influence of variations on navigation policy. We evaluate our method in simulated and real-world environments. Experiments show that our method outperforms the baselines by a large margin and achieves robust navigation performance with varying working conditions.

<img width="898" alt="image" src="https://github.com/wzcai99/Distill-Navigator/assets/115710611/fe932c15-e28d-4720-aa75-b9b8fb975f61">

> [**Robust Navigation with Cross-Modal Fusion and Knowledge Transfer**](https://arxiv.org/pdf/2309.13266.pdf). 
> Wenzhe Cai*, Guangran Cheng*, Lingyue Kong, Lu Dong, Changyin Sun (*equal contribution). IEEE Conference on Robotics and Automation (ICRA), 2023.
> [**Website**](https://sites.google.com/view/distillnav)
### Installation ###
Step 1: Git clone this repository
```
git clone https://github.com/wzcai99/Distill-Navigator.git
cd Distill-Navigator
```
Step 2: Download the simulator
```
wget https://github.com/DRL-CASIA/COG-sim2real-challenge/releases/download/v3.0/reality_linux_v3.0.zip 
unzip reality_linux_v3.0
chmod +x reality_linux_v3.0/cog_sim2real_env.x86_64
```
Step 3: Create the conda environment and install the dependencies
```
conda env create -f environment.yml
```

### Training ###
Step 1: Train the Teacher Module on perfect environment
```
python teacher_train.py
```
Step 2: Train the Student Module on noisy environment
```
python student_train.py
```
Before runnning the step 2, please first modify the 'teacher_model' in teacher_config.yaml.
### Evaluation ###
Step 1: Evaluate the Student Module on different settings
```
python student_evaluate.py
```
### Notes ###
You can also do the same steps for training a baseline method (RL + Domain Randomization).
If you use a headless server for training and evaluation, make sure the following code exists in the main python file
```
os.environ['DISPLAY'] = ":1"
```
If not, please remove this code in the main file for monitoring the training/evaluation process.
Before evaluating the model, remember to modify the 'modeldir' and 'model_name' in the config file and make it compatiable with the saved model path. To change the environment noise settings, you can also finish it by changing the "speed_scale", "episode_pose_noise_scale" etc in the config file.

### Acknowledgement ###
We would like to thank all the contributors and organizers from CASIA for their efforts in building simulation and real-world robot platforms. 

### Citation ###
If you find our distill-navigator is useful for your project or research, please cite our paper:
```
@article{Cai2023RobustNW,
  title={Robust Navigation with Cross-Modal Fusion and Knowledge Transfer},
  author={Wenzhe Cai and Guangran Cheng and Lingyue Kong and Lu Dong and Changyin Sun},
  journal={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023},
  pages={10233-10239}
}
```


