# Code for paper: A Minimalist Ensemble Policy Evaluation Operator for Deep Reinforcement Learning


## Install

### Optionally: Create a new python virtual environment 

We recommend readers use the anaconda tool.

```shell
conda create -n mepe python=3.7.4
```

And then activate the created environment

```shell
conda activate mepe
```

### 
Firstly, install the requirements.txt

```shell
pip install -r requirements.txt
```

The required python package is as follows: 
```shell
torch==1.7.1+cu101
numpy==1.18.1
gym==0.12.1
pybullet==2.7.1
roboschool==1.0.48
pandas==1.0.1
tqdm
scikit-image
tensorboard
torch
torchvision
gym==0.25.1
patchelf
termcolor
seaborn==0.9.0
opencv-python
kornia
```

## Run

For running ME-DDPG:

```shell
cd Bullet
python main_me_ddpg.py --env HopperBulletEnv-v0  --seed 0 --dropout_p 0.1
```

```bash
(mepe) python main_me_ddpg.py --env HopperBulletEnv-v0  --seed 0 --dropout_p 0.1

---------------------------------------
Policy: ME-DDPG, Env: HopperBulletEnv-v0, Seed: 0
---------------------------------------
3%|██▍                              | 28308/1000000 [00:28<59:30, 272.17it/s]
```


For running ME-SAC:
```shell
cd Bullet
python main_me_sac.py --env HopperBulletEnv-v0  --seed 0 --dropout_p 0.1
```
```bash
(mepe) python main_me_sac.py --env HopperBulletEnv-v0  --seed 0 --dropout_p 0.1
---------------------------------------
Policy: ME-SAC, Env: HopperBulletEnv-v0, Seed: 0
---------------------------------------
3%|██▌                          | 30202/1000000 [00:47<1:36:08, 168.12it/s]
```


For running ME-CURL
```shell
cd Atari
python main.py --game=ms_pacman
```

```shell
(mepe) python main.py --game=ms_pacman
Algorithm name:  mepe
2%|█▍                      | 1703/100000 [00:11<1:22:28, 19.86it/s]
```