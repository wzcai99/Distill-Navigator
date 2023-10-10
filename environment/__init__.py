from .cog_env import Cog_Env
from .cog_env import ENVIRONMENT_IDS as COG_ENVIRONMENTS

from .vector_env.vector_env import VecEnv
from .vector_env.dummy_vecenv import DummyVecEnv
from .vector_env.subproc_vecenv import SubprocVecEnv

from argparse import Namespace
from typing import Optional

def make_envs(env_id:str,
              seed:int,
              config:Optional[Namespace],
              ):
    def _thunk():
        if env_id in COG_ENVIRONMENTS:
            env = Cog_Env(env_id,seed,config)
        else:
            raise NotImplementedError
        return env
    parallels = config.parallels
    return SubprocVecEnv([_thunk for i in range(parallels)])
  