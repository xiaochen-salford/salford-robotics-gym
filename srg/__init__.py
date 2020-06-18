import distutils.version
import os, sys, warnings

from . import error
from .version import VERSION as __version__

from .core import Env, GoalEnv, Wrapper, ObservationWrapper, ActionWrapper, RewardWrapper
from .spaces import Space
from .envs import make, spec, register
from . import logger, vector

__all__ = ["Env", "Space", "Wrapper", "make", "spec", "register"]