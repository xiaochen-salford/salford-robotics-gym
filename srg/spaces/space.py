# Based on https://raw.githubusercontent.com/openai/gym/master/gym/spaces/space.py

from srg.utils import seeding

class Space:
  """
  Defines the observation and action spaces, so you can write generic
  code that applies to any Env. For example, you can choose a random
  action.
  """
  def __init__(self, shape=None, dtype=None):
    import numpy as np  # takes about 300-400ms to import, so we load lazily
    self.shape = None if shape is None else tuple(shape)
    self.dtype = None if dtype is None else np.dtype(dtype)
    self._np_random = None

  @property
  def np_random(self):
    """Lazily seed the rng since this is expensive and only needed if
    sampling from this space.
    """
    if self._np_random is None:
        self.seed()

    return self._np_random

  def sample(self):
    """Randomly sample an element of this space. Can be 
    uniform or non-uniform sampling based on boundedness of space."""
    raise NotImplementedError

  def seed(self, seed=None):
    """Seed the PRNG of this space. """
    self._np_random, seed = seeding.np_random(seed)
    return [seed]

  def contains(self, x):
    """
    Return boolean specifying if x is a valid
    member of this space
    """
    raise NotImplementedError

  def __contains__(self, x):
    return self.contains(x)

  def to_jsonable(self, sample_n):
    """Convert a batch of samples from this space to a JSONable data type."""
    # By default, assume identity is JSONable
    return sample_n

  def from_jsonable(self, sample_n):
    """Convert a JSONable data type to a batch of samples from this space."""
    # By default, assume identity is JSONable
    return sample_n