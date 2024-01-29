from typing import Any
import jax.numpy as np

# define our own PRNGKeyT type to make it explicit that we are using JAX's PRNGKey type
Array = np.ndarray
PRNGKeyT = Any
