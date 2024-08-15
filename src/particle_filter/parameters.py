import jax.numpy as jnp
import jax.random as random
import toml
from typing import Union, Optional, Dict, Any
from scipy.stats import norm


class Parameter:
    def __init__(self, value=None, dist=None, loc=None, scale=None, is_fixed=True):
        self.value = value
        self.dist = dist
        self.loc = loc
        self.scale = scale
        self.is_fixed = is_fixed

    def sample(self, key=None):
        if self.is_fixed:
            return self.value
        elif self.dist == "normal":
            self.value = norm(loc=self.value, scale=self.scale).rvs()
            return self.value
        else:
            raise ValueError(f"Unsupported distribution: {self.dist}")


class ModelParameters:
    def __init__(self, config_file: str):
        self._params = self.load_from_toml(config_file)

    def load_from_toml(self, config_file):
        # Load TOML file
        config = toml.load(config_file)
        model_params = config.get("model_params", {})

        params = {}
        for key, value in model_params.items():
            if "dist" in value:
                params[key] = Parameter(
                    value=value["loc"],
                    dist=value["dist"],
                    loc=value.get("loc"),
                    scale=value.get("scale"),
                    is_fixed=value.get("is_fixed", True),
                )
            else:
                params[key] = Parameter(
                    value=value["value"], is_fixed=value.get("is_fixed", True)
                )
        return params

    def __getattr__(self, name):
        if name in self._params:
            return self._params[name].value
        else:
            raise AttributeError(f"Parameter {name} does not exist.")

    def update_all(self, key=None):
        for key, param in self._params.items():
            if not param.is_fixed:
                self._params[key].value = param.sample(key)

    def __setattr__(self, name, value):
        if name == "_params":
            super().__setattr__(name, value)
        elif name in self._params:
            self._params[name].value = value
        else:
            raise AttributeError(f"Parameter {name} does not exist.")

    def update_param(self, key, value):
        """Update a parameter (key) with a new value."""
        if key in self._params:
            self._params[key].value = value
        else:
            raise AttributeError(f"Parameter {key} does not exist.")

