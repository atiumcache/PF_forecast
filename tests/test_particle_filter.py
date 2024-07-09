import unittest
from unittest import mock
from unittest.mock import call, MagicMock
from particle_filter import *
import os
import pandas as pd
from filter_forecast.algo_init import *
from particle_filter import log_pf
from particle_filter.log_pf import GlobalParameters, Particles
import jax.numpy as jnp

class TestParticleFilter(unittest.TestCase):

    def test_pf_init(self):
        global_params = GlobalParameters(num_particles=100,
                                         population=100000,
                                         time_steps=100,
                                         beta_prior=(0.1, 0.15))
        particles = Particles(gp=global_params)
        self.assertEqual(len(particles.betas), 100)