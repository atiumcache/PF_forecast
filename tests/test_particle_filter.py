import unittest
from unittest import mock
from unittest.mock import call, MagicMock
from particle_filter import *
import os
import pandas as pd
from filter_forecast.algo_init import *
from particle_filter import log_pf
from particle_filter.log_pf import InitSettings, ParticleCloud
import jax.numpy as jnp


class TestParticleFilter(unittest.TestCase):

    def test_pf_init(self):
        global_params = InitSettings(
            num_particles=100, population=100000, time_steps=100, beta_prior=(0.1, 0.15)
        )
        particles = ParticleCloud(gp=global_params)
        n = global_params.num_particles
        self.assertEqual(len(particles.betas), n)
        self.assertEqual(len(particles.states), n)
        self.assertEqual(len(particles.weights), n)
        self.assertIsInstance(particles.weights, jnp.ndarray)
        self.assertIsInstance(particles.betas, jnp.ndarray)
        self.assertIsInstance(particles.states, jnp.ndarray)

    def test_update_particle(self):
