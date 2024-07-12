import unittest
import numpy as np
import jax.numpy as jnp
from particle_filter.log_pf import *

class TestParticleCloud(unittest.TestCase):

    def setUp(self):
        self.settings = InitSettings(
            num_particles=100,
            population=1000,
            time_steps=10,
            dt=1.0,
            seed_size=0.005
        )
        self.particle_cloud = ParticleCloud(self.settings)

    def test_initialization(self):
        self.assertEqual(self.particle_cloud.states.shape[0], self.settings.num_particles)
        self.assertEqual(self.particle_cloud.weights.shape[0], self.settings.num_particles)
        self.assertEqual(self.particle_cloud.betas.shape[0], self.settings.num_particles)

    def test_initial_state(self):
        initial_state = self.particle_cloud.get_initial_state()
        self.assertEqual(len(initial_state), 5)
        self.assertAlmostEqual(sum(initial_state), self.settings.population, delta=1e-5)

    def test_update_single_particle(self):
        initial_state = self.particle_cloud.states[0].copy()
        self.particle_cloud.update_single_particle(0, 0, self.settings.dt)
        updated_state = self.particle_cloud.states[0]
        self.assertFalse(jnp.array_equal(initial_state, updated_state))

    def test_update_all_particles(self):
        initial_states = self.particle_cloud.states.copy()
        self.particle_cloud.update_all_particles(0)
        updated_states = self.particle_cloud.states
        self.assertFalse(jnp.array_equal(initial_states, updated_states))

if __name__ == '__main__':
    unittest.main()
