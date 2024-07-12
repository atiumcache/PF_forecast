import unittest
import numpy as np
import jax.numpy as jnp
import particle_filter.log_pf as log_pf
from particle_filter.init_settings import InitSettings
import particle_filter.setup_pf as setup_pf


class TestParticleCloud(unittest.TestCase):

    def setUp(self):
        self.settings = InitSettings(
            num_particles=100, population=1000, location_code='04', dt=1.0, seed_size=0.005
        )
        self.particle_cloud = log_pf.ParticleCloud(self.settings)

    def test_initialization(self):
        self.assertEqual(
            self.particle_cloud.states.shape[0], self.settings.num_particles
        )
        self.assertEqual(
            self.particle_cloud.weights.shape[0], self.settings.num_particles
        )
        self.assertEqual(
            self.particle_cloud.betas.shape[0], self.settings.num_particles
        )

    def test_initial_state(self):
        initial_state = self.particle_cloud.get_initial_state()
        self.assertEqual(len(initial_state), 5)
        self.assertAlmostEqual(sum(initial_state), self.settings.population, delta=1e-5)

    def test_update_single_particle(self):
        initial_state = self.particle_cloud.states[0].copy()
        self.particle_cloud._update_single_particle([1,2,3,4,5], 0, 0.1, self.settings.dt)
        updated_state = self.particle_cloud.states[0]
        self.assertFalse(jnp.array_equal(initial_state, updated_state))

    def test_update_all_particles(self):
        initial_states = self.particle_cloud.states.copy()
        self.particle_cloud.update_all_particles(0)
        updated_states = self.particle_cloud.states
        self.assertFalse(jnp.array_equal(initial_states, updated_states))

    def test_observation_class(self):
        hosp_cases = np.array([3, 5, 6, 3, 8, 9, 121, 7])
        observations = log_pf.ObservationData(observations=hosp_cases)
        self.assertEqual(observations.get_observation(2), 6)

if __name__ == "__main__":
    unittest.main()
