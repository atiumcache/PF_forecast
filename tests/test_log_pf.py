import unittest
import numpy as np
import jax.numpy as jnp
from filter_forecast.particle_filter import log_pf
from filter_forecast.particle_filter.init_settings import InitSettings


class TestParticleCloud(unittest.TestCase):

    def setUp(self):
        self.settings = InitSettings(
            num_particles=10,
            population=1000,
            location_code="04",
            dt=1.0,
            seed_size=0.005,
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
        self.particle_cloud._update_single_particle(
            [1, 2, 3, 4, 5], 0, 0.1, self.settings.dt
        )
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

    def test_compute_single_weight(self):
        reported_data = 17
        particle_estimates = [17, 23]
        weight1 = self.particle_cloud._compute_single_weight(reported_data,
                                                             particle_estimates[0])
        weight2 = self.particle_cloud._compute_single_weight(reported_data,
                                                             particle_estimates[1])
        # better estimate should have bigger weight
        self.assertTrue(weight1 > weight2, msg="The better estimate has a "
                                               "lower weight.")
        self.assertIsInstance(weight1, float)
        self.assertIsInstance(weight2, float)

    def test_compute_all_weights(self):
        self.particle_cloud.hosp_estimates = jnp.zeros(
            self.particle_cloud.settings.num_particles)
        self.particle_cloud.hosp_estimates = jnp.ones(self.particle_cloud.settings.num_particles)
        best_estimate_index = 1
        self.particle_cloud.hosp_estimates = (
            self.particle_cloud.hosp_estimates.at[best_estimate_index].set(10))
        reported_data = 20
        self.particle_cloud.compute_all_weights(reported_data)
        max_index = jnp.argmax(self.particle_cloud.weights)
        self.assertEqual(max_index, best_estimate_index, "The best estimate "
                                                         "does not have the "
                                                         "highest weight.")


if __name__ == "__main__":
    unittest.main()
