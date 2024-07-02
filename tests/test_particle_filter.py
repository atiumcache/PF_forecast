import unittest
from unittest import mock
from unittest.mock import call, MagicMock
from particle_filter import *
import os
import pandas as pd
from filter_forecast.algo_init import *
from Abstract.Resampler import Resampler
from Abstract.Perturb import Perturb

class TestParticleFilter(unittest.TestCase):

    @mock.patch('filter_forecast.state.State.load_hospital_data', return_value=pd.DataFrame())
    def test_state_instance(self, mock_load_hospital_data):
        os.chdir('..')
        state = State('4')
        self.assertEqual(state.location_code, '04')
        self.assertEqual(state.population, 7151502)
        mock_load_hospital_data.assert_called_once()

    def test_get_data_since_week_26(self):
        pass

    def test_initialize_algo(self):
        state_population = 1000000
        loc_code = "04"

        algorithm = initialize_algo(state_population, loc_code)

        self.assertIsInstance(algorithm, TimeDependentAlgo)

        self.assertIsInstance(algorithm.integrator, LSODASolver)

        self.assertIsInstance(algorithm.perturb, Perturb)

        self.assertIsInstance(algorithm.resampler, Resampler)

        # Check context properties
        self.assertEqual(algorithm.ctx.location_code, loc_code)
        self.assertEqual(algorithm.ctx.population, state_population)
        np.testing.assert_array_equal(algorithm.ctx.weights, np.ones(1000))
        self.assertEqual(algorithm.ctx.seed_loc, [1])
        self.assertEqual(algorithm.ctx.seed_size, 0.005)
        self.assertEqual(algorithm.ctx.forward_estimation, 1)
        self.assertIsInstance(algorithm.ctx.rng, np.random.Generator)
        self.assertIsInstance(algorithm.ctx.particle_count, int)
