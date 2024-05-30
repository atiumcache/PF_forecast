from typing import Callable, Dict

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import cm

from Abstract.Algorithm import Algorithm
from Abstract.Integrator import Integrator
from Abstract.Perturb import Perturb
from Abstract.Resampler import Resampler
from utilities.sankey import visualize_particles
from utilities.Utils import *
from utilities.Utils import Context


class TimeDependentAlgo(Algorithm):
    """Main particle filtering algorithm as described in Calvetti et. al."""

    def __init__(
        self,
        integrator: Integrator,
        perturb: Perturb,
        resampler: Resampler,
        ctx: Context,
    ) -> None:
        """Constructor passes back to the parent, nothing fancy here"""
        super().__init__(integrator, perturb, resampler, ctx)

    def initialize(self, params: Dict[str, int], priors: Dict[str, Callable]) -> None:
        """Initialize the parameters and their flags for estimation"""
        for _, (key, val) in enumerate(params.items()):
            if val == ESTIMATION.STATIC:
                self.ctx.estimated_params[key] = ESTIMATION.STATIC
            elif val == ESTIMATION.VARIABLE:
                self.ctx.estimated_params[key] = ESTIMATION.VARIABLE

        for _ in range(self.ctx.particle_count):
            """Setup the particles at t = 0, important we make a copy of the params dictionary before using it isto setup each particle."""

            p_params = params.copy()
            """Call the priors to generate values for the estimated params and set their values in the new params."""
            for _, (key, val) in enumerate(self.ctx.estimated_params.items()):
                if (p_params[key] == ESTIMATION.STATIC) or (
                    p_params[key] == ESTIMATION.VARIABLE
                ):
                    p_params[key] = priors[key]()

            seeds = []
            """Generate seeds from U(0,seed_size*pop) in the length of the seed loc array"""
            for _ in range(len(self.ctx.seed_loc)):
                seeds.append(
                    self.ctx.rng.uniform(0, self.ctx.seed_size * self.ctx.population)
                )

            state = np.concatenate(
                (
                    np.array([self.ctx.population], dtype=np.float_),
                    [0 for _ in range(self.ctx.state_size - 1)],
                )
            )
            for i in range(len(seeds)):
                state[self.ctx.seed_loc[i]] += seeds[i]
                state[0] -= seeds[i]

            self.particles.append(
                Particle(
                    param=p_params,
                    state=state.copy(),
                    observation=np.array(
                        [0 for _ in range(self.ctx.forward_estimation)]
                    ),
                )
            )

    def run(self, df: pd.DataFrame, runtime: int) -> None:
        """
        Inputs: time series data and runtime.
        Outputs: csv files for parameter and state values.
        """
        data1 = df.to_numpy()
        data1 = np.delete(data1, 0, 1)

        "Initialize labels and first column of sankey matrix"
        if self.ctx.run_sankey == True:
            self.ctx.sankey_indices.append(np.arange(self.ctx.particle_count))

        """Arrays to hold all the output data"""
        eta_quantiles = []

        state = []
        LL = []
        ESS = []
        gamma_quantiles = []

        state_quantiles = []
        beta_quantiles = []
        beta = []
        eta = []
        q = []
        q_quantiles = []
        observations = []
        gamma = []

        while self.ctx.clock.time < runtime:

            self.particles = self.integrator.propagate(self.particles, self.ctx)

            obv = data1[
                self.ctx.clock.time : self.ctx.clock.time
                + (self.ctx.forward_estimation)
            ]

            self.ctx.weights = self.resampler.compute_weights(
                self.ctx, obv, self.particles
            )

            self.particles = self.resampler.resample(self.ctx, self.particles)

            self.particles = self.perturb.randomly_perturb(self.ctx, self.particles)

            beta_quantiles.append(
                quantiles([particle.param["beta"] for particle in self.particles])
            )
            beta.append(
                np.mean([particle.param["beta"] for particle in self.particles])
            )

            state.append(
                np.mean([particle.state for particle in self.particles], axis=0)
            )
            eta_quantiles.append(
                quantiles([particle.param["eta"] for particle in self.particles])
            )
            eta.append(np.mean([particle.param["eta"] for particle in self.particles]))

            gamma_quantiles.append(
                quantiles([particle.param["gamma"] for particle in self.particles])
            )
            gamma.append(
                np.mean([particle.param["gamma"] for particle in self.particles])
            )
            observations.append(
                quantiles([particle.observation for particle in self.particles])
            )

            print(f"Iteration: {self.ctx.clock.time}", end="\r")
            self.ctx.clock.tick()

        pd.DataFrame(beta).to_csv("./datasets/average_beta.csv")
        pd.DataFrame(eta).to_csv("./datasets/average_eta.csv")
        pd.DataFrame(gamma).to_csv("./datasets/average_gamma.csv")

        pd.DataFrame(beta_quantiles).to_csv("./datasets/beta_quantiles.csv")
        pd.DataFrame(eta_quantiles).to_csv("./datasets/eta_quantiles.csv")
        pd.DataFrame(gamma_quantiles).to_csv("./datasets/gamma_quantiles.csv")
        pd.DataFrame(observations).to_csv("./datasets/particle_observation.csv")

        pd.DataFrame(state).to_csv("./datasets/ESTIMATED_STATE.csv")

        state_quantiles = np.array(state_quantiles)
        beta_quantiles = np.array(beta_quantiles)
        eta_quantiles = np.array(eta_quantiles)
        gamma_quantiles = np.array(gamma_quantiles)

        # sankey test code
        if self.ctx.run_sankey == True:
            visualize_particles(self.ctx.particle_count, self.ctx.sankey_indices)
