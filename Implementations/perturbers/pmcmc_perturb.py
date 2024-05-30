from typing import Dict, List

import numpy as np

from Abstract.Perturb import Perturb
from utilities.Utils import ESTIMATION, Context, Particle, jacob


class PMMH_Perturb(Perturb):
    def __init__(self, hyper_params: Dict) -> None:
        """The perturbation scheme defined in Calvetti et. al. Uses the log-normal perturbations in the linear-domain for both the static parameters and the
        time-dependent parameters."""
        super().__init__(hyper_params=hyper_params)

    def randomly_perturb(
        self, ctx: Context, particleArray: List[Particle]
    ) -> List[Particle]:

        static_param_mat = []
        static_names = []

        var_param_mat = []
        var_names = []

        for particle in particleArray:
            static_params = []
            variable_params = []

            for _, (key, val) in enumerate(particle.param.items()):

                """TODO I feel this is slow as well, intializing new lists each iteration is inefficient and the parameter search could potentially be made more
                efficient. Maybe there's a better way to store parameters than a dictionary. Looping over the entire dictionary each time is problematic, we could
                hold the information in a smarter data structure."""

                if key in ctx.estimated_params:

                    if ctx.estimated_params[key] == ESTIMATION.STATIC:
                        static_names.append(key)
                        static_params.append(val)

                    elif ctx.estimated_params[key] == ESTIMATION.VARIABLE:
                        var_names.append(key)
                        variable_params.append(val)

            static_param_mat.append(static_params)
            var_param_mat.append(variable_params)

        # #names of the estimated parameters
        var_names = np.unique(np.array(var_names))
        static_names = np.unique(np.array(static_names))

        # '''Needs to be a numpy array for proper indexing. Did the log operation up front to amortize the cost.'''
        # static_param_mat = np.log(np.array(static_param_mat))
        # var_param_mat = np.log(np.array(var_param_mat))

        # '''Setting up the main diagonal of the matrix based on the construction used in Calvetti et. al.'''
        # state1 = np.array([self.hyperparameters['sigma1']/ctx.population]) ** 2

        # otherstates = np.array([self.hyperparameters['sigma1']**2 for _ in range(ctx.state_size-1)])

        # param_variance = np.array([self.hyperparameters['sigma2']**2 for _ in range(len(var_names))])

        # C = np.concatenate((state1,otherstates,param_variance))

        # '''Generate diagonal matrix.'''

        # C = np.diag(C).astype(float)

        # '''Main perturbation loop'''
        # for i in range(len(particleArray)):
        #     log_state = np.log(particleArray[i].state)
        #     td_vec = np.concatenate((log_state,var_param_mat[i])) #Everything in here is log
        #     perturbed = np.exp(ctx.rng.multivariate_normal(td_vec,C))

        #     '''Normalization

        #     These steps ensure the population stays constant, we basically just normalize and multiply by the population.

        #     TODO I wonder if this step causes significant bias in the perturbation. Might be worth evaluating.

        #     '''
        #     perturbed[0:ctx.state_size] /= np.sum(perturbed[0:ctx.state_size])
        #     perturbed[0:ctx.state_size] *= ctx.population

        #     '''Returns the perturbed values to the particles. Dependent on the ctx.state_size value.'''
        #     particleArray[i].state = perturbed[:ctx.state_size]
        #     for j,name in enumerate(var_names):
        #         particleArray[i].param[name] = perturbed[ctx.state_size+j:ctx.state_size+j+1][0]

        for particle in particleArray:
            for name in var_names:
                particle.param[name] = np.exp(
                    ctx.rng.normal(
                        np.log(particle.param[name]), self.hyperparameters["sigma2"]
                    )
                )

        return particleArray
