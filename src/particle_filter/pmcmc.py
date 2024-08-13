import jax.numpy as jnp
from jax.numpy.linalg import cholesky
from jax.typing import ArrayLike
from typing import Callable
import jax.random as random


class PMCMC:
    """Represents a Particle MCMC Algorithm."""

    def __init__(self, iterations: int, init_theta: ArrayLike, prior: Callable):
        self.num_params = len(init_theta)
        self.iterations = iterations
        self.prior = prior
        self.key = random.PRNGKey(47)

        self.mle_betas = None
        self.mle_hospitalizations = None
        self.mle_states = None
        self.max_likelihood = float("-inf")

        self.thetas = jnp.zeros((self.num_params, iterations))
        self.likelihoods = jnp.zeros(iterations)
        self.accept_record = jnp.zeros(iterations)

        self.mu = jnp.zeros(self.num_params)
        self.cov = jnp.eye(self.num_params)

        self.thetas[:, 0] = init_theta
        self.likelihoods[0] = prior(init_theta)

        if jnp.isfinite(self.likelihoods[0]):
            # TODO: insert filter logic here
            pass

    def run_pmcmc(self) -> None:
        """Runs the MCMC algorithm.

        Returns:
            None. Quantities of interest are accessible via the instance variables.
        """

        for i in range(1, self.iterations):

            theta_prev = self.thetas[:, i - 1]
            self.key, subkey = random.split(self.key)
            random_params = random.normal(key=subkey, shape=(self.num_params))
            chol_matrix = cholesky((2.38**2 / self.num_params) * cov)
            theta_prop = theta_prev + chol_matrix @ random_params
            
            new_likelihood = prior(theta_prop)
            
            if jnp.isfinite(new_likelihood):
                # TODO: Implement filter logic
                # TODO: Filter needs to output likelihood, amongst other variables/arrays
                pf_likelihood, pf_hosp_estimates, pf_states, pf_betas = None, None, None, None
                new_likelihood += jnp.sum(pf_likelihood)

                if new_likelihood > self.max_likelihood:
                    self.update_new_mle(new_likelihood, pf_hosp_estimates, pf_states, pf_betas)

                self.accept_reject(theta_prop, theta_prev, new_likelihood)
            else:
                # Reject automatically, because the ratio will be negative infinity
                self.reject_proposal()
            
            self.update_cov(i)

    def accept_reject(self, theta_prop, theta_prev, new_likelihood):
        ratio = (new_likelihood - self.likelihoods[i - 1])
        self.key, subkey = random.split(self.key)
        u = random.uniform(key=subkey, minval=0, maxval=1)

    def accept_proposal(self):
        pass
    
    def reject_proposal(self):
        self.thetas[:, i] = self.thetas[:, i - 1]
        self.likelihoods[i] = self.likelihoods[i - 1]
    
    def update_cov(self, current_iter):
        pass

    def update_new_mle(self, new_likelihood, particle_estimates, particle_states, particle_betas):
        self.max_likelihood = new_likelihood
        self.mle_hospitalizations = particle_estimates
        self.mle_states = particle_states
        self.mle_betas = particle_betas


            
            
            
