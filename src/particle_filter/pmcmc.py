import jax.numpy as jnp
from jax.numpy.linalg import cholesky
from jax.typing import ArrayLike
from typing import Callable
import jax.random as random
from typing import Dict


class PMCMC:
    def __init__(self, iterations: int, init_theta: Dict, prior: Callable) -> None:
        """Initializes a Particle MCMC algorithm.

        Args:
            iterations: Number of MCMC iterations.
            init_theta: Initial proposal for theta,
                the parameter vector that will be passed into the particle filter.
            prior: Prior distribution function.
        """
        self._num_params = len(init_theta)
        self._iterations = iterations
        self._prior = prior
        self._key = random.PRNGKey(47)

        self._mle_betas = None
        self._mle_hospitalizations = None
        self._mle_states = None
        self._max_likelihood = float("-inf")

        self._thetas = jnp.zeros((self._num_params, iterations))
        self._likelihoods = jnp.zeros(iterations)
        self._accept_record = jnp.zeros(iterations)

        self._mu = jnp.zeros(self._num_params)
        self._cov = jnp.eye(self._num_params)

        self._thetas[:, 0] = init_theta.values()
        self._likelihoods[0] = prior(init_theta)

        self.theta_dictionary_template = init_theta

        if jnp.isfinite(self.likelihoods[0]):
            # TODO: insert filter logic here
            pass

    def run_algo(self) -> None:
        """Runs the MCMC algorithm.

        Returns:
            None: Quantities of interest are accessible via the instance attributes.
        """
        for i in range(1, self._iterations):
            theta_prev = self._thetas[:, i - 1]
            self._key, subkey = random.split(self._key)
            random_params = random.normal(key=subkey, shape=(self._num_params))
            cholesky_matrix = cholesky((2.38**2 / self._num_params) * self._cov)
            theta_prop = theta_prev + cholesky_matrix @ random_params

            proposal_likelihood = self._prior(theta_prop)

            if jnp.isfinite(proposal_likelihood):
                theta_dict = self.convert_theta_to_dict(theta_prop)

                # TODO: Implement filter logic
                # TODO: Filter needs to output likelihood, amongst other variables/arrays
                pf_likelihood, pf_hosp_estimates, pf_states, pf_betas = (
                    None,
                    None,
                    None,
                    None,
                )
                proposal_likelihood += jnp.sum(pf_likelihood)

                if proposal_likelihood > self._max_likelihood:
                    self.update_new_mle(
                        proposal_likelihood, pf_hosp_estimates, pf_states, pf_betas
                    )

                self.accept_reject(theta_prop, proposal_likelihood, i)

            else:
                # Reject automatically, because the ratio will be negative infinity.
                self.reject_proposal(i)

            self.update_cov(i)

    def convert_theta_to_dict(self, theta: ArrayLike) -> Dict[str, float]:
        """
        Converts a theta vector into a dictionary.
        The dictionary can be parsed by the PF.

        This avoids having to match up the exact indices of theta with
        particle filter input.

        Returns:
            Dictionary containing the parameter values.
        """
        new_theta_dict = {}
        for index, key in enumerate(self.theta_dictionary_template):
            new_theta_dict[key] = theta[index].item()
        return new_theta_dict

    def accept_reject(self, theta_prop: ArrayLike, new_likelihood: float, iteration: int) -> None:
        """
        Metropolis-Hastings algorithm to determine if the proposed theta
        is accepted or rejected.

        Args:
            theta_prop: the proposed theta vector
            new_likelihood: the likelihood of theta_prop
            iteration: the current MCMC iteration

        Returns:
            None: This method modifies instance attributes in place.
        """
        acceptance_probability = min(1, new_likelihood - self._likelihoods[iteration - 1])
        self._key, subkey = random.split(self._key)
        u = random.uniform(key=subkey, minval=0, maxval=1).item()
        if jnp.log(u) < acceptance_probability:
            self.accept_proposal(theta_prop, new_likelihood, iteration)
        else:
            self.reject_proposal(iteration)

    def accept_proposal(self, theta_prop: ArrayLike, new_likelihood: float, iteration: int) -> None:
        """
        Implements the logic for an accepted proposal.

        Args:
            theta_prop: the proposed theta vector
            new_likelihood: the likelihood of theta_prop
            iteration: the current MCMC iteration

        Returns:
            None: This method modifies instance attributes in place.
        """
        self._thetas[:, iteration] = theta_prop
        self._likelihoods[iteration] = new_likelihood
        self._accept_record[iteration] = 1

    def reject_proposal(self, i: int):
        """
        Implements the logic for a rejected proposal.
        Instance attributes are updated in place.
        The previous theta and likelihood are copied to the current iteration.
        """
        self._thetas[:, i] = self._thetas[:, i - 1]
        self._likelihoods[i] = self._likelihoods[i - 1]

    def update_cov(self, current_iter):
        pass

    def update_new_mle(
        self, new_likelihood, particle_estimates, particle_states, particle_betas
    ):
        """
        Updates the MLE based on the new likelihood and PF output.
        """
        self._max_likelihood = new_likelihood
        self._mle_hospitalizations = particle_estimates
        self._mle_states = particle_states
        self._mle_betas = particle_betas

    @property
    def mle_betas(self):
        """
        Get the beta values from the maximum likelihood run of the particle filter.
        """
        return self._mle_betas

    @property
    def mle_hospitalizations(self):
        """
        Get the hospitalization estimates from the maximum likelihood
        run of the particle filter.
        """
        return self._mle_hospitalizations

    @property
    def mle_states(self):
        """
        Get the state vectors at each time step from the maximum likelihood run of the particle filter.
        """
        return self._mle_states
