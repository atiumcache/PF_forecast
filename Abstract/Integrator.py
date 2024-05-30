from abc import ABC, abstractmethod
from typing import List, Tuple

from numpy import int_

from utilities.Utils import Context, Particle


class Integrator(ABC):
    """Class encapulating the function necessary to propagate the system of interest from one step to the next. Thin wrapper around whatever integrator we are using."""

    @abstractmethod
    def propagate(self, particleArray: List[Particle], ctx: Context) -> List[Particle]:
        """Propagates the state forward one step and returns an array of states and observations across the the integration period

        Args:
            particleArray: A list of particles, this will be self.particles from Algorithm.
            ctx: The Algorithm's context object is passed as well, in case algorithm metadata is needed.

        Returns:
            Outputs the updated particle list. Note that as python lists are mutable and therefore passed by reference
            we could forego the return, however I've found that for consistentcy purposes it's better to ensure the
            self.particles list in the Algorithm is updated via assignment.

        """
        pass

    """Note: observations are forced to be integers, if using a deterministic integrator round off values before returning them from propagate
    python may do this for you via downcasting, but it's very up in the air, best to ensure data is correct yourself"""
