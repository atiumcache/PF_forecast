from abc import ABC, abstractmethod
from typing import Dict, List

from utilities.Utils import Context, Particle


class Perturb(ABC):
    """Class encapsulating the perturbation function necessary to generate new particle proposals"""

    hyperparameters: Dict[str, float]  # A dictionary of perturbation parameters

    def __init__(self, hyper_params: Dict) -> None:
        """The perturber has special hyperparameters which tell randomly_perturb how much to move the parameters and state, implementation defined."""
        self.hyperparameters = hyper_params

    @abstractmethod
    def randomly_perturb(
        self, ctx: Context, particleArray: List[Particle]
    ) -> List[Particle]:
        """Implementations of this method will take a list of particles and perturb it according to a user defined distribution.

        Args:
            ctx: The Algorithm's Context object, in case metadata is necessary.

            particleArray: A list of particles, the self.particles list in Algorithm.

        Returns:
            Outputs the updated particle list. Note that as python lists are mutable and therefore passed by reference
            we could forego the return, however I've found that for consistentcy purposes it's better to ensure the
            self.particles list in the Algorithm is updated via assignment.

        """
        pass
