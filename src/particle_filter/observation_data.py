from dataclasses import dataclass

from jax._src.basearray import ArrayLike


@dataclass
class ObservationData:
    """Stores the observed/reported data (Hospitalization case counts)."""

    observations: ArrayLike

    def get_observation(self, t: int) -> int:
        """Returns the observation at time t.

        An observation is the new hospitalizations case count
        on day t.
        """
        return self.observations[t]
