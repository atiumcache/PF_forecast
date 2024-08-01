from dataclasses import dataclass, field


@dataclass
class ModelParameters:
    """
    SIRH model parameters, for the RHS function.
    """

    gamma: float = field(default_factory=lambda: 0.06)
    mu: float = field(default_factory=lambda: 0.004)
    q: float = field(default_factory=lambda: 0.1)
    eta: float = field(default_factory=lambda: 0.1)
    std: float = field(default_factory=lambda: 10.0)
    R: float = field(default_factory=lambda: 50.0)
    hosp: int = field(default_factory=lambda: 10)
    L: int = field(default_factory=lambda: 90)
    D: int = field(default_factory=lambda: 10)
