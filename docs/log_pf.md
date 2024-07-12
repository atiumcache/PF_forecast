# Log-Domain Particle Fitler

```
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
```

- gamma: The recovery rate, representing the fraction of infected individuals who recover per unit time.                                         
- mu: The mortality rate, indicating the fraction of infected individuals who die per unit time.                                                 
- q: The hospitalization rate, representing the fraction of infected individuals who require hospitalization per unit time.                      
- eta: The rate at which hospitalized individuals recover and are discharged from the hospital.                                                  
- std: The standard deviation of the process noise, which accounts for the variability in the model's predictions.                               
- R: The basic reproduction number, representing the average number of secondary infections produced by a single infected individual in a fully  
   susceptible population.                                                                                                                        
- hosp: The initial number of hospitalized individuals at the start of the simulation.                                                           
- L: The latency period, representing the average time between exposure to the virus and the onset of infectiousness.                            
- D: The duration of the infectious period, indicating the average time an individual remains infectious.                                        

