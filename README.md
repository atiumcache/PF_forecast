# SMC_EPI
Particle filter code for research in static and time dependent parameter estimation. Derived from:   

C. Gentner, S. Zhang, and T. Jost, “Log-PF: particle filtering in logarithm domain,” Journal of Electrical and Computer Engineering, vol. 2018, Article ID 5763461, 11 pages, 2018.

D. Calvetti, A. Hoover, J. Rose, and E. Somersalo. Bayesian particle filter algorithm for learning epidemic dynamics. Inverse Problems, 2021

The code is structured in a class hierarchy, abstract base classes(ABC) are used to template implementations of the basic algorithm. 

Abstract/Algorithm: 
This ABC is the template for the entrypoint of the algorithm, it has 5 fields, 3 of which all also ABC's templating subfunctionality required for running the algorithm. 


