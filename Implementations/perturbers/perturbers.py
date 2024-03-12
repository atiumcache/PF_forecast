from Abstract.Perturb import Perturb
from typing import List,Dict
import numpy as np
from utilities.Utils import Context,Particle,ESTIMATION,jacob

'''Multivariate normal perturbations to all parameters and state variables after log transform'''
class MultivariatePerturbations(Perturb): 
    def __init__ (self,hyper_params:Dict)-> None: 
        '''The perturbation scheme defined in Calvetti et. al. Uses the log-normal perturbations in the linear-domain for both the static parameters and the 
        time-dependent parameters. '''
        super().__init__(hyper_params=hyper_params)


    def randomly_perturb(self,ctx:Context,particleArray: List[Particle])->List[Particle]: 
        """Implementations of this method will take a list of particles and perturb it according to a user defined distribution. 
        
        Args:
            ctx: The Algorithm's Context object, in case metadata is necessary. 

            particleArray: A list of particles, the self.particles list in Algorithm. 

        Returns: 
            Outputs the updated particle list. Note that as python lists are mutable and therefore passed by reference
            we could forego the return, however I've found that for consistentcy purposes it's better to ensure the 
            self.particles list in the Algorithm is updated via assignment.  

        """

        '''All this craziness is just used to extract the static and time-varying parameters from each particle and put them in lists, maybe not necessary.
        
        TODO
            I feel like this could be improved, no need to store all the parameter names at the very least, just extract them from the first particle as they will all 
            be the same. 
        
        '''

        '''_mat holds the values of both the time-dependent and static parameters. 
        
        TODO
            A numpy array would be faster here. Appending to lists is horrendously slow. We know the necessary size of the matrix any can just set up an matrix of zeroes, 
            and set it via indexed access. 
        
        '''
        static_param_mat = []
        static_names = []

        var_param_mat = []
        var_names = []

        for particle in particleArray: 
            static_params = []
            variable_params = []

            for _,(key,val) in enumerate(particle.param.items()):

                '''TODO I feel this is slow as well, intializing new lists each iteration is inefficient and the parameter search could potentially be made more 
                efficient. Maybe there's a better way to store parameters than a dictionary. Looping over the entire dictionary each time is problematic, we could 
                hold the information in a smarter data structure.'''

                if(key in ctx.estimated_params): 

                    if(ctx.estimated_params[key] == ESTIMATION.STATIC):
                        static_names.append(key)
                        static_params.append(val)

                    elif(ctx.estimated_params[key] == ESTIMATION.VARIABLE): 
                        var_names.append(key)
                        variable_params.append(val)

            static_param_mat.append(static_params)
            var_param_mat.append(variable_params)

        #names of the estimated parameters
        var_names = np.unique(np.array(var_names))
        static_names = np.unique(np.array(static_names))

        '''Needs to be a numpy array for proper indexing. Did the log operation up front to amortize the cost.'''
        static_param_mat = np.log(np.array(static_param_mat))
        var_param_mat = np.log(np.array(var_param_mat))

        '''Only bother with the static parameter estimation if there are static parameters to estimate.'''
        if(len(static_names) > 0): 


            '''Computes the log_mean as defined in Calvetti et.al. '''
            log_mean = 0
            for i,param_vec in enumerate(static_param_mat): 
                log_mean += ctx.prior_weights[i] * (param_vec)
            

            '''Computes the covariance of the logarithms of the particles'''
            cov = 0
            for i,param_vec in enumerate(static_param_mat): 
                cov += ctx.prior_weights[i] * np.outer(param_vec-log_mean,param_vec-log_mean)

            '''Holds the hyperparameter a, defined in terms of h'''
            a = np.sqrt(1-(self.hyperparameters["h"])**2)

            #if else for the multivariate normal bug 

            '''TODO I feel like we should be able to use the multivariate normal distribution for all of this, the if check is inefficient but ctx.rng.multivariate_normal 
            doesn't work in the single variate case. Also the numpy array for cov has an extra dimension if the dimension is 1, using squeeze as a stopgap. There's got to be a 
            better way to tackle this.'''
            if(len(static_param_mat[0]) == 1): 
                new_statics = ctx.rng.normal(a * static_param_mat[i] + (1-a)*log_mean,(self.hyperparameters["h"]**2)*np.squeeze(cov))

            else: 
                for i in range(len(particleArray)): 
                    new_statics = ctx.rng.multivariate_normal(a * static_param_mat[i] + (1-a)*log_mean,(self.hyperparameters["h"]**2)*cov)


            '''puts the perturbed static parameters back in the particle field'''
            for j,static in enumerate(new_statics): 
                particleArray[i].param[static_names[j]] = np.exp(static)


        '''Perturb the variable parameters

        TODO This setup is very dependent on the model being similar to the Calvetti model, the first population needing to be scaled as it corresponds to 
        S, and is dependent on the ctx.state_size parameter. Furthermore the variable names must be appended at the end of the state. There might be a nicer 
        way to do this. 

        '''


        '''Setting up the main diagonal of the matrix based on the construction used in Calvetti et. al.'''
        state1 = np.array([self.hyperparameters['sigma1']/ctx.population]) ** 2

        otherstates = np.array([self.hyperparameters['sigma1']**2 for _ in range(ctx.state_size-1)])

        param_variance = np.array([self.hyperparameters['sigma2']**2 for _ in range(len(var_names))])

        C = np.concatenate((state1,otherstates,param_variance))

        '''Generate diagonal matrix.'''

        C = np.diag(C).astype(float)


        '''Main perturbation loop'''
        for i in range(len(particleArray)): 
            log_state = np.log(particleArray[i].state)
            td_vec = np.concatenate((log_state,var_param_mat[i])) #Everything in here is log
            perturbed = np.exp(ctx.rng.multivariate_normal(td_vec,C))

            '''Normalization
            
            These steps ensure the population stays constant, we basically just normalize and multiply by the population. 

            TODO I wonder if this step causes significant bias in the perturbation. Might be worth evaluating. 
            
            '''
            perturbed[0:ctx.state_size] /= np.sum(perturbed[0:ctx.state_size])
            perturbed[0:ctx.state_size] *= ctx.population

            '''Returns the perturbed values to the particles. Dependent on the ctx.state_size value.'''
            particleArray[i].state = perturbed[:ctx.state_size]
            for j,name in enumerate(var_names): 
                particleArray[i].param[name] = perturbed[ctx.state_size+j:ctx.state_size+j+1][0]

        return particleArray

class LogMultivariatePerturbations(Perturb): 
    '''TODO Reference the comments in Multivariate Perturbations, much applies here with regards to optimization and possible bias.'''


    def __init__ (self,hyper_params:Dict)-> None: 
        '''The log variant of the perturbation scheme defined in Calvetti et. al. Uses the log-normal perturbations in the log-domain for both the static parameters and the 
        time-dependent parameters. '''
        super().__init__(hyper_params=hyper_params)

    def randomly_perturb(self,ctx:Context,particleArray: List[Particle])->List[Particle]: 
        """Implementations of this method will take a list of particles and perturb it according to a user defined distribution. 
        
        Args:
            ctx: The Algorithm's Context object, in case metadata is necessary. 

            particleArray: A list of particles, the self.particles list in Algorithm. 

        Returns: 
            Outputs the updated particle list. Note that as python lists are mutable and therefore passed by reference
            we could forego the return, however I've found that for consistentcy purposes it's better to ensure the 
            self.particles list in the Algorithm is updated via assignment.  

        """

        '''All this craziness is just used to extract the static and time-varying parameters from each particle and put them in lists, 
        this is the same scheme as in MultivariatePerturbations, it could definitely be improved.'''
        static_param_mat = []
        static_names = []

        var_param_mat = []
        var_names = []

        for particle in particleArray: 
            static_params = []
            variable_params = []

            for _,(key,val) in enumerate(particle.param.items()):

                if(key in ctx.estimated_params): 

                    if(ctx.estimated_params[key] == ESTIMATION.STATIC):
                        static_names.append(key)
                        static_params.append(val)

                    elif(ctx.estimated_params[key] == ESTIMATION.VARIABLE): 
                        var_names.append(key)
                        variable_params.append(val)

            static_param_mat.append(static_params)
            var_param_mat.append(variable_params)

        #names of the estimated parameters
        var_names = np.unique(np.array(var_names))
        static_names = np.unique(np.array(static_names))

        #matrix of variable and static parameters from each particle
        static_param_mat = np.log(np.array(static_param_mat))
        '''static_param_mat is of shape (particle_count,len(static_names))'''

        var_param_mat = np.log(np.array(var_param_mat))
        '''var_param_mat is of shape (particle_count,len(var_names))'''

        if(len(static_names) > 0): 

            
            a = np.sqrt(1-(self.hyperparameters["h"])**2) #setup the hyperparameter a as defined in Calvetti et. al.


            deltas = np.zeros((ctx.particle_count,len(static_names)))
            '''set up an array of zeroes of shape (particle_count,static_names)'''

            '''Now compute the deltas, we must log each parameter '''
            for i in range(len(static_names)): 
                sub_theta = static_param_mat[:,i]
                '''sub_theta shape is (particle_count,1)'''

                for j in range(ctx.particle_count):
                    log_theta = sub_theta[j]
                    log_log = np.log(np.abs(log_theta))
                    '''Absolute value here accounts for taking log of a negative number, 
                    log_theta is surely negative. We take the complex part outside the computation. '''

                    '''Make sure to add the weights to the deltas. Must be the log weights for this to work properly. '''
                    deltas[j,i] = ctx.prior_weights[j] + log_log

            ξ = []
            for i in range(len(static_names)): 
                '''Compute ξ elementwise and multiply by -1 to account for the sign change.'''
                ξ.append(-1 * np.exp(jacob(deltas[:,i])))

            '''Need to pull the last element, remember jacob returns the array of partial sums. '''
            ξ = np.array(ξ)[:,-1]


            '''Now for Σ, I called these elements psis to match up with the notation I used in the paper. Vectorization at work here, psis shape is 
            (particle_count,len(static_names))
            '''
            psis = np.array(static_param_mat - ξ)

            matrix_set = []
            for i in range(ctx.particle_count): 
                matrix_set.append(np.outer(psis[i,:] - ξ,psis[i,:]-ξ))
            matrix_set = np.array(matrix_set)

            '''Set up the matrix set, you could definitely do this all with numpy arrays but it would be slightly complicated, would give a small speed boost. 
            TODO Set this up as a vectorized operation. 
            
            '''

            Σ = np.zeros((len(static_names),len(static_names)))

            '''Algorithm to find the minimum value for C
            
            find the absolute value of the minimum element in the off diagonal and add e + epsilon. Here epsilon is 1
            
            '''
            #C = np.abs(np.min(matrix_set[:,0,1])) + np.exp(1) + 1
            C = 100

            for i in range(len(static_names)):
                for j in range(len(static_names)):
                    if(i != j):

                        '''On the off diagonals we compute the elements using the mean shifting approach. Thus we need two sets of deltas, 
                        one for the mean shifted quantity and one for the constant to subtract to invert the map.'''

                        deltas_Y = np.log(matrix_set[:,i,j] + C * np.ones_like(matrix_set[:,i,j])) + ctx.prior_weights

                        deltas_Z = ctx.prior_weights + np.log(C * np.ones_like(ctx.prior_weights))

                        Z = np.exp(jacob(deltas_Z)[-1])

                        Σ[i,j] = np.exp(jacob(deltas_Y)[-1]) - Z

                    else: 
                        '''On the diagonal the elements are guaranteed to be positive, so there isn't a '''

                        deltas = np.log(matrix_set[:,i,j]) + ctx.prior_weights
                        Σ[i,j] = np.exp(jacob(deltas)[-1])

            if(len(static_param_mat[0]) == 1): 
                new_statics = ctx.rng.normal(a * static_param_mat[i] + (1-a)*ξ,(self.hyperparameters["h"]**2)*np.squeeze(Σ))

            for i in range(len(particleArray)): 
                new_statics = ctx.rng.multivariate_normal(a * static_param_mat[i] + (1-a)*ξ,(self.hyperparameters["h"]**2)*Σ)


            '''puts the perturbed static parameters back in the particle field'''
            for j,static in enumerate(new_statics): 
                particleArray[i].param[static_names[j]] = np.exp(static)

        
        


        '''Perturb the variable parameters '''


        state1 = np.array([self.hyperparameters['sigma1']/ctx.population]) ** 2
        otherstates = np.array([self.hyperparameters['sigma1']**2 for _ in range(ctx.state_size-1)])
        param_variance = np.array([self.hyperparameters['sigma2']**2 for _ in range(len(var_names))])

        C = np.concatenate((state1,otherstates,param_variance))

        C = np.diag(C).astype(float)


        '''Main perturbation loop'''
        for i in range(len(particleArray)): 


            log_state = np.log(particleArray[i].state)
            if(np.any(np.isnan(log_state))): 
                print(particleArray[i])
                print(np.sum(particleArray[i].state))


            td_vec = (np.concatenate((log_state,var_param_mat[i])))
            perturbed = np.exp(ctx.rng.multivariate_normal(td_vec,C))

            '''Normalization'''
            perturbed[0:ctx.state_size] /= np.sum(perturbed[0:ctx.state_size])
            perturbed[0:ctx.state_size] *= ctx.population


            particleArray[i].state = perturbed[:ctx.state_size]
            for j,name in enumerate(var_names): 
                particleArray[i].param[name] = perturbed[ctx.state_size+j:ctx.state_size+j+1][0]

        return particleArray
    
class DynamicPerturbations(Perturb):

    def __init__ (self,hyper_params:Dict)-> None: 
        '''The perturber has special hyperparameters which tell randomly_perturb how much to move the parameters and state'''
        super().__init__(hyper_params=hyper_params)


    def randomly_perturb(self,ctx:Context,particleArray: List[Particle])->List[Particle]: 
        '''Implementations of this method will take a list of particles and perturb it according to a user defined distribution'''

        '''All this craziness is just used to extract the static and time-varying parameters from each particle and put them in lists'''
        static_param_mat = []
        static_names = []

        var_param_mat = []
        var_names = []

        for particle in particleArray: 
            static_params = []
            variable_params = []

            for _,(key,val) in enumerate(particle.param.items()):

                if(key in ctx.estimated_params): 

                    if(ctx.estimated_params[key] == ESTIMATION.STATIC):
                        static_names.append(key)
                        static_params.append(val)

                    elif(ctx.estimated_params[key] == ESTIMATION.VARIABLE): 
                        var_names.append(key)
                        variable_params.append(val)

            static_param_mat.append(static_params)
            var_param_mat.append(variable_params)

        #names of the estimated parameters
        var_names = np.unique(np.array(var_names))
        static_names = np.unique(np.array(static_names))

        #matrix of variable and static parameters from each particle
        static_param_mat = np.log(np.array(static_param_mat))
        var_param_mat = np.log(np.array(var_param_mat))


        if(len(static_names) > 0): 
            '''Computes the log_mean as defined in Calvetti et.al. '''
            log_mean = 0
            for i,param_vec in enumerate(static_param_mat): 
                log_mean += ctx.weights[i] * (param_vec)

            '''Computes the covariance of the logarithms of the particles'''
            cov = 0
            for i,param_vec in enumerate(static_param_mat): 
                cov += ctx.weights[i] * np.outer(param_vec-log_mean,param_vec-log_mean)

            '''Holds the hyperparameter a, defined in terms of h'''
            a = np.sqrt(1-(self.hyperparameters["h"])**2)

            #if else for the multivariate normal bug 

            if(len(static_param_mat[0]) == 1): 
                new_statics = ctx.rng.normal(a * static_param_mat[i] + (1-a)*log_mean,(self.hyperparameters["h"]**2)*cov)

            else: 
                for i in range(len(particleArray)): 
                    new_statics = ctx.rng.multivariate_normal(a * static_param_mat[i] + (1-a)*log_mean,(self.hyperparameters["h"]**2)*cov)


            '''puts the perturbed static parameters back in the particle field'''
            for j,static in enumerate(new_statics): 
                particleArray[i].param[static_names[j]] = np.exp(static)


        '''Perturb the variable parameters '''

        
        avg_state = np.mean([(particle.state) for particle in particleArray],axis=0)


        '''Computes the covariance of the particle states'''
        cov = 0
        for i,particle in enumerate(particleArray): 
            cov += ctx.weights[i] * np.outer((particle.state)-avg_state,(particle.state)-avg_state)

        print(cov)
 

        '''state perturbation loop'''
        for i in range(len(particleArray)): 
            state = (particleArray[i].state)
            perturbed = np.abs(ctx.rng.multivariate_normal(state,cov))

            '''Normalization'''
            perturbed[0:ctx.state_size] /= np.sum(perturbed[0:ctx.state_size])
            perturbed[0:ctx.state_size] *= ctx.population

            particleArray[i].state = perturbed[:ctx.state_size]

            for j,name in enumerate(var_names): 
                particleArray[i].param[name] = np.exp(ctx.rng.normal(np.log(particleArray[i].param[name]),self.hyperparameters['sigma2'] ** 2))




        return particleArray


        
   