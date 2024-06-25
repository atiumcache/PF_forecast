#*[-----------------------------------------------------------------------------------------------]*#
#*[ Objective : This R program detects changepoints in beta_t and logit(beta_t) before a given    ]*#
#*[             time point under the AR(1) piecewise linear trends model using GA. Also, the      ]*#
#*[             program tests the model and generates next 28 days' beta_t values.                ]*#
#*[ Updated   : June 2, 2024                                                                      ]*#
#*[ Developers: Jaechoul Lee    
#*[ Modified by: Andrew Attilio                                                                   ]*#
#*[-----------------------------------------------------------------------------------------------]*#

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop("At least one argument must be supplied (output directory).", call. = FALSE)
}

working_dir <- args[1]
output_dir <- args[2]
location_code <- args[3]

# Setup directories
WD <- working_dir
WD.out <- output_dir

# Load required packages
.libPaths("/scratch/apa235/R_packages")

if(!require(dplyr)){
    install.packages("dplyr", lib="/scratch/apa235/R_packages", repos = "http://cran.us.r-project.org")
    library(dplyr)
}

if(!require(glue)){
    install.packages("glue", dependencies=TRUE, lib="/scratch/apa235/R_packages", repos = "http://cran.us.r-project.org")
    library(glue)
}

# Load the R code for the piecewise linear trends model and GA
source( file=paste(WD, "/r_scripts/lib_ga-PLT_v1-0.R",sep="") )

#*[-----------------------------------------------------------------------------------------------]*#
### Step 0-1: Read the Arizona daily new influenza hospitalizations data
#*[-----------------------------------------------------------------------------------------------]*#

# Read the AZ daily new influenza hospitalizations data
data_path_extension <- glue("/datasets/hosp_data/hosp_{location_code}_filtered.csv")
df.flu <- read.csv( file=paste(WD, data_path_extension ,sep=""),header=TRUE )
colnames( df.flu ) <- c("time_0","hosp")

df.flu$time_1 <- df.flu$time_0+1                          # [CAUTION] time starts at 1 instead of 0


#*[-----------------------------------------------------------------------------------------------]*#
### Step 1-1: Read the beta_t series
#*[-----------------------------------------------------------------------------------------------]*#

# Read the estimated beta series
beta_path_extension <- glue("/datasets/pf_results/{location_code}_average_beta.csv")
df.beta <- read.csv( file=paste(WD, beta_path_extension, sep=""), header=TRUE )
colnames( df.beta ) <- c("time_0","beta.t")

df.beta$time_1 <- df.beta$time_0+1                        # [CAUTION] time starts at 1 instead of 0


#*[-----------------------------------------------------------------------------------------------]*#
### Step 1-2: Determine a target beta_t during period t_bgn:t_end
#*[-----------------------------------------------------------------------------------------------]*#

# Set up a time period
t_bgn <- 1                                              # 1st day for analysis
t_end <- 60                                             # last day for analysis
n_fct <- 28                                             # number of days for forecasting

# Extract beta.t values for the selected time period and for 28 days after the time period
b.t     <- ts( df.beta$beta.t[ which(t_bgn   <= df.beta$time_1 & df.beta$time_1 <= t_end      ) ], start=t_bgn   )
b.t_act <- ts( df.beta$beta.t[ which(t_end+1 <= df.beta$time_1 & df.beta$time_1 <= t_end+n_fct) ], start=t_end+1 )


#*[-----------------------------------------------------------------------------------------------]*#
### Step 1-3: Compute logit(beta_t) during the period t_bgn:t_end
#*[-----------------------------------------------------------------------------------------------]*#

# Compute logit(beta.t) with maximum value of beta.t
b.t_max  <- 0.36                                         # [NOTE] Maximum statistic of beta_t is 0.26578
                                                        # mean(b.t)+6*sd(b.t)=0.3985; mean(b.t)+5*sd(b.t)=0.3604
lb.t     <- log( b.t/(b.t_max-b.t) )                    # logit with b.t_max as max
lb.t_act <- log( b.t_act/(b.t_max-b.t_act) )
t.t      <- time( lb.t )                                # times associated with lb.t


#*[-----------------------------------------------------------------------------------------------]*#
### Step 2-1: Find changepoints in logit(beta_t) for PLT model with AR(1) during period t_bgn:t_end
#*[-----------------------------------------------------------------------------------------------]*#

# Detect changepoints using GA
i <- 21                                                 # i is used for a seed number
ga.out <- ga.cpt_ts( y=lb.t,
             fitness=fit.PLTar_BIC,
             gen.size=200,max.itr=200,p.mut=0.05,       # gen.size=200,max.itr=125,p.mut=0.05,
             seed=10*(i-1)+543,
             is.graphic=FALSE,
             is.print=FALSE,
             is.export=FALSE
          )

ga.sol <- ga.out$solution                               # GA estimated changepoints
ga.bic <- ga.out$val.sol[length(ga.out$val.sol)]        # optimized value of penalized likelihood

ga.sol                                                  # [1] 4 28 33 40 45 with b.t_max=0.40 & max.itr=125
                                                        # [1] 4 29 33 39 45 with b.t_max=0.36 & max.itr=125
                                                        # [1] 4 29 33 40 44 with b.t_max=0.36 & max.itr=200
ga.bic                                                  # [1] -39.08868 with ga.sol: 4 29 33 39 45
                                                        # [1] -40.75605 with ga.sol: 4 29 33 40 44

ga.sol <- c(4, 29, 33, 40, 44)

fit.PLTar_BIC( y=lb.t, cp=ga.sol )                      # [1] -40.75605
fit.PLTar_BIC( y=lb.t, cp=c(4, 28, 33, 40, 45) )        # [1] -35.06563
fit.PLTar_BIC( y=lb.t, cp=c(4, 29, 33, 40, 46) )        # [1] -39.61175

#*[-----------------------------------------------------------------------------------------------]*#
### Step 2-2: Fit PLT model with AR(1) and GA changepoints to logit(beta_t)
#*[-----------------------------------------------------------------------------------------------]*#

# Fit a piecewise linear trends model with AR(1) errors and GA changepoints
ga.sol <- c(4, 29, 33, 40, 44)

fit.PLTar_GA <- fit.PLTar( y=lb.t, cp=ga.sol )
fit.PLTar_GA


# Display the fitted PLT model
t.t <- time( lb.t )                                     # times associated with logit(b.t)
lb.t_trd <- fit.PLTar_trends( y=lb.t, cp=ga.sol )       # piecewise trends line with changepoints



coef_fit.PLTar <- coef( fit.PLTar_GA )                  # PLT model parameter estimates
coef_fit.PLTar
#        ar1    intercept            t D.m.Series 1 D.m.Series 2 D.m.Series 3 D.m.Series 4 
# 0.45087177  -0.67105622   0.01296767   0.29550401  -0.44784479   0.26092521  -0.16551070

coef_trd <- cumsum( coef_fit.PLTar[-(1:2)] )            # trend estimates over each segment
coef_trd                                                # trend estimates for t_bgn:t_end
#          t D.m.Series 1 D.m.Series 2 D.m.Series 3 D.m.Series 4 
# 0.01296767   0.30847168  -0.13937311   0.12155209  -0.04395860

#*[-----------------------------------------------------------------------------------------------]*#
### Step 2-3: Model diagnostics
#*[-----------------------------------------------------------------------------------------------]*#

# Detrended beta_t series for residual autocorrelation
lb.t_detrd <- lb.t - lb.t_trd                           # detrended logit(beta.t) series
summary( lb.t_detrd )
#       Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -0.4419521 -0.0673531  0.0166310 -0.0002396  0.0833731  0.4230481

# Find the final residual series fromm PLT-AR(1) model fit
w.t <- resid( fit.PLTar_GA )
summary( w.t )
#      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# -0.309596 -0.070166  0.014012 -0.000465  0.062372  0.439193


#*[-----------------------------------------------------------------------------------------------]*#
### Step 2-4: Forecasting
#*[-----------------------------------------------------------------------------------------------]*#

# Forecast logit(beta.t) for next 28 days
lb.t_fct <- fct.PLTar( fit=fit.PLTar_GA, y=lb.t, cp=ga.sol, n.ahead=n_fct )

lb.t_fct.95l <- lb.t_fct$pred - 1.96*lb.t_fct$se
lb.t_fct.95u <- lb.t_fct$pred + 1.96*lb.t_fct$se


# Inverse logit transformation
b.t_fct.prd <- b.t_max*exp(lb.t_fct$pred)/(1+exp(lb.t_fct$pred))
b.t_trd     <- b.t_max*exp(lb.t_trd)/(1+exp(lb.t_trd))
b.t_fct.95l <- b.t_max*exp(lb.t_fct.95l)/(1+exp(lb.t_fct.95l))
b.t_fct.95u <- b.t_max*exp(lb.t_fct.95u)/(1+exp(lb.t_fct.95u))


#*[-----------------------------------------------------------------------------------------------]*#
### Step 2-5: Trajectories of beta.t based on logit(beta.t)
#*[-----------------------------------------------------------------------------------------------]*#

# Construct 1000 trajectories of logit(beta.t) for next 28 days
n_rpt <- 1000                                           # number of repetitions

set.seed(123)
out_trj.bootp <- matrix( 0,nrow=n_rpt,ncol=n_fct )      # bootstrap method

for (g in 1:n_rpt) {
  lb.t_trj <- trj.PLTar( fit=fit.PLTar_GA, y=lb.t, cp=ga.sol, n.ahead=n_fct, is.bootp=TRUE )
  out_trj.bootp[g,] <- b.t_max*exp(lb.t_trj)/(1+exp(lb.t_trj)) # inverse logit transformation
}
colnames(out_trj.bootp) <- c("d01","d02","d03","d04","d05","d06","d07","d08","d09","d10",
                             "d11","d12","d13","d14","d15","d16","d17","d18","d19","d20",
                             "d21","d22","d23","d24","d25","d26","d27","d28")

# Save the results
write.table( out_trj.bootp,file=paste(WD.out,"/out_logit-beta_trj_bootp.csv",sep=""),
             sep=",",quote=FALSE,row.names=FALSE,col.names=TRUE )

set.seed(123)
out_trj.rnorm <- matrix( 0,nrow=n_rpt,ncol=n_fct )      # random number generation method

for (g in 1:n_rpt) {
  lb.t_trj <- trj.PLTar( fit=fit.PLTar_GA, y=lb.t, cp=ga.sol, n.ahead=n_fct, is.bootp=FALSE )
  out_trj.rnorm[g,] <- b.t_max*exp(lb.t_trj)/(1+exp(lb.t_trj))  # inverse logit transformation
}
colnames(out_trj.rnorm) <- c("d01","d02","d03","d04","d05","d06","d07","d08","d09","d10",
                             "d11","d12","d13","d14","d15","d16","d17","d18","d19","d20",
                             "d21","d22","d23","d24","d25","d26","d27","d28")

# Save the results
write.table( out_trj.rnorm,file=paste(WD.out,"/out_logit-beta_trj_rnorm.csv",sep=""),
             sep=",",quote=FALSE,row.names=FALSE,col.names=TRUE )


