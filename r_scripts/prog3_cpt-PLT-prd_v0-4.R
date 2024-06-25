# Original R Script, before I (Andrew) made changes and
# renamed to `beta_trend_forecast.R`



#*[-----------------------------------------------------------------------------------------------]*#
#*[ Objective : This R program detects changepoints in beta_t and logit(beta_t) before a given    ]*#
#*[             time point under the AR(1) piecewise linear trends model using GA. Also, the       ]*#
#*[             program tests the model and generates next 28 days' beta_t values.                ]*#
#*[ Updated   : Mar 25, 2024                                                                      ]*#
#*[ Developers: Jaechoul Lee                                                                      ]*#
#*[-----------------------------------------------------------------------------------------------]*#

# Setup directories
WD <- getwd()
WD.out <- paste( WD, "/Documents/!1Research/Paper/Epidemiology/P05_FluSIR/Analysis/", sep="" ) # on Mac
WD.out <- paste( WD, "/!1Research/Paper/Epidemiology/P05_FluSIR/Analysis/", sep="" )           # on Windows
WD.out <- c("G:\\My Drive\\papers\\POMP\\change point\\2024_3_26\\")
# Load required packages
library( dplyr )
library( ggplot2 )

# Load the R code for the piecewise linear trends model and GA
source( file=paste(WD.out,"lib_ga-PLT_v1-0.R",sep="") )

#*[-----------------------------------------------------------------------------------------------]*#
### Step 0-1: Read the Arizona daily new influenza hospitalizations data
#*[-----------------------------------------------------------------------------------------------]*#

# Read the AZ daily new influenza hospitalizations data
df.flu <- read.csv( file="G:\\My Drive\\papers\\POMP\\change point\\AZ_FLU_HOSPITALIZATIONS.csv",header=TRUE )
colnames( df.flu ) <- c("time_0","hosp")

df.flu$time_1 <- df.flu$time_0+1                          # [CAUTION] time starts at 1 instead of 0
dim( df.flu )
# [1] 223   3

# Data visualization
dev.new( width=10, height=4 )
ggplot( df.flu, aes(x=time_0,y=hosp) ) +
  geom_line( color="#00AFBB", linewidth=0.75 ) +
  ylab( "Hospitalizations" ) +
  xlab( "Time" ) +
  theme_bw()

#*[-----------------------------------------------------------------------------------------------]*#
### Step 1-1: Read the beta_t series
#*[-----------------------------------------------------------------------------------------------]*#

# Read the estimated beta series
df.beta <- read.csv( file=paste("G:\\My Drive\\papers\\POMP\\change point\\average_beta.csv",sep=""),header=TRUE )
colnames( df.beta ) <- c("time_0","beta.t")

df.beta$time_1 <- df.beta$time_0+1                        # [CAUTION] time starts at 1 instead of 0
dim( df.beta )
# [1] 120   3

summary( df.beta$beta.t )
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.03161 0.07473 0.12346 0.12185 0.16631 0.26578

# Data visualization
dev.new( width=10, height=4 )
ggplot( df.beta, aes(x=time_0,y=beta.t) ) +
  geom_line( color="#00AFBB",linewidth=0.75 ) +
  ylab( "beta_t" ) +
  xlab( "Time" ) +
  theme_bw()

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

summary(b.t)
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.1105  0.1323  0.1673  0.1698  0.1999  0.2658 
sd(b.t)
# [1] 0.03811991

# Sample ACF and PACF for autocorrelation
dev.new( width=8, height=4 )
par( mfrow=c(2,2),mex=0.75 )
acf( b.t,lag.max=28,ylim=c(-0.5,1) )
pacf( b.t,lag.max=28,ylim=c(-0.5,1) )
acf( diff(b.t),lag.max=28,ylim=c(-0.5,1) )
pacf( diff(b.t),lag.max=28,ylim=c(-0.5,1) )

#*[-----------------------------------------------------------------------------------------------]*#
### Step 1-3: Compute logit(beta_t) during the period t_bgn:t_end
#*[-----------------------------------------------------------------------------------------------]*#

# Compute logit(beta.t) with maximum value of beta.t
b.t_max  <- 0.36                                         # [NOTE] Maximum statistic of beta_t is 0.26578
                                                        # mean(b.t)+6*sd(b.t)=0.3985; mean(b.t)+5*sd(b.t)=0.3604
lb.t     <- log( b.t/(b.t_max-b.t) )                    # logit with b.t_max as max
lb.t_act <- log( b.t_act/(b.t_max-b.t_act) )
t.t      <- time( lb.t )                                # times associated with lb.t

# Timeplot of logit(beta.t)
dev.new( width=8, height=4 )
ggplot() +
  geom_line( aes(x=t.t,y=lb.t), color="#00AFBB", linewidth=0.75 ) +
  ylab( "log(beta_t/(beta_t_max-beta_t))" ) +
  xlab( "Time" ) +
  theme_bw()

# Sample ACF and PACF for autocorrelation
dev.new( width=8,height=4 )
par( mfrow=c(2,2),mex=0.75 )
acf( lb.t,lag.max=28,ylim=c(-0.5,1) )
pacf( lb.t,lag.max=28,ylim=c(-0.5,1) )
acf( diff(lb.t),lag.max=28,ylim=c(-0.5,1) )
pacf( diff(lb.t),lag.max=28,ylim=c(-0.5,1) )

#*[-----------------------------------------------------------------------------------------------]*#
### Step 2-1: Find changepoints in logit(beta_t) for PLT model with AR(1) during period t_bgn:t_end
#*[-----------------------------------------------------------------------------------------------]*#

# Detect changepoints using GA
i <- 21                                                 # i is used for a seed number
ga.out <- ga.cpt_ts( y=lb.t,
             fitness=fit.PLTar_BIC,
             gen.size=200,max.itr=200,p.mut=0.05,       # gen.size=200,max.itr=125,p.mut=0.05,
             seed=10*(i-1)+543,
             is.graphic=TRUE,
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

#arima(x = y, order = c(1, 0, 0), xreg = cbind(t, D.m), include.mean = TRUE, 
#    method = "CSS-ML")
#
#Coefficients:
#         ar1  intercept       t  D.m.Series 1  D.m.Series 2  D.m.Series 3  D.m.Series 4
#      0.4509    -0.6711  0.0130        0.2955       -0.4478        0.2609       -0.1655
# s.e.  0.1149     0.0846  0.0048        0.0390        0.0571        0.0587        0.0451
#
# sigma^2 estimated as 0.01713:  log likelihood = 36.76,  aic = -57.51

# Display the fitted PLT model
t.t <- time( lb.t )                                     # times associated with logit(b.t)
lb.t_trd <- fit.PLTar_trends( y=lb.t, cp=ga.sol )       # piecewise trends line with changepoints

dev.new( width=8,height=4 )
ggplot() +
  geom_line( aes(x=t.t,y=lb.t),color="#00AFBB",linewidth=0.75 ) +
  geom_line( aes(x=t.t,y=lb.t_trd),color="blue",linewidth=0.75 ) +
  geom_vline( xintercept=ga.sol[-1],lwd=0.5,lty=2,colour="tomato" ) +
  ylab( "logit(beta_t)" ) +
  xlab( "Time" ) +
  theme_bw()

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

# Sample ACF and PACF of detrended logit(beta.t) series. This shows an AR(1) or MA(1) autocorrelation.
dev.new()
par( mfrow=c(2,1),mex=0.75 )
acf( lb.t_detrd,lag.max=20,ylim=c(-0.5,1) )
pacf( lb.t_detrd,lag.max=20,ylim=c(-0.5,1) )

# Find the final residual series fromm PLT-AR(1) model fit
w.t <- resid( fit.PLTar_GA )
summary( w.t )
#      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# -0.309596 -0.070166  0.014012 -0.000465  0.062372  0.439193

dev.new( width=8,height=4 )
par( mfrow=c(1,1),mex=0.75 )
plot.ts( w.t,type="o",ylim=c(-0.50,0.50),xlab="Time",main="Residual series" )

dev.new()
par( mfrow=c(2,1),mex=0.75 )
acf( w.t,lag.max=20,ylim=c(-0.5,1) )
pacf( w.t,lag.max=20,ylim=c(-0.5,1) )

# Check normality for the final residual series
dev.new( width=10,height=6 )
par( mfrow=c(1,2),mex=0.75 )
hist( w.t,freq=FALSE,                                         # histogram of final residual series
      breaks=seq(-0.5,0.5,0.05),
      col="grey85",ylim=c(0,6),
      main="Residual Histogram")
u <- seq(-0.5,0.5,length=500)
lines( u,dnorm(u,mean=mean(w.t),sd=sd(w.t)),lty=1,col="red" ) # add theoretical normal density
qqnorm( w.t )                                                 # normal Q-Q plot
qqline( w.t,col="red" )                                       # add a reference line

#*[-----------------------------------------------------------------------------------------------]*#
### Step 2-4: Forecasting
#*[-----------------------------------------------------------------------------------------------]*#

# Forecast logit(beta.t) for next 28 days
lb.t_fct <- fct.PLTar( fit=fit.PLTar_GA, y=lb.t, cp=ga.sol, n.ahead=n_fct )

lb.t_fct.95l <- lb.t_fct$pred - 1.96*lb.t_fct$se
lb.t_fct.95u <- lb.t_fct$pred + 1.96*lb.t_fct$se

dev.new( width=8,height=4 )
par( mfrow=c(1,1),mex=0.75 )
ts.plot( lb.t, lb.t_fct$pred, type="o", col=1:2, ylim=c(-2.0,1.5), main="logit(beta_t) with 28 days predicted")
lines( lb.t_trd, col="orange" )
lines( lb.t_act, col="gray50" )
lines( lb.t_fct.95l,col="blue",lty="dashed" )
lines( lb.t_fct.95u,col="blue",lty="dashed" )
abline( v=ga.sol[-1], col="green", lty=2 )
legend( "topright", lty=c("solid","solid","dashed"),
        legend=c("Actual","Predicted","Upper & Lower 95%"),col=c("black","red","blue") )

# Inverse logit transformation
b.t_fct.prd <- b.t_max*exp(lb.t_fct$pred)/(1+exp(lb.t_fct$pred))
b.t_trd     <- b.t_max*exp(lb.t_trd)/(1+exp(lb.t_trd))
b.t_fct.95l <- b.t_max*exp(lb.t_fct.95l)/(1+exp(lb.t_fct.95l))
b.t_fct.95u <- b.t_max*exp(lb.t_fct.95u)/(1+exp(lb.t_fct.95u))

dev.new( width=8,height=4 )
par( mfrow=c(1,1),mex=0.75 )
ts.plot( b.t, b.t_fct.prd, type="o", col=1:2, ylim=c(0,0.3), main="beta_t with 28 days predicted")
lines( b.t_trd, col="orange" )
lines( b.t_act, col="gray50" )
lines( b.t_fct.95l,col="blue",lty="dashed" )
lines( b.t_fct.95u,col="blue",lty="dashed" )
abline( v=ga.sol[-1], col="green", lty=2 )
legend( "topright", lty=c("solid","solid","dashed"),
        legend=c("Actual","Predicted","Upper & Lower 95% CI"),col=c("black","red","blue") )

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
write.table( out_trj.bootp,file=paste(WD.out,"Out_prog3/out_logit-beta_trj_bootp.csv",sep=""),
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
write.table( out_trj.rnorm,file=paste(WD.out,"Out_prog3/out_logit-beta_trj_rnorm.csv",sep=""),
             sep=",",quote=FALSE,row.names=FALSE,col.names=TRUE )

# Display 50 trajectories of beta.t for next 28 days
df.beta_trj.bootp <- read.csv( file=paste(WD.out,"Out_prog3/out_logit-beta_trj_bootp.csv",sep=""),header=TRUE )
df.beta_trj.rnorm <- read.csv( file=paste(WD.out,"Out_prog3/out_logit-beta_trj_rnorm.csv",sep=""),header=TRUE )

dev.new( width=10,height=5 )
par( mfrow=c(1,1),mex=0.75 )
ts.plot( b.t, b.t_fct.prd, type="o", col=1:2, ylim=c(0,0.3), main="beta.t with 28 days predicted (bootstrap)")
lines( b.t_trd,col="orange" )
lines( b.t_act,col="gray50" )
for (g in 1:50) {
  b.t_trj <- ts( as.numeric(df.beta_trj.bootp[g,]), start=t_end+1 )
  lines( b.t_trj,col="blue",lty="dashed" )
}
abline( v=ga.sol[-1], col="green", lty=2 )
legend( "topright", lty=c("solid","solid","solid"),
        legend=c("Actual","Predicted","Simulated"),col=c("black","red","blue") )

dev.new( width=10,height=5 )
par( mfrow=c(1,1),mex=0.75 )
ts.plot( b.t, b.t_fct.prd, type="o", col=1:2, ylim=c(0,0.3), main="beta.t with 28 days predicted (rnorm)")
lines( b.t_trd,col="orange" )
lines( b.t_act,col="gray50" )
for (g in 1:50) {
  b.t_trj <- ts( as.numeric(df.beta_trj.rnorm[g,]), start=t_end+1 )
  lines( b.t_trj,col="blue",lty="dashed" )
}
abline( v=ga.sol[-1], col="green", lty=2 )
legend( "topright", lty=c("solid","solid","solid"),
        legend=c("Actual","Predicted","Simulated"),col=c("black","red","blue") )

# Figure of 10 trajectories
dev.new( width=10,height=5 )
par( mfrow=c(1,1),mex=0.75, cex=1.2 )
ts.plot( b.t, b.t_fct.prd, col=1:2, ylim=c(0,0.3) )
lines( b.t_trd,col="red" )
lines( b.t_act,col="gray50" )
for (g in 1:10) {
  b.t_trj <- ts( as.numeric(df.beta_trj.rnorm[g,]), start=t_end+1 )
  lines( b.t_trj,col="blue",lty="dashed" )
}
abline( v=ga.sol[-1], col="purple", lty=2 )
legend( "topright", lty=c("solid","solid","solid"),
        legend=c("Actual","Predicted","Simulated"),col=c("black","red","blue") )








#*[-----------------------------------------------------------------------------------------------]*#
### Step 3-1: Find changepoints in beta_t for PCS model with AR(1) during period t_bgn:t_end
#*[-----------------------------------------------------------------------------------------------]*#

# Detect changepoints using GA
i <- 21                                                 # i is used for a seed number
ga.out <- ga.cpt_ts( y=lb.t,
             fitness=fit.PCSar_BIC,
             gen.size=200,max.itr=200,p.mut=0.05,       # gen.size=200,max.itr=200,p.mut=0.05,
             seed=10*(i-1)+543,
             is.graphic=TRUE,
             is.print=FALSE,
             is.export=FALSE
          )

ga.sol <- ga.out$solution                               # GA estimated changepoints
ga.bic <- ga.out$val.sol[length(ga.out$val.sol)]        # optimized value of penalized likelihood

ga.sol                                                  # no changepoints detected
ga.bic

fit.PCSar( y=lb.t, cp=ga.sol )
fit.PCSar_BIC( y=lb.t, cp=ga.sol )

#*[-----------------------------------------------------------------------------------------------]*#
### Step 3-2: Fit PCS model with AR(1) and GA changepoints to logit(beta_t)
#*[-----------------------------------------------------------------------------------------------]*#

# Fit a piecewise linear trends model with AR(1) errors and GA changepoints
ga.sol <- c(4, 29, 33, 40, 44)                          # a trial with GA from piecewise linear trends

fit.PCSar_GA <- fit.PCSar( y=lb.t, cp=ga.sol )
fit.PCSar_GA

# Display the fitted PLT model
t.t <- time( lb.t )                                     # times associated with logit(b.t)
lb.t_trd <- fit.PCSar_trends( y=lb.t, cp=ga.sol )       # piecewise cubic trends with changepoints

dev.new( width=8,height=4 )
ggplot() +
  geom_line( aes(x=t.t,y=lb.t),color="#00AFBB",linewidth=0.75 ) +
  geom_line( aes(x=t.t,y=lb.t_trd),color="blue",linewidth=0.75 ) +
  geom_vline( xintercept=ga.sol[-1],lwd=0.5,lty=2,colour="tomato" ) +
  ylab( "logit(beta_t)" ) +
  xlab( "Time" ) +
  theme_bw()


