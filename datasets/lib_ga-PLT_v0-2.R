### Compute the MDL penalty
penalty.MDL <- function(y,cp) {                    # y   : data
                                                   # cp  : changepoint chromosome (m; tau_1,...,tau_m)
  n <- length(y)                                   # n   : sample size
  m <- cp[1]                                       # m   : number of changepoints

  if (m == 0) {
    pnt <- 0
  } else {
    tau.ext <- c(cp[-1],n+1)                       # tau.ext: changepoints in days (tau_1,...,tau_m,N+1)
    n.r <- numeric(length=m)                       # n.r : no. of observations in each regime
    for (i in 1:m) {
      n.r[i] <- sum(!is.na(y[tau.ext[i]:(tau.ext[i+1]-1)]))
    }
    pnt <- log(m+1)+0.5*sum(log(n.r))+sum(log(tau.ext[-1]))
  }

  return(pnt)                                      # smaller is better
}

### Compute MDL of a PLT model fit under independence and changepoints
fit.PLT_MDL <- function( y, cp ) {                 # y    : data
                                                   # cp   : changepoint chromosome (m; tau_1,...,tau_m)
  n <- length( y )                                 # n    : sample size
  m <- cp[1]                                       # m    : number of changepoints
  tau <- cp[-1]                                    # tau  : time locations of changepoints

  t <- time( y )

  if ( m==0 ) {
    lm.PLT <- lm( y ~ t )
  } else {
    D.m <- matrix( 0, nrow=n, ncol=m )
    for ( k in 1:m ) {
      D.m[,k] <- pmax( t-tau[k], 0 )
    }
    lm.PLT <- lm( y ~ t + D.m )
  }

  pnllik.MDL <- -as.numeric(logLik( lm.PLT )) + penalty.MDL(y=y,cp=cp)  # fitness: -log(L)+penalty
  return( pnllik.MDL ) 
}

### Compute MDL of a PLT model fit under AR(1) and changepoints
fit.PLTar_MDL <- function( y, cp ) {               # y    : data
                                                   # cp   : changepoint chromosome (m; tau_1,...,tau_m)
  n <- length( y )                                 # n    : sample size
  m <- cp[1]                                       # m    : number of changepoints
  tau <- cp[-1]                                    # tau  : time locations of changepoints

  t <- time( y )

  if ( m==0 ) {
    ar.PLT <- arima( y,order=c(1,0,0),xreg=t,include.mean=TRUE,method="CSS-ML" )
  } else {
    D.m <- matrix( 0, nrow=n, ncol=m )
    for ( k in 1:m ) {
      D.m[,k] <- pmax( t-tau[k], 0 )
    }
    ar.PLT <- arima( y,order=c(1,0,0),xreg=cbind(t,D.m),include.mean=TRUE,method="CSS-ML" )
  }

  pnllik.MDL <- -2*as.numeric(logLik( ar.PLT )) + penalty.MDL(y=y,cp=cp)  # fitness: -2*log(L)+penalty
  return(pnllik.MDL) 
}

### Compute BIC of a PLT model fit under independence and changepoints
fit.PLT_BIC <- function( y, cp ) {                 # y    : data
                                                   # cp   : changepoint chromosome (m; tau_1,...,tau_m)
  n <- length( y )                                 # n    : sample size
  m <- cp[1]                                       # m    : number of changepoints
  tau <- cp[-1]                                    # tau  : time locations of changepoints

  t <- time( y )

  if ( m==0 ) {
    lm.PLT <- lm( y ~ t )
  } else {
    D.m <- matrix( 0, nrow=n, ncol=m )
    for ( k in 1:m ) {
      D.m[,k] <- pmax( t-tau[k], 0 )
    }
    lm.PLT <- lm( y ~ t + D.m )
  }

  return( BIC(lm.PLT) )
}

### Compute BIC of a PLT model fit under AR(1) and changepoints
fit.PLTar_BIC <- function( y, cp ) {               # y    : data
                                                   # cp   : changepoint chromosome (m; tau_1,...,tau_m)
  n <- length( y )                                 # n    : sample size
  m <- cp[1]                                       # m    : number of changepoints
  tau <- cp[-1]                                    # tau  : time locations of changepoints

  t <- time( y )

  if ( m==0 ) {
    ar.PLT <- arima( y,order=c(1,0,0),xreg=t,include.mean=TRUE,method="CSS-ML" )
  } else {
    D.m <- matrix( 0, nrow=n, ncol=m )
    for ( k in 1:m ) {
      D.m[,k] <- pmax( t-tau[k], 0 )
    }
    ar.PLT <- arima( y,order=c(1,0,0),xreg=cbind(t,D.m),include.mean=TRUE,method="CSS-ML" )
  }

  return( BIC(ar.PLT) )
}

### Fit a PLT model under AR(1) and changepoints
fit.PLTar <- function( y, cp ) {                   # y    : data
                                                   # cp   : changepoint chromosome (m; tau_1,...,tau_m)
  n <- length( y )                                 # n    : sample size
  m <- cp[1]                                       # m    : number of changepoints
  tau <- cp[-1]                                    # tau  : time locations of changepoints

  t <- time( y )

  if ( m==0 ) {
    ar.PLT <- arima( y,order=c(1,0,0),xreg=t,include.mean=TRUE,method="CSS-ML" )
  } else {
    D.m <- matrix( 0, nrow=n, ncol=m )
    for ( k in 1:m ) {
      D.m[,k] <- pmax( t-tau[k], 0 )
    }
    ar.PLT <- arima( y,order=c(1,0,0),xreg=cbind(t,D.m),include.mean=TRUE,method="CSS-ML" )
  }

  return( ar.PLT )
}

### Find the estimated trends in a PLT model under AR(1) and changepoints
fit.PLTar_trends <- function( y, cp ) {            # y    : data
                                                   # cp   : changepoint chromosome (m; tau_1,...,tau_m)
  n <- length( y )                                 # n    : sample size
  m <- cp[1]                                       # m    : number of changepoints
  tau <- cp[-1]                                    # tau  : time locations of changepoints

  t <- time( y )

  if ( m==0 ) {
    ar.PLT <- arima( y,order=c(1,0,0),xreg=t,include.mean=TRUE,method="CSS-ML" )
    coef_reg <- coef( ar.PLT )[-1]                 # remove AR(1) parameter
    y_trd <- coef_reg[1] + coef_reg[2]*t
  } else {
    D.m <- matrix( 0, nrow=n, ncol=m )
    for ( k in 1:m ) {
      D.m[,k] <- pmax( t-tau[k], 0 )
    }
    ar.PLT <- arima( y,order=c(1,0,0),xreg=cbind(t,D.m),include.mean=TRUE,method="CSS-ML" )
    coef_reg <- coef( ar.PLT )[-1]                 # remove AR(1) parameter

    y_trd.m <- rep(0,times=n)
    for ( k in 1:m ) {
      y_trd.m <- y_trd.m + coef_reg[2+k]*D.m[,k]
    }
    y_trd <- coef_reg[1] + coef_reg[2]*t + y_trd.m
  }

  return( y_trd )
}

### Generate a simulated series from the fitted PLT model
trj.PLTar <- function( fit, y, cp, n.ahead, is.bootp=TRUE ) { # y    : data
                                                   # cp   : changepoint chromosome (m; tau_1,...,tau_m)
  n <- length( y )                                 # n    : sample size
  m <- cp[1]                                       # m    : number of changepoints
  tau <- cp[-1]                                    # tau  : time locations of changepoints

  t <- time( y )
  t.1 <- time( y )[1]                              # t.1  : 1st time point in y
  t.n <- time( y )[n]                              # t.n  : last time point in y

  t_all <- ts( t.1:(t.n+n.ahead), start=t.1 )
  t_new <- ts( (t.n+1):(t.n+n.ahead), start=t.n+1 )

  # Compute piecewise linear trends mean function
  coef_fit <- coef( fit )                          # fitted model parameter estimates
  coef_reg <- coef_fit[-1]                         # remove AR(1) parameter
  coef_ar1 <- coef_fit[1]                          # AR(1) parameter

  if ( m==0 ) {
    y_trd <- coef_reg[1] + coef_reg[2]*t_all
  } else {
    D.m_all <- matrix( 0, nrow=n+n.ahead, ncol=m )
    for ( k in 1:m ) {
      D.m_all[,k] <- pmax( t_all-tau[k], 0 )
    }
    y_trd.m <- rep(0,times=n+n.ahead)
    for ( k in 1:m ) {
      y_trd.m <- y_trd.m + coef_reg[2+k]*D.m_all[,k]
    }
    y_trd <- coef_reg[1] + coef_reg[2]*t_all + y_trd.m
  }

  w <- resid( fit )                                # w    : residual series of the fit
  w_rct <- w[(n+1-n.ahead):n]                      # w.rct: most recent 28 residuals 

  if (is.bootp==TRUE) {   
    w_trj <- ts( sample( x=w_rct,size=n.ahead,replace=FALSE ), start=t.n+1 )
  } else {
    w_trj <- ts( rnorm( n=n.ahead,mean=0,sd=sd(w_rct) ), start=t.n+1 )
  }

  y_trj <- ts( rep(0,times=n.ahead), start=t.n+1 )
  y_trj[1] <- y_trd[n+1] + coef_ar1*(y[n]-y_trd[n]) + w_trj[1]
  for (k in 2:n.ahead) {
    y_trj[k] <- y_trd[n+k] + coef_ar1*(y_trj[k-1]-y_trd[n+k-1]) + w_trj[k]
  }

  return( y_trj )
}

### Predict for a PLT model under AR(1) and changepoints
fct.PLTar <- function( fit, y, cp, n.ahead ) {     # y    : data
                                                   # cp   : changepoint chromosome (m; tau_1,...,tau_m)
  n <- length( y )                                 # n    : sample size
  m <- cp[1]                                       # m    : number of changepoints
  tau <- cp[-1]                                    # tau  : time locations of changepoints

  t <- time( y )
  t.1 <- time( y )[1]                              # t.1  : 1st time point in y
  t.n <- time( y )[n]                              # t.n  : last time point in y

  t_all <- ts( t.1:(t.n+n.ahead), start=t.1 )
  t_new <- ts( (t.n+1):(t.n+n.ahead), start=t.n+1 )

  if ( m==0 ) {
    ar.PLT_fct <- predict( fit, n.ahead=n.ahead, newxreg=t_new )
  } else {
    D.m_all <- matrix( 0, nrow=n+n.ahead, ncol=m )
    for ( k in 1:m ) {
      D.m_all[,k] <- pmax( t_all-tau[k], 0 )
    }
    D.m_new <- D.m_all[(n+1):(n+n.ahead),]
    D.m <- D.m_all[1:n,]
    ar.PLT_fct <- predict( fit, n.ahead=n.ahead, newxreg=cbind(t_new,D.m_new) )
  }

  return( ar.PLT_fct )
}

### Compute BIC of a SMS model fit under independence and changepoints
fit.SMS_BIC <- function( y, cp ) {                 # y    : data
                                                   # cp   : changepoint chromosome (m; tau_1,...,tau_m)
  n <- length( y )                                 # n    : sample size
  m <- cp[1]                                       # m    : number of changepoints

  tau.ext <- c(1,cp[-1],n+1)

  D.m <- matrix( 0, nrow=n, ncol=m+1 )             # m+1 regimes if n.cpt = m
  for ( k in 1:(m+1) ) {
    D.m[tau.ext[k]:(tau.ext[k+1]-1),k] <- 1
  }

  lm.SMS <- lm( y ~ -1 + D.m )

  return( BIC(lm.SMS) )
}

### Compute BIC of a SMS model fit under AR(1) and changepoints
fit.SMSar_BIC <- function( y, cp ) {               # y    : data
                                                   # cp   : changepoint chromosome (m; tau_1,...,tau_m)
  n <- length( y )                                 # n    : sample size
  m <- cp[1]                                       # m    : number of changepoints

  tau.ext <- c(1,cp[-1],n+1)

  D.m <- matrix( 0, nrow=n, ncol=m+1 )             # m+1 regimes if n.cpt = m
  for ( k in 1:(m+1) ) {
    D.m[tau.ext[k]:(tau.ext[k+1]-1),k] <- 1
  }

  ar.SMS <- arima( y,order=c(1,0,0),xreg=D.m,include.mean=FALSE,method="CSS-ML" )

  return( BIC(ar.SMS) )
}

### GA for time series
ga.cpt_ts <- function(y,fitness,gen.size,max.itr,p.mut,
                      seed,is.graphic,is.print,is.export) {
  n <- length(y)                                   # sample size
  t.1 <- time(y)[1]                                # 1st time point
  t.n <- time(y)[n]                                # last time point

  # Changepoint configuration
  m.max <- 5                                       # max number of possible changepoints in generation 1
  t.rng <- (t.1+4):(t.n-3)                         # range of allowed changepoint times
  Confg <- list()                                  # changepoint configuration for a generation
  Confg.sol <- list()                              # best changepoint configuration for a generation
  Confg.ALL <- list()                              # changepoint configuration for all generations

  Pnlik <- matrix(0,nrow=max.itr,ncol=gen.size)    # penalized likelihood for all changepoint configurations
  Pnlik.sol <- numeric(length=max.itr)             # smallest penalized likelihood value for each generation

  if (is.graphic) {
    dev.new(width=12,height=6)
  }

  # Initial generation
  set.seed(seed)
  for (g in 1:1) {
    if (is.print) print(paste("#----------[  Generation =",g,"has begun at",Sys.time()," ]----------#"))

    Confg[[1]] <- as.integer(0)                    # A chromosome of no changepoints is always considered
    j <- 2                                         # loop index for generation
    for (k in 1:(gen.size*1000)) {                 # This loop still works both when n.cpt>=1 and n.cpt=0
      m <- rbinom(1,size=m.max,prob=0.4)           # [!CAUTION!] Adjust prob=0.4 for other settings
      tau <- sort(sample(t.rng,size=m,replace=FALSE))
      chrom <- c(m,tau)                            # changepoint locations (m; tau_1,...,tau_m)
      Confg[[j]] <- as.integer(chrom)

      is.pass <- FALSE
      if (m == 0) {
        is.pass <- TRUE
      } else {
        if (all(diff(c(1,tau,n+1)) > 4)) {         # allow two consecutive changepoint times > 4
          is.pass <- TRUE
        }
      }

      if (length(unique(Confg[1:j])) == j & is.pass == TRUE) {
        j <- j+1                                   # generation increases when (1) a new child chrom is born
      }                                            #                       and (2) the above condition is met  

      if (j > gen.size) break                      # Produce a generation of gen.size
    }                                              # Ending loop in k

    ### Compute penalized log-likelihood for each chromosome
    for (j in 1:gen.size) {
      chrom <- Confg[[j]]

      if (is.print) print(chrom)

      Pnlik[g,j] <- fitness(y=y,cp=chrom)

      if (is.graphic) {
        plot.ts(y,xlab="Time",ylab="",col="gray",
                main=paste("Generation",g,"& Child",j,"( PLKHD =",format(Pnlik[g,j],nsmall=3),")"))
        abline(v=chrom[-1],col="blue",lty=2)
      }
    }

    loc.sol <- which(Pnlik[g,] == min(Pnlik[g,]))
    chrom.sol <- Confg[[loc.sol]]
    Confg.sol[[g]] <- chrom.sol
    Confg.ALL[[g]] <- Confg
    Pnlik.sol[g] <- Pnlik[g,loc.sol]

    if (is.export) {
      capture.output(Confg,file=paste(WD.out,sprintf("GA-Gen_%03d.txt",g),sep=""),append=FALSE)
      write.table(t(format(Pnlik[g,],nsmall=12)),file=paste(WD.out,"GA-Pnlik.csv",sep=""),
                  sep=",",quote=FALSE,row.names=FALSE,col.names=FALSE,append=FALSE)
    }
  }                                                # Ending loop in g

  # Next generations from 2 to gen.size
  for (g in 2:max.itr) {
    if (is.print) print(paste("#----------[  Generation =",g,"has begun at",Sys.time()," ]----------#"))

    # Rank chromosomes in the (g-1)th generation
    gen.rank <- rank(-Pnlik[g-1,])
    gen.rank.sum <- sum(gen.rank)

    # Generate g-th generation: the fittest chromosome carries over to next generation
    Confg.pre <- Confg.ALL[[g-1]]
    Confg[[1]] <- Confg.sol[[g-1]]
    Pnlik[g,1] <- Pnlik.sol[g-1]

    j <- 2                                         # index for child in a generation
    for (k in 2:(gen.size*1000)) {
      # Select father and mother chromosomes
      loc.prt <- sample(1:gen.size,size=2,replace=FALSE,prob=gen.rank/gen.rank.sum)
      loc.dad <- loc.prt[1]
      loc.mom <- loc.prt[2]
      chrom.dad <- Confg.pre[[loc.dad]]
      chrom.mom <- Confg.pre[[loc.mom]]

      # Producing child chromosomes
      # Step 1: Combining
      tau_S1 <- sort(union(chrom.dad[-1],chrom.mom[-1]))  # Do not allow identical changepoint times
      m_S1 <- length(tau_S1)
      if (m_S1 == 0) {
        # Step 2: Thinning (SKIP!!!)
        # Step 3: Shifting (SKIP!!!)
        # Step 4: Mutation
        m_S4 <- rbinom(1,size=2,prob=p.mut)               # [!CAUTION!] Adjust p.mut for other settings
        tau_S4 <- sort(sample(t.rng,size=m_S4,replace=FALSE))
      } else {
        # Step 2: Thinning
        ran.val_S2 <- runif(m_S1,min=0,max=1)
        tau_S2 <- tau_S1[ran.val_S2 <= 0.5]
        m_S2 <- length(tau_S2)

        # Step 3: Shifting
        ran.val_S3 <- sample(c(-1,0,1),size=m_S2,replace=TRUE,prob=c(0.3,0.4,0.3))
        tau_S3.tmp <- sort(unique(tau_S2+ran.val_S3))
        tau_S3 <- tau_S3.tmp[tau_S3.tmp %in% t.rng]       # Changepoints must occur in t.rng
        m_S3 <- length(tau_S3)

        # Step 4: Mutation
        m_S4.mut <- rbinom(1,size=2,prob=p.mut)           # [!CAUTION!] Adjust p.mut for other settings
        tau_S4.mut <- sort(sample(t.rng,size=m_S4.mut,replace=FALSE))
        tau_S4 <- sort(unique(c(tau_S3,tau_S4.mut)))
        m_S4 <- length(tau_S4)
      }

      m <- m_S4                                    # number of changepoints
      tau <- tau_S4
      chrom <- c(m,tau)                            # changepoint locations (m; xi_1,...,xi_m)
      Confg[[j]] <- as.integer(chrom)

      is.pass <- FALSE
      if (m == 0) {
        is.pass <- TRUE
      } else {
        if (all(diff(c(1,tau,n+1)) > 4)) {         # allow two consecutive changepoint times > 4
          is.pass <- TRUE
        }
      }

      if (length(unique(Confg[1:j])) == j & is.pass == TRUE) {
        j <- j+1                                   # generation increases when (1) a new child chrom is born
      }                                            #                       and (2) the above condition is met  

      if (j > gen.size) break                      # Produce a generation of gen.size
    }                                              # Ending loop in k

    # Compute penalized log-likelihood for each chromosome
    for (j in 1:gen.size) {
      chrom <- Confg[[j]]

      if (is.print) print(chrom)

      Pnlik[g,j] <- fitness(y=y,cp=chrom)

      if (is.graphic) {
        plot.ts(y,xlab="Time",ylab="",col="gray",
                main=paste("Solution in Generation",g-1,
                           "( PLKHD =",format(Pnlik.sol[g-1],nsmall=3),") vs",
                           "Generation",g,"& Child",j,
                           "( PLKHD =",format(Pnlik[g,j],nsmall=3),")"))
        abline(v=chrom.sol[-1],col="red",lty=1)
        abline(v=chrom[-1],col="blue",lty=2)
      }
    }

    loc.sol <- which(Pnlik[g,] == min(Pnlik[g,]))

    chrom.sol <- Confg[[loc.sol]]
    Confg.sol[[g]] <- chrom.sol
    Confg.ALL[[g]] <- Confg
    Pnlik.sol[g] <- Pnlik[g,loc.sol]

    if (is.print) {
      print(c(k,j))
      print(chrom.sol)
      print(paste("MDL =",format(Pnlik.sol[g],nsmall=3)),quote=FALSE)
    }

    if (is.export) {
      capture.output(Confg,file=paste(WD.out,sprintf("GA-Gen_%03d.txt",g),sep=""),append=FALSE)
      write.table(t(format(Pnlik[g,],nsmall=12)),file=paste(WD.out,"GA-Pnlik.csv",sep=""),
                  sep=",",quote=FALSE,row.names=FALSE,col.names=FALSE,append=TRUE)
    }
  }                                                # Ending loop in g

  list(gen.all=Confg.ALL,gen.sol=Confg.sol,val.all=Pnlik,val.sol=Pnlik.sol,solution=chrom.sol)
}                                                  # Ending function: ga.cpt_ts

