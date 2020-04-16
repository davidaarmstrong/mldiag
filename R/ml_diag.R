#' Function for evaluating model fit.
#'
#' Model fit function for the \code{ml_diag} function.
#'
#' Returns either the square correlation between the dependent variable and its
#' predictions or the expected proportional reduction in error if the model is
#' a binomial GLM.
#'
#' @param y Dependent variable used in the model.
#' @param yhat Predictions of the dependent variable.
#' @param type Character string giving the type of model used (currently only binomial and gaussian are supported).
#'
#' @return A scalar giving the model fit
#' @export

corfun <- function(y, yhat, type=c("gaussian", "binomial")){
  type <- match.arg(type)
  if(type == "gaussian"){
    res <- cor(y, yhat, use="pair")^2
  }
  if(type == "binomial"){
    tab <- table(round(y))
    pmc <- max(tab)/sum(tab)
    pcp <- mean(round(y) == round(yhat))
    res <- (pcp-pmc)/(1-pmc)
  }
  return(res)
}

#' Xgboost shrinkage function
#'
#' Xgboost shrinkage function for the \code{ml_diag} function.
#'
#' @param obj An object of class \code{lm} or \code{glm} with \code{family=binomial}.
#' @param train A vector of observations to be used in the training set.
#' @param test A vector of observations to be used in the testing set.
#' @param ... Other arguments to be passed down to the \code{xgboost} function.
#'
#' @return A list which contains the predicted values for the training set, predicted values for the
#' testing set and the \code{xgboost} model used to generate the predictions.
#'
#' @export
#'
#' @importFrom xgboost xgb.cv xgboost
xgb_shrink <- function(obj, train, test, ...){
  UseMethod("xgb_shrink")
}

#' @export
#' @method xgb_shrink lm
xgb_shrink.lm <- function(obj, train, test=NULL, ...){
  X <- model.matrix(obj)
  y <- model.response(model.frame(obj))
  args <- list(...)
  args$data <- X[train, -1]
  args$label <- y[train]
  if(!("objective" %in% names(args))){
    args$objective <- "reg:linear"
  }
  args.cv <- args
  args.cv$nrounds <- 100
  args.cv$nfold <- ifelse(nrow(X)/5 > 25, 5, floor(nrow(X)/25))
  gb1 <- do.call(xgb.cv, args.cv)
  w <- which.min(c(unlist(gb1$evaluation_log[,4])))
  args$nrounds <- w
  gb <- do.call(xgboost, args)
  yhat.train <- predict(gb, newdata = X[train, -1])
  if(!is.null(test)){
    yhat.test <- predict(gb, newdata=model.matrix(obj)[test, -1])
  }else{
    yhat.test <- NULL
  }
  return(list(yhat.train = yhat.train, yhat.test=yhat.test, mod = gb))
}

#' @export
#' @method xgb_shrink glm
xgb_shrink.glm <- function(obj, train, test=NULL, ...){
  X <- model.matrix(obj)
  y <- model.response(model.frame(obj))
  args <- list(...)
  args$data <- X[train, -1]
  args$label <- y[train]
  if(!("objective" %in% names(args))){
    args$objective <- "reg:logistic"
  }
  args.cv <- args
  args.cv$nrounds <- 100
  args.cv$nfold <- ifelse(nrow(X)/5 > 25, 5, floor(nrow(X)/25))
  gb1 <- do.call(xgb.cv, args.cv)
  w <- which.min(c(unlist(gb1$evaluation_log[,4])))
  args$nrounds <- w
  gb <- do.call(xgboost, args)
  yhat.train <- predict(gb, newdata = X[train, -1])
  if(!is.null(test)){
    yhat.test <- qlogis(predict(gb, newdata=model.matrix(obj)[test, -1]))
  }else{
    yhat.test <- NULL
  }
  return(list(yhat.train = yhat.train, yhat.test=yhat.test, mod = gb))
}

#' bartMachine shrinkage function
#'
#'bartMachine shrinkage function for the \code{ml_diag} function.
#'
#' @param obj An object of class \code{lm} or \code{glm} with \code{family=binomial}.
#' @param train A vector of observations to be used in the training set.
#' @param test A vector of observations to be used in the testing set.
#' @param ... Other arguments to be passed down to the \code{bartMachine} function.
#'
#' @return A list which contains the predicted values for the training set, predicted values for the
#' testing set and the \code{bartMachine} model used to generate the predictions.
#'
#' @export
#'
#' @importFrom bartMachine bartMachine
#' @importFrom tidyr unnest
#' @importFrom magrittr %>%
#' @name %>%
bm_shrink <- function(obj, train, test, ...){
  UseMethod("bm_shrink")
}

#' @export
#' @method bm_shrink lm
bm_shrink.lm <- function(obj, train, test=NULL, ...){
    X <- model.frame(obj) %>% unnest(cols=c()) %>% as.data.frame()
    X <- cbind(X[,-1], X[[1]])
    names(X) <- paste0("x", 1:ncol(X))
    names(X)[ncol(X)] <- "y"
    args <- list(...)
    args$Xy <- X[train, ]
    b1 <- do.call(bartMachine, args)
    yhat.train <- b1$y_hat_train
    if(!is.null(test)){
      yhat.test <- predict(b1, new_data=X[test, -ncol(X)])
    }else{
      yhat.test <- NULL
    }
  return(list(yhat.train = yhat.train, yhat.test=yhat.test, mod = b1))
}

#' @export
#' @method bm_shrink lm
bm_shrink.glm <- function(obj, train, test=NULL, ...){
  X <- model.frame(obj) %>% unnest(cols=c()) %>% as.data.frame()
  X <- cbind(X[,-1], X[[1]])
  names(X) <- paste0("x", 1:ncol(X))
  names(X)[ncol(X)] <- "y"
  if(!is.factor(X$y)){
    X$y <- as.factor(X$y)
  }
  args <- list(...)
  args$Xy <- X[train, ]
  b1 <- do.call(bartMachine, args)
  yhat.train <- qnorm(predict(b1, new_data=X[train, -ncol(X)], type="prob"))
  if(!is.null(test)){
    yhat.test <- qnorm(predict(b1, new_data=X[test, -ncol(X)], type="prob"))
  }else{
    yhat.test <- NULL
  }
  return(list(yhat.train = yhat.train, yhat.test=yhat.test, mod = b1))
}


#' Random forest shrinkage function
#'
#' Random forest shrinkage function for the \code{ml_diag} function.
#'
#' @param obj An object of class \code{lm} or \code{glm} with \code{family=binomial}.
#' @param train A vector of observations to be used in the training set.
#' @param test A vector of observations to be used in the testing set.
#' @param ... Other arguments to be passed down to the \code{randomForest} function.
#'
#' @return A list which contains the predicted values for the training set, predicted values for the
#' testing set and the \code{randomForest} model used to generate the predictions.
#'
#' @export
#'
#' @importFrom randomForest randomForest
rf_shrink <- function(obj, train, test, ...){
  UseMethod("rf_shrink")
}

#' @export
#' @method rf_shrink lm
rf_shrink.lm <- function(obj, train, test=NULL, ...){
  X <- model.matrix(obj)
  y <- model.response(model.frame(obj))
  args <- list(...)
  args$x <- X[train, -1]
  args$y <- y[train]
  r1 <- do.call(randomForest, args)
  yhat.train <- predict(r1, newdata=X[train, -1], type="response")
  if(!is.null(test)){
    yhat.test <- predict(r1, newdata=X[test, ], type="response")
  }else{
    yhat.test <- NULL
  }
  return(list(yhat.train = yhat.train, yhat.test=yhat.test, mod = r1))
}

#' @export
#' @method rf_shrink lm
rf_shrink.glm <- function(obj, train, test=NULL, ...){
  X <- model.matrix(obj)
  y <- model.response(model.frame(obj))
  if(!is.factor(y)){
    y <- as.factor(y)
  }
  args <- list(...)
  args$x <- X[train, -1]
  args$y <- y[train]
  r1 <- do.call(randomForest, args)
  p1 <- predict(r1, newdata=X[train, -1], type="prob")[,2]
  p1b <- p1[-which(p1 == 1 | p1 == 0)]
  p1 <- ifelse(p1 == 1, max(p1b) + (1-max(p1b))/2, p1)
  p1 <- ifelse(p1 == 0, min(p1b) - .5*min(p1b), p1)
  yhat.train <- qnorm(p1)
  if(!is.null(test)){
    p2 <- predict(r1, newdata=X[test, ], type="prob")[,2]
    p2b <- p2[-which(p2 == 1 | p2 == 0)]
    p2 <- ifelse(p2 == 1, max(p2b) + (1-max(p2b))/2, p2)
    p2 <- ifelse(p2 == 0, min(p2b) - .5*min(p2b), p2)
    yhat.test <- qnorm(p2)
  }else{
    yhat.test <- NULL
  }
  return(list(yhat.train = yhat.train, yhat.test=yhat.test, mod = r1))
}




#' Machine Learning Diagnostics for Generalized Linear Models
#'
#' A decoupling shrinkage and selection (DSS) approach to model diagnostics.
#'
#' Model diagnostics are often based on model residuals.  The \code{ml_diag} function
#' uses a DSS approach to model diagnostics.  Here, the we use non-parametric
#' machine learning tools (like \code{xgboost}, \code{randomForest} or \code{bartMachine})
#' to generate the best possible predictions from the included model covariates.
#' These predictions serve as an adjusted dependent variable that we predict with the
#' parametric model originally fit to the data.  If the fit of this auxiliary model
#' is good, then the original parametric model is well specified.  If, however, the
#' model fit is poor, then there are important interactions and/or non-linearities
#' that are not accounted for in the original parametric model.  We then either
#' jackknife out each variable or sequentially exclude each variable in turn based
#' on best model fit improvement to see which variables are the cause of
#' problems.
#'
#' @param mod An object of class \code{lm} or a \code{glm} with \code{family=binomial}.
#' @param data A data frame continaing the data used to estimate \code{mod}
#' @param shrinkEngine The methods used in the shrinkage phase of the model.
#' @param shrinkEngine.args Arguments to be passed down to the shrinkage engine.
#' @param sampleProp Proportion of data (randomly sapmled) to use in the analaysis.  The training and testing samples will be returned with the function. Defaults to using 50\% of the data.
#' @param retainMarginal A vector of names of factors in the dataset where you want the marginal distribution to be respected in the training and testing samples.  The random sampling is done within each combination of these values, so unless you have a lot of data, there should be relatively few of these.
#' @param ... Arguments to be passed down to the shrinkage engine.
#'
#' @return A list with the following elements:
#'   \item{paramFit}{The r-squared for the shrinkage estimate regressed on the parametric model specification}
#'   \item{termFits1}{The r-squared for the shrinkage estimates regressed on the parametric model specification with each model term jackknifed out in turn.}
#'   \item{termFits2}{The r-squared from the shrinkage estimates regressed on the parametric model specification with the model terms removed sequentially (and cumulatively) based on lack of from the \code{termFits1} return.}
#'   \item{train.sample}{Observations used in the training sample after data with only model variables had been listwise deleted.}
#'   \item{test.sample}{Observations in the testing sample after data with only model variables had been listwise deleted.}
#'
#' @importFrom stats cor drop.terms family fitted formula gaussian model.frame model.matrix model.response na.omit predict qlogis qnorm terms update
#'
#' @export

ml_diag <- function(
  mod,
  data,
  shrinkEngine =c("xgboost", "randomForest", "bartMachine"),
  shrinkEngine.args=list("xgboost" = list(params = list(max_depth = 4, eta=.1)), "randomForest" = list(), "bartMachine"=list()),
  sampleProp = .5,
  retainMarginal = NULL,
  ...){
  if(!(family(mod)$family %in% c("gaussian", "binomial"))){
    stop("Original model must be either gaussian glm (esitmated with glm or lm) or a binomial glm.\n")
  }
  shrinkEngine <- match.arg(shrinkEngine)
  names(data) <- gsub(".", "_", names(data), fixed=T)
  form <- formula(mod)
  orig <- mod
  form <- formula(mod)
  av <- all.vars(formula(mod))
  pnames <- grep(".", av, fixed=T, value=T)
  if(length(pnames) > 0){
    stop("The following Variables have periods in the name: ", pnames,
         "\n try the following: names(data) <- gsub('.', '_', names(data), fixed=TRUE)\n Then re-run the model with the new variable names")
  }
  yhat.orig <- predict(mod, type="response")
  form <- formula(mod)
  av <- all.vars(formula(mod))
  Xdat <- data[, av[-1]]
  Xdat <- na.omit(Xdat)
  for(i in 1:ncol(Xdat)){
    if(is.factor(Xdat[[i]])){
      Xdat[[i]] <- droplevels(Xdat[[i]])
    }
  }
  if(sampleProp < 1){
    if(!is.null(retainMarginal)){
      if(!all(retainMarginal %in% names(Xdat))){
        stop("All variables in retainMarginal must be in the input data\n")
      }
      samples <- by(1:nrow(Xdat), list(Xdat[,retainMarginal]), function(x){
        train = sample(x, ceiling(sampleProp*length(x)), replace=F)
        test = setdiff(x, train)
        if(length(test) == 0){
          warning("Use of retainMarginal led to test-set cells of size 0.  In this case, the training set and testing set are the same.\n")
          test = train
        }
        return(list(test=test, train=train))
      })
      train.sample <- c(unlist(lapply(samples, function(x)x$train)))
      test.sample <- c(unlist(lapply(samples, function(x)x$test)))
    }
    else{
      train.sample <- sample(1:nrow(Xdat), floor(sampleProp*nrow(Xdat)), replace=FALSE)
      test.sample <- setdiff(1:nrow(Xdat), train.sample)
    }
  }else{
      train.sample <- 1:nrow(Xdat)
      test.sample <- NULL
  }

  shrink_fun <- switch(shrinkEngine,
                      "xgboost" = "xgb_shrink",
                      "randomFrest" = "rf_shrink",
                      "bartMachine" = "bm_shrink")

#  out <- do.call(shrink_fun, list(obj=mod, train=train.sample, test=test.sample))
  out <- do.call(shrink_fun, list(obj=mod, train=train.sample, test=test.sample, ...))
  if(inherits(mod, "glm")){
    updated.mod <- update(mod, out$yhat.train ~ ., data=data[train.sample, ], family=gaussian)
  }else{
    updated.mod <- update(mod, out$yhat.train ~ ., data=data[train.sample, ])
  }
  paramFit <- cor(fitted(updated.mod), model.response(model.frame(updated.mod)))^2
  nTerms <- length(attr(terms(mod), "term.labels"))
  termFits <- rep(NA, nTerms)
  for(i in 1:nTerms){
    tmp.terms <- drop.terms(terms(mod), dropx = i, keep.response=TRUE)
    tmp.mod <- update(mod, formula(tmp.terms))
#    tmp.out <- do.call(shrink_fun, list(obj=mod, train=train.sample, test=test.sample))
    tmp.out <- do.call(shrink_fun, list(obj=mod, train=train.sample, test=test.sample, ...))
    if(inherits(mod, "glm")){
      updated.tmp <- update(mod, tmp.out$yhat.train ~ ., data=data[train.sample, ], family=gaussian)
    }else{
      updated.tmp <- update(mod, tmp.out$yhat.train ~ ., data=data[train.sample, ])
    }
    termFits[i] <- cor(fitted(updated.tmp), model.response(model.frame(updated.tmp)))^2
  }
  termFits <- data.frame(
    variable = factor(1:nTerms, labels=attr(terms(mod), "term.labels")),
    fit = termFits)

  fitOrder <- attr(terms(mod), "term.labels")[order(termFits$fit)]
  termFits2 <- rep(NA, nTerms-1)
  tmp.terms <- terms(mod)
  for(i in 1:(nTerms-1)){
    labs <- attr(tmp.terms, "term.labels")
    tmp.terms <- drop.terms(tmp.terms, dropx= which(labs == fitOrder[i]), keep.response=TRUE)
    tmp.mod <- update(mod, formula(tmp.terms))
#    tmp.out <- do.call(shrink_fun, list(obj=mod, train=train.sample, test=test.sample))
    tmp.out <- do.call(shrink_fun, list(obj=mod, train=train.sample, test=test.sample, ...))
    if(inherits(mod, "glm")){
      updated.tmp <- update(mod, tmp.out$yhat.train ~ ., data=data[train.sample, ], family=gaussian)
    }else{
      updated.tmp <- update(mod, tmp.out$yhat.train ~ ., data=data[train.sample, ])
    }
    termFits2[i] <- cor(fitted(updated.tmp), model.response(model.frame(updated.tmp)))^2
  }
  termFits2 <- data.frame(
    variabel = factor(1:length(termFits2), labels = fitOrder[-length(fitOrder)]),
    fit = termFits2)
  return(list(
    paramFit = paramFit,
    termFits1 = termFits,
    termFits2 = termFits2,
    train.sample = train.sample,
    test.sample = test.sample
  ))

}



