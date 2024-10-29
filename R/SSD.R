#' Supervised Sparse Decomposition (SSD) Class
#'
#' This class implements Supervised Sparse Decomposition (SSD) for supervised feature extraction.
#'
#' @field X Input matrix, containing the data for decomposition.
#' @field L Target matrix, used in supervised decomposition.
#' @field D Dictionary matrix, initialized or learned.
#' @field Z Coefficients matrix for sparse coding.
#' @field W Weight matrix for feature extraction.
#' @field G Diagonal matrix controlling sparsity in decomposition.
#' @field A Regression matrix for supervised learning.
#' @field h Dimensionality of the latent space.
#' @field lambda_ Regularization parameter for decomposition.
#' @field nepoch Number of epochs for model training.
#' @field method Method for decomposition.
#' @field optmethod Optimization method for weight matrix.
#' @field tauw Learning rate for gradient-based optimization.
#' @field initialization Initialization type for matrices.
#' @field beta Regularization parameter for supervised learning.
#' @field X_approximation_error_ratio Ratio of approximation error in the input space.
#' @field Z_approximation Approximation of matrix Z.
#' @field reg_mse Mean squared error for supervised regression.
#'
#' @section Methods:
#' - \code{initialize(h, lambda_, nepoch, method, optmethod, tauw, initialization, beta)}: Initializes the SSD model with given parameters.
#' - \code{fit(X, L, D)}: Fits the SSD model to the input data X and target L, optionally with dictionary matrix D.
#'
#' @export
SSD <- R6Class("Supervised Sparse Decomposition",
               public = list(
                X = NULL,
                L = NULL,
                D = NULL,
                Z = NULL,
                W = NULL,
                G = NULL,
                A = NULL,
                h = NULL,
                lambda_ = NULL,
                nepoch = NULL,
                method = NULL,
                optmethod = NULL,
                tauw = NULL,
                initialization = NULL,
                beta = NULL,
                X_approximation_error_ratio = NULL,
                Z_approximation = NULL,
                reg_mse = NULL,
                
                #' @description Initializes an SSD object.
                #' @param h Integer, dimensionality of the latent space.
                #' @param lambda_ Numeric, regularization parameter.
                #' @param nepoch Integer, number of epochs for training.
                #' @param method Character, method for decomposition.
                #' @param optmethod Character, optimization method for W.
                #' @param tauw Numeric, learning rate.
                #' @param initialization Integer, type of initialization.
                #' @param beta Numeric, regularization parameter for supervised learning.
                initialize = function(h, lambda_, nepoch, method, optmethod, tauw, initialization, beta) {
                    self$h <- h
                    self$lambda_ <- lambda_
                    self$nepoch <- nepoch
                    self$method <- method
                    self$optmethod <- optmethod
                    self$tauw <- tauw
                    self$initialization <- initialization
                    self$beta <- beta
                },
                
                #' @description Fits the SSD model to input data and target matrix.
                #' @param X Matrix, input data for decomposition.
                #' @param L Matrix, target matrix for supervised decomposition.
                #' @param D Matrix, optional dictionary matrix. If NULL, D will be learned.
                #' @return The fitted SSD object.
                fit = function(X, L, D = NULL) {
                    L_rescale <- L
                    A0_rescale <- runif(1)

                    m <- nrow(X)
                    N <- ncol(X)

                    FIX_D <- !is.null(D)

                    set.seed(42)
                    if (!FIX_D) {
                        D <- do.call(spams.trainDL, c(list(X = X), DLparam))
                        D <- D %*% diag(1 / sqrt(colSums(D^2)))
                    }

                    K <- ncol(D)
                    if (initialization == 1) {
                    W <- cbind(matrix(runif(K), nrow = K), t(D))
                    } else {
                    W <- 0.1 * (-0.05 + 0.1 * matrix(runif(K * (m + 1)), nrow = K))
                    }

                    X_intercept <- rbind(rep(1, N), X)

                    G_old <- diag(K)
                    W_old <- W
                    Z <- NULL
                    A <- matrix(runif(1 * K), nrow = 1)

                    for (ite in 1:nepoch) {
                        G <- G_old
                        W <- W_old
                        SigmaWX <- 1 / (1 + exp(- (W %*% X_intercept)))
                        
                        if (!is.null(Z)) {
                            tau_d <- 0.99 * 2 / norm(Z %*% t(Z), "F")
                            for (j in 1:50) {
                            if (FIX_D) {
                                if (beta != 0) {
                                A <- .gradDesc(A, Z, rbind(X, L_rescale - A0_rescale * rep(1, N)), tau_d)
                                A <- matrix(A, nrow = 1)
                                A0_rescale <- mean(A0_rescale - 0.1 * (A0_rescale + A %*% Z - L_rescale))
                                }
                            } else {
                                if (beta != 0) {
                                DNew <- .gradDesc(rbind(D, A), Z, rbind(X, L_rescale - A0_rescale * rep(1, N)), tau_d)
                                D <- DNew[1:(nrow(DNew) - 1), ]
                                A <- DNew[nrow(DNew), ]
                                A <- matrix(A, nrow = 1)
                                A0_rescale <- mean(A0_rescale - 0.1 * (A0_rescale + A %*% Z - L_rescale))
                                D <- abs(D) %*% diag(1 / sqrt(colSums(D^2)))
                                } else {
                                D <- .gradDesc(D, Z, X, tau_d)
                                D <- abs(D) %*% diag(1 / sqrt(colSums(D^2)))
                                }
                            }
                            }
                        }

                        if (beta != 0) {
                            X_eq <- as.matrix(rbind(X, G %*% SigmaWX, L_rescale - A0_rescale * rep(1, N)))
                            D_eq <- as.matrix(rbind(D, diag(K), A))
                        } else {
                            X_eq <- as.matrix(rbind(X, G %*% SigmaWX))
                            D_eq <- as.matrix(rbind(D, diag(K)))
                        }
                        
                        D_eq <- D_eq %*% diag(1 / sqrt(colSums(D_eq^2)))

                        Z <- spams.lasso(X_eq, D_eq, lambda1 = lambda_, lambda2 = 0, pos = TRUE, numThreads = -1)

                        Z <- diag(1 / sqrt(colSums(rbind(D, diag(K), A)^2))) %*% Z

                        SigmaWX <- .sigmoid(W %*% X_intercept)
                        eps <- .Machine$double.eps
                        temp <- diag(1 / (rowSums(SigmaWX^2) + eps))
                        G <- diag((temp %*% SigmaWX) %*% t(Z))
                        G <- diag(G)

                        err_old <- norm(G %*% .sigmoid(W_old %*% X_intercept) - Z, "F") / norm(Z, "F")
                        if (optmethod == 0) {
                            for (j in 1:50) {
                                W <- ProjW(W, Z, X_intercept, tauw, G)
                            }
                            W_new <- W
                        } else if (optmethod == 1) {
                            result <- optim(
                                par = as.vector(W_old), 
                                fn = .object_fun_matrix_form,
                                gr = .gradient_fun,
                                Z = Z, 
                                X_intercept = X_intercept, 
                                G = G, 
                                control = list(maxit = 100),
                                method = "L-BFGS-B"
                            )
                            W_new <- matrix(result$par, nrow = K)
                        }
                        err_cur <- norm(G %*% .sigmoid(W_new %*% X_intercept) - Z, "F") / norm(Z, "F")

                        if (err_cur <= err_old) {
                            W_old <- W_new
                            err_old <- err_cur
                        } else {
                            tauw <- tauw * 0.1
                        }

                        trloss <- numeric(1)
                        trloss[ite] <- norm(X - D %*% Z, "F")^2 + norm(G %*% .sigmoid(W_old %*% X_intercept) - Z, "F")^2
                        X_approximation_error_ratio <- norm(X - D %*% (G %*% .sigmoid(W_old %*% X_intercept)), "F") / norm(X, "F")

                        A0 <- matrix(A0_rescale, nrow = 1)

                        if (beta != 0) {
                            Z_intercept <- rbind(rep(1, N), Z)
                            L_hat <- cbind(A0, A) %*% Z_intercept
                            reg_mse <- mean((L_hat - L)^2)
                            cat(sprintf("Ite %d Object error: %f Z approx err ratio: %f Reg MSE: %f\n", ite, trloss[ite], X_approximation_error_ratio, reg_mse))
                            flush.console()
                        } else {
                            cat(sprintf("Ite %d Object error: %f Z approx err ratio: %f\n", ite, trloss[ite], X_approximation_error_ratio))
                            flush.console()
                        }

                        G_old <- G
                    }

                    self$X_approximation_error_ratio <- X_approximation_error_ratio
                    # self$X_error_ratio <- X_error_ratio
                    self$W <- W_old
                    self$G <- G
                    self$D <- D
                    self$Z <- Z
                    self$Z_approximation <- G %*% .sigmoid(W_old %*% X_intercept)
                    if (beta != 0) {
                        self$A <- cbind(A0, A)
                        self$reg_mse <- reg_mse
                    }
                }
               ))

#' Gradient Descent Helper Function
#'
#' This function performs gradient descent or mini-batch gradient descent on the dictionary matrix.
#'
#' @param D Matrix, the dictionary matrix to be updated.
#' @param Z Matrix, the sparse representation matrix.
#' @param X Matrix, input data.
#' @param tau_d Numeric, learning rate for gradient descent.
#' @return Updated dictionary matrix.
#' @noRd
.gradDesc <- function(D, Z, X, tau_d) {
  if (ncol(X) < 1000) {
    D <- D - tau_d * (D %*% Z - X) %*% t(Z)
  } else {
    S <- ncol(X)
    r <- sample(S)
    for (i in seq(1, S, by = 1000)) {
      X1 <- X[, r[i:min(i + 999, S)]]
      Z1 <- Z[, r[i:min(i + 999, S)]]
      D <- D - tau_d * (D %*% Z1 - X1) %*% t(Z1)
    }
  }
  return(D)
}

#' Sigmoid Function
#'
#' Applies the sigmoid function element-wise to a matrix or vector.
#'
#' @param x Numeric, input matrix or vector.
#' @return Numeric matrix or vector with sigmoid applied.
#' @noRd
.sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

#' Objective Function for Matrix Form
#'
#' Calculates the objective function value for supervised sparse decomposition.
#'
#' @param theta Numeric, vector representing weight matrix W.
#' @param Z Matrix, sparse representation.
#' @param G Matrix, diagonal matrix for sparsity control.
#' @param X_intercept Matrix, input data with intercept term.
#' @return Numeric, objective function value.
#' @noRd
.object_fun_matrix_form <- function(theta, Z, G, X_intercept) {
  W <- matrix(theta, nrow = nrow(Z), ncol = nrow(X_intercept))
  SigmaWX <- .sigmoid(W %*% X_intercept)
  cost <- sum((G %*% SigmaWX - Z)^2) + 0.0001 * sum(W^2)
  return(cost)
}

#' Gradient Function for Matrix Form
#'
#' Computes the gradient of the objective function for optimization.
#'
#' @param theta Numeric, vector representing weight matrix W.
#' @param Z Matrix, sparse representation.
#' @param G Matrix, diagonal matrix for sparsity control.
#' @param X_intercept Matrix, input data with intercept term.
#' @return Numeric, vector representing the gradient.
#' @noRd
.gradient_fun <- function(theta, Z, G, X_intercept) {
  W <- matrix(theta, nrow = nrow(Z), ncol = nrow(X_intercept))
  SigmaWX <- .sigmoid(W %*% X_intercept)
  dW <- 2 * (G %*% SigmaWX - Z) * (G %*% SigmaWX) * (1 - SigmaWX)
  dW <- dW %*% t(X_intercept) + 2 * 0.0001 * W
  grad <- as.vector(dW)
  return(grad)
}
