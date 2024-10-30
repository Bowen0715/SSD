#' Supervised Sparse Decomposition (SSD) Class
#'
#' This class implements Supervised Sparse Decomposition (SSD) for supervised feature extraction.
#'
#' @importFrom R6 R6Class
#' 
#' @field X Input matrix, containing the data for decomposition.
#' @field L Target matrix, used in supervised decomposition.
#' @field D Dictionary matrix, initialized or learned.
#' @field Z Coefficients matrix for sparse coding.
#' @field W Weight matrix for feature extraction.
#' @field G Diagonal matrix controlling sparsity in decomposition.
#' @field A Regression matrix for supervised learning.
#' 
#' @field h Dimensionality of the latent space.
#' @field lambda Regularization parameter for decomposition.
#' @field nepoch Number of epochs for model training.
#' @field beta Regularization parameter for supervised learning.
#' @field recon_err_ra Ratio of approximation error in the input space.
#' @field Z_approx Approximation of matrix Z.
#' @field reg_mse Mean squared error for supervised regression.
#'
#' @section Methods:
#' - \code{initialize(h, lambda, nepoch, beta)}: Initializes the SSD model with given parameters.
#' - \code{fit(X, L, D)}: Fits the SSD model to the input data X and target L, optionally with dictionary matrix D.
#'
#' @export
SSD <- R6::R6Class("Supervised Sparse Decomposition",
            public = list(
                X = NULL,
                L = NULL,
                D = NULL,
                Z = NULL,
                W = NULL,
                G = NULL,
                A = NULL,

                h = NULL,
                lambda = NULL,
                nepoch = NULL,
                beta = NULL,

                Z_approx = NULL,
                recon_err_ra = NULL,
                reg_mse = NULL,

                #' @description Initializes an SSD object.
                #' @param h Integer, dimensionality of the latent space.
                #' @param lambda Numeric, regularization parameter.
                #' @param nepoch Integer, number of epochs for training.
                #' @param beta Numeric, regularization parameter for supervised learning.
                initialize = function(h, lambda, nepoch = 1000, beta = 1) {
                    self$h <- h
                    self$lambda <- lambda
                    self$nepoch <- nepoch
                    self$beta <- beta
                },
                
                #' @description Fits the SSD model to input data and target matrix.
                #' @param X Matrix, input data for decomposition.
                #' @param L Matrix, target matrix for supervised decomposition.
                #' @param D Matrix, optional dictionary matrix. If NULL, D will be learned.
                #' @return The fitted SSD object.
                fit = function(X, L, D = NULL) {
                    set.seed(42)
                    A0 <- runif(1)
                    m <- nrow(X)
                    N <- ncol(X)
                    trloss <- numeric(nepoch)

                    DLparam <- list(
                        'K' = self$h,
                        'lambda1' = self$lambda,
                        'numThreads' = -1,
                        'batchsize' = 2500,
                        'iter' = 100
                    )
                    eps <- .Machine$double.eps
                    FIX_D <- !is.null(D) # Flag for fix D mode
                    if (!FIX_D) {
                        D <- do.call(spams.trainDL, c(list(X = X), DLparam))
                        D <- D %*% diag(1 / (sqrt(colSums(D^2)) + eps))
                    }
                    K <- ncol(D)
                    W <- cbind(matrix(runif(K), nrow = K), t(D))
                    X_intercept <- rbind(rep(1, N), X)


                    G <- diag(K)
                    Z <- NULL
                    A <- matrix(runif(1 * K), nrow = 1)

                    for (ite in 1:self$nepoch) {
                        SigmaWX <- .sigmoid(W %*% X_intercept)
                        
                        if (!is.null(Z)) {
                            tau_d <- 0.99 * 2 / norm(Z %*% t(Z), "F")
                            for (j in 1:50) {
                                if (FIX_D) {
                                    if (self$beta != 0) {
                                        A <- .gradDesc(A, Z, rbind(X, L - c(A0 * rep(1, N))), tau_d)
                                        A <- matrix(A, nrow = 1)
                                        A0 <- mean(A0 - 0.1 * (A0 + A %*% Z - L))
                                    }
                                } else {
                                    if (self$beta != 0) {
                                        DNew <- .gradDesc(rbind(D, A), Z, rbind(X, L - c(A0 * rep(1, N))), tau_d)
                                        D <- DNew[1:(nrow(DNew) - 1), ]
                                        A <- DNew[nrow(DNew), ]
                                        A <- matrix(A, nrow = 1)
                                        A0 <- mean(A0 - 0.1 * (A0 + A %*% Z - L))
                                        D <- abs(D) %*% diag(1 / (sqrt(colSums(D^2)) + eps))
                                    } else {
                                        D <- .gradDesc(D, Z, X, tau_d)
                                        D <- abs(D) %*% diag(1 / (sqrt(colSums(D^2)) + eps))
                                    }
                                }
                            }
                        }

                        if (self$beta != 0) {
                            X_eq <- as.matrix(rbind(X, G %*% SigmaWX, L - c(A0 * rep(1, N))))
                            D_eq <- as.matrix(rbind(D, diag(K), A))
                        } else {
                            X_eq <- as.matrix(rbind(X, G %*% SigmaWX))
                            D_eq <- as.matrix(rbind(D, diag(K)))
                        }
                        
                        D_eq <- D_eq %*% diag(1 / (sqrt(colSums(D_eq^2)) + eps))

                        Z <- spams.lasso(X_eq, D_eq, lambda1 = self$lambda, lambda2 = 0, pos = TRUE, numThreads = -1)
                        Z <- as.matrix(diag(1 / (sqrt(colSums(rbind(D, diag(K), A)^2)) + eps)) %*% Z)

                        SigmaWX <- .sigmoid(W %*% X_intercept)
                        temp <- diag(1 / (rowSums(SigmaWX^2) + eps))
                        G <- diag((temp %*% SigmaWX) %*% t(Z))
                        G <- diag(G)

                        err_old <- norm(G %*% .sigmoid(W %*% X_intercept) - Z, "F") / norm(Z, "F")

                        result <- optim(
                            par = as.vector(W), 
                            fn = .object_fun_matrix_form,
                            gr = .gradient_fun,
                            Z = Z, 
                            X_intercept = X_intercept, 
                            G = G, 
                            control = list(maxit = 100),
                            method = "L-BFGS-B"
                        )
                        W_new <- matrix(result$par, nrow = K)

                        err_cur <- norm(G %*% .sigmoid(W_new %*% X_intercept) - Z, "F") / norm(Z, "F")

                        if (err_cur <= err_old) {
                            W <- W_new
                            err_old <- err_cur
                        }

                        trloss[ite] <- norm(X - D %*% Z, "F")^2 + norm(G %*% .sigmoid(W %*% X_intercept) - Z, "F")^2
                        recon_err_ra <- norm(X - D %*% (G %*% .sigmoid(W %*% X_intercept)), "F") / norm(X, "F")

                        if (self$beta != 0) {
                            A0mat <- matrix(A0, nrow = 1)
                            Z_intercept <- rbind(rep(1, N), Z)
                            L_hat <- cbind(A0mat, A) %*% Z_intercept
                            reg_mse <- mean((L_hat - L)^2)
                            cat(sprintf("Ite %d Object error: %f Z approx err ratio: %f Reg MSE: %f\n", ite, trloss[ite], recon_err_ra, reg_mse))
                            flush.console()
                        } else {
                            cat(sprintf("Ite %d Object error: %f Z approx err ratio: %f\n", ite, trloss[ite], recon_err_ra))
                            flush.console()
                        }
                    }

                    self$recon_err_ra <- recon_err_ra
                    # self$X_error_ratio <- X_error_ratio
                    self$W <- W
                    self$G <- G
                    self$D <- D
                    self$Z <- Z
                    self$Z_approx <- G %*% .sigmoid(W %*% X_intercept)
                    if (self$beta != 0) {
                        self$A <- cbind(A0mat, A)
                        self$reg_mse <- reg_mse
                    }
                }
            ))

#' Gradient Descent Helper Function
#'
#' This function performs gradient descent on the dictionary matrix.
#'
#' @param D Matrix, the dictionary matrix to be updated.
#' @param Z Matrix, the sparse representation matrix.
#' @param X Matrix, input data.
#' @param tau_d Numeric, learning rate for gradient descent.
#' @return Updated dictionary matrix.
#' @noRd
.gradDesc <- function(D, Z, X, tau_d) {
    D - tau_d * (D %*% Z - X) %*% t(Z)
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
