library(devtools) # or library(remotes)
# Install the SPAMS R package
install_github("getspams/spams-R")
# Install the SSD R package
install_github("Bowen0715/SSD-R")
set.seed(42)

library(SSD)
# X Input matrix (m x N), m features, N samples.
# L Label vector (1 x N).
# h number of patterns, dimensionality of the latent space.
# lambda Regularization parameter for decomposition.
# nepoch Number of epochs for model training (default 1000).
# beta Regularization parameter for supervised learning (default 1).
X <- matrix(runif(10 * 50), nrow = 10, ncol = 50) 
L <- c(rep(0, 25), rep(1, 25))
h <- 5 # 5 patterns
lambda <- 0.1
nepoch <- 100 

model <- SSD$new(h, lambda, nepoch)
model$fit(X, L)

# D Matrix (m x h), the dictionary matrix to be updated.
# Z Matrix (h x N), the sparse representation matrix.
D <- model$D
Z <- model$Z