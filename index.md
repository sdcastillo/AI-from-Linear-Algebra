---
layout: default
title: AI from Linear Algebra
description: Least squares, Gaussian GLMs, and the matrix factorizations that modern models still run on.
samwiki: true
---

A supervised model is a linear map until the loss or a nonlinearity says otherwise. Here the response is a crash score. Each row of `model_matrix.csv` is a roadway, weather, and work-area feature vector, and `y.csv` holds the numeric scores. With $$N = 17354$$ observations and $$p = 20$$ columns, including an intercept, a Gaussian GLM with the identity link is ordinary least squares: the mean is $$X\beta$$, and the fitted vector is the orthogonal projection of $$y$$ onto the column space of $$X$$.

The SciLab script solves that projection with the normal equations. When $$X$$ has full column rank, the Gram matrix $$X^\top X$$ is symmetric positive definite and the minimizer of $$\lVert y - X\beta \rVert_2^2$$ is

$$
\hat\beta = (X^\top X)^{-1} X^\top y.
$$

The same vector is the maximum-likelihood estimate when the noise is independent and $$y \sim \mathcal{N}(X\beta, \sigma^2 I)$$. Residuals $$r = y - X\hat\beta$$ give the unbiased variance $$\hat\sigma^2 = \lVert r \rVert_2^2 / (N - p)$$, and coefficient standard errors are the square roots of the diagonal of $$\hat\sigma^2 (X^\top X)^{-1}$$. The script’s $$t$$-statistics are $$\hat\beta_j / \mathrm{se}(\hat\beta_j)$$, with a normal tail approximation once the residual degrees of freedom are large.

Forming $$X^\top X$$ squares the condition number, so a modestly ill-conditioned design becomes unstable under an explicit inverse. A thin QR factorization $$X = QR$$ recovers the same coefficients from $$\hat\beta = R^{-1} Q^\top y$$ without building the Gram matrix. Production least squares, and the linear solve inside each Newton step of a GLM, prefers that QR factorization or a Cholesky factor of the Gram matrix over an explicit inverse.

The same product is a layer. A dense map $$z = Wx + b$$ is an affine function; a network stacks those maps under a nonlinearity $$\sigma$$. For square loss, the gradient in the linear predictor is the residual, and the gradient in $$W$$ is an outer product of that residual with the input. At a least-squares optimum the normal equations say the same thing: $$X^\top(y - X\hat\beta) = 0$$. Gradient descent on a positive-definite quadratic walks to that point. The closed form jumps there.

Choosing the model is choosing the columns. The R formula `Crash_Score ~ . + Work_Area * Rd_Class - Month` builds a design matrix, drops the month main effect, and adds the interaction as a product of two columns. Logistic regression and a softmax classifier keep a design matrix and change the loss from squared error to cross-entropy, so the score equations are nonlinear and Newton or gradient steps replace the single solve. Attention uses the same geometry with different factors: scores are $$QK^\top / \sqrt{d}$$, a scaled Gram matrix of queries and keys. Principal components are the leading right singular vectors of a centered data matrix, and truncating the SVD is the optimal rank-$$k$$ approximation in the Frobenius norm.

- **Gaussian GLM.** $$\hat\beta = (X^\top X)^{-1} X^\top y$$ on `model_matrix.csv` and `y.csv` is the MLE under $$y \sim \mathcal{N}(X\beta, \sigma^2 I)$$, with $$\hat\sigma^2 = \mathrm{RSS}/(N-p)$$ and $$N - p = 17334$$.
- **Interactions as columns.** `Work_Area * Rd_Class` is a product of two feature columns, appended to $$X$$. The hypothesis space is the span of the columns that remain.
- **GLMs without a single solve.** Logistic and Poisson models still use a design matrix, but the Hessian depends on $$\hat\beta$$. Fisher scoring repeatedly solves a weighted least-squares problem $$X^\top W X \, \delta = X^\top W z$$.
- **Deep learning.** Each dense layer is a matrix multiply, and backpropagation multiplies Jacobians. With square loss and no nonlinearity, the critical point is exactly where the residual is orthogonal to the columns of $$X$$.
- **Factorizations used as models.** QR stabilizes the solve in `scilab.txt`. The SVD of a centered matrix is PCA. Low-rank adapters and embedding tables are constrained factorizations of a larger weight matrix.

<script async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"></script>
