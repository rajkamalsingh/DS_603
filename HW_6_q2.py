import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Define the new matrix P and vector q
P = np.array([[5.005, 4.995], [4.995, 5.005]])
q = np.array([-2, 4])

# Define the objective function f(x) with log base 10
def f(x):
    x = np.array(x, dtype=float)
    quad_term = 0.5 * x.T @ P @ x
    linear_term = q.T @ x
    log_term = np.log10(np.exp(-2 * x[0]) + np.exp(-x[1]))  # log base 10
    return quad_term + linear_term + log_term

# Define the gradient of f(x)
def grad_f(x):
    x = np.array(x, dtype=float)
    grad_quad = P @ x + q
    exp_term_1 = np.exp(-2 * x[0])
    exp_term_2 = np.exp(-x[1])
    log_term_gradient = np.array([-2 * exp_term_1, -exp_term_2]) / (exp_term_1 + exp_term_2)
    log_term_gradient /= np.log(10)  # Adjust gradient for log base 10
    return grad_quad + log_term_gradient

# Exact line search method to find the optimal step size for gradient descent
def exact_line_search(x, d):
    def f_alpha(alpha):
        return f(x + alpha * d)

    result = minimize_scalar(f_alpha)
    return result.x if result.success else 0.0

# Gradient Descent with Exact Line Search
def gradient_descent_exact(x0, tol=1e-2, max_iter=5000):
    xk = x0
    x_seq = [xk.copy()]
    for _ in range(max_iter):
        grad = grad_f(xk)
        if np.linalg.norm(grad) < tol:
            break
        d = -grad
        alpha = exact_line_search(xk, d)
        xk = xk + alpha * d
        x_seq.append(xk.copy())
    return np.array(x_seq)

# Gradient Descent with Backtracking Line Search
def gradient_descent_backtracking(x0, tol=1e-2, max_iter=5000, alpha_init=0.15, gamma=0.7, beta=0.8):
    xk = x0
    x_seq = [xk.copy()]
    for _ in range(max_iter):
        grad = grad_f(xk)
        if np.linalg.norm(grad) < tol:
            break
        d = -grad
        alpha = alpha_init
        while f(xk + alpha * d) > f(xk) + gamma * alpha * np.dot(grad, d):
            alpha *= beta
        xk = xk + alpha * d
        x_seq.append(xk.copy())
    return np.array(x_seq)

# Initial point
x0 = np.array([1, 2])

# Run both methods
x_seq_exact = gradient_descent_exact(x0)
x_seq_backtracking = gradient_descent_backtracking(x0)

# Print the final solutions obtained by each method
optimal_exact = x_seq_exact[-1]
optimal_backtracking = x_seq_backtracking[-1]

print("Optimal solution with Exact Line Search:", optimal_exact)
print("Optimal solution with Backtracking Line Search:", optimal_backtracking)

# Plot the sequence of solutions for both methods
plt.figure(figsize=(10, 5))
plt.plot(x_seq_exact[:, 0], x_seq_exact[:, 1], 'bo-', label="Exact Line Search")
plt.plot(x_seq_backtracking[:, 0], x_seq_backtracking[:, 1], 'ro-', label="Backtracking Line Search")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Sequence of solutions xk for Gradient Descent Methods")
plt.legend()
plt.grid(True)
plt.show()

