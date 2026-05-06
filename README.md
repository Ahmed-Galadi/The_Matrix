# Linear & Logistic Regression – From Scratch

This branch contains a pure C++ implementation of simple linear regression
and logistic regression, built on a custom **Matrix** class (`Matrix.hpp/.cpp`).
No external ML libraries – only the standard library.

## What we built

### 1. Matrix class
A `Matrix` wrapper around `std::vector<std::vector<double>>` supporting:
- Construction (empty, sized with initial value, copy)
- Element access via `operator()(row, col)` (1‑based indexing)
- Addition, subtraction, scalar multiplication
- Matrix multiplication (row‑by‑column dot product)
- Transpose
- `apply(double (*func)(double))` – element‑wise function mapping
- Pretty printing with Unicode box‑drawing characters (┌ ┐ └ ┘)
- Random initialisation (`randomize(min, max)`)

### 2. Linear regression (gradient descent)
- Reads `data.csv` (mileage, price)
- Builds design matrix `X` (column of 1s for bias, column of normalized mileage)
  and target vector `y` (prices)
- Trains parameters θ₀ (bias) and θ₁ (slope) using:
	pred = X * theta
	error = pred - y
	grad = Xᵗ * error / m
	theta = theta - lr * grad
- Normalization of mileage to [0,1] for stable convergence
- Learning rate ~0.1, 1000 iterations

### 3. Logistic regression
- Converts prices to binary labels (above/below average price)
- Trains separate θ using sigmoid activation:
	z = X * theta
	pred = sigmoid(z)
	error = pred - y_binary

### 4. Visualisation (Python)
- `visualisation.py` reads a CSV dumped by the C++ program
- Plots data points + linear regression line + logistic S‑curve
- Dependencies: `matplotlib` (see `requirements.txt`)

## Key concepts learned
- Representing data as matrices (design matrix)
- Hypothesis function (linear combination)
- Cost function (Mean Squared Error)
- Gradient descent
- Feature scaling / normalization
- Binary classification with logistic regression (sigmoid)
- The split between training and prediction programs

## Branch info
This branch (`regression`) freezes the regression-only stage.
The next phase (neural networks with `DenseLayer`) continues on another branch
or in follow-up commits.
