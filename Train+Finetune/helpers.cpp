#include "helpers.h"
#include <cmath>

double sigmoid(double z) {
    return (1.0 / (1.0 + std::exp(-z)));
}

double sigmoid_derivative(double z) {
    double sig = sigmoid(z);
    return (sig * (1.0 - sig));
}

double relu(double x) {
    return (x > 0 ? x : 0.0);
}

double relu_derivative(double x) {
    return (x > 0 ? 1.0 : 0.0);
}