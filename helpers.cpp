#include "helpers.h"
#include <cmath>

double sigmoid(double z) {
    return (1.0 / (1.0 + std::exp(-z)));
}