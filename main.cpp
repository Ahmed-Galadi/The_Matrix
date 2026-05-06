#include <iostream>
#include "HiddenLayer.hpp"

int main() {
    // ===== 1. Create layer (2 inputs -> 1 neuron) =====
    HiddenLayer layer(2, 1);

    // Manually set weights and biases for predictable results
    Matrix w(2, 1);
    w(1,1) = 1.0;   // weight for input 1
    w(2,1) = 1.0;   // weight for input 2
    layer.setWeights(w);

    Matrix b(1, 1, 0.0);   // bias = 0
    layer.setBiases(b);

    // ===== 2. Prepare input data =====
    Matrix X(3, 2);         // 3 samples, 2 features each
    X(1,1) = 1.0;  X(1,2) = 2.0;
    X(2,1) = 1.0;  X(2,2) = 2.0;
    X(3,1) = 1.0;  X(3,2) = 2.0;

    // ===== 3. Forward pass =====
    Matrix out = layer.forward(X);
    std::cout << "Forward output (should be all 3.0):\n";
    out.print();
    std::cout << "\n";

    // ===== 4. Prepare upstream gradient (same shape as output) =====
    Matrix upstreamGrad(3, 1, 1.0);   // all 1.0

    // ===== 5. Backward pass =====
    double learningRate = 0.1;
    Matrix inputGrad = layer.backward(upstreamGrad, learningRate);

    // ===== 6. Print results =====
    std::cout << "Input gradient (should be all 1.0):\n";
    inputGrad.print();
    std::cout << "\n";

    std::cout << "New weights (should be [0.7, 0.4]):\n";
    layer.getWeights().print();
    std::cout << "\n";

    std::cout << "New biases (should be [-0.3]):\n";
    layer.getBiases().print();

    return 0;
}