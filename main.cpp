#include <iostream>
#include <cmath>
#include "HiddenLayer.hpp"   // includes Matrix.hpp

// Activation functions (as above)
double relu(double x) { return x > 0 ? x : 0; }
double relu_derivative(double x) { return x > 0 ? 1.0 : 0.0; }
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

int main() {
    // ===================== DATA =====================
    Matrix X(4, 2);
    X(1,1)=0; X(1,2)=0;
    X(2,1)=0; X(2,2)=1;
    X(3,1)=1; X(3,2)=0;
    X(4,1)=1; X(4,2)=1;

    Matrix y(4, 1);
    y(1,1)=0;
    y(2,1)=1;
    y(3,1)=1;
    y(4,1)=0;

    // ===================== NETWORK =====================
    HiddenLayer hidden(2, 4);   // 2 inputs -> 4 neurons
    HiddenLayer output(4, 1);   // 4 -> 1 output

    double lr = 0.5;
    int epochs = 5000;

    // ===================== TRAINING =====================
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // ----- Forward pass -----
        Matrix Z1 = hidden.forward(X);
        Matrix A1 = Z1.apply(relu);

        Matrix Z2 = output.forward(A1);
        Matrix A2 = Z2.apply(sigmoid);

        // ----- Loss (optional print) -----
        double loss = 0;
        for (int i=0; i<4; ++i) {
            double pred = A2(i+1,1);
            double target = y(i+1,1);
            loss += - (target * std::log(pred + 1e-12) + (1-target)*std::log(1-pred + 1e-12));
        }
        if (epoch % 500 == 0)
            std::cout << "Epoch " << epoch << " Loss: " << loss/4 << "\n";

        // ----- Backward pass (output layer) -----
        Matrix dZ2 = A2 - y;   // gradient for sigmoid+cross-entropy
        Matrix dA1 = output.backward(dZ2, lr);

        // ----- Backward pass (hidden layer, with ReLU) -----
        Matrix relu_mask = Z1.apply(relu_derivative);
        Matrix dZ1 = dA1.hadamard(relu_mask);
        hidden.backward(dZ1, lr);
    }

    // ===================== FINAL PREDICTIONS =====================
    std::cout << "\nFinal predictions:\n";
    Matrix A1_final = hidden.forward(X).apply(relu);
    Matrix A2_final = output.forward(A1_final).apply(sigmoid);
    for (int i=0; i<4; ++i) {
        std::cout << "Input (" << X(i+1,1) << "," << X(i+1,2) << ") -> "
                  << A2_final(i+1,1) << " (target " << y(i+1,1) << ")\n";
    }

    return 0;
}