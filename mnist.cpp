#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include "Matrix.hpp"
#include "HiddenLayer.hpp"
#include "helpers.h"

// ---------- one-hot encoding ----------
Matrix oneHotEncode(int label, int numClasses) {
    Matrix oneHot(1, numClasses, 0.0);
    oneHot(1, label + 1) = 1.0;   // 1‑based indexing; label 0..9
    return oneHot;
}

// ---------- main ----------
int main() {
    // ===================== DATA LOADING =====================
    std::ifstream file("mnist_train.csv");   // MNIST dataset (label, pixel1, pixel2, ..., pixel784)
    if (!file) {
        std::cerr << "Cannot open mnist_train.csv\n";
        return 1;
    }

    std::string header;
    std::getline(file, header);        // skip column names

    std::vector<std::vector<double>> images;
    std::vector<int> labels;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        // first column = label
        std::getline(ss, val, ',');
        int label = std::stoi(val);
        labels.push_back(label);

        std::vector<double> pixels;
        while (std::getline(ss, val, ',')) {
            pixels.push_back(std::stod(val) / 255.0);   // normalize to [0,1]
        }
        images.push_back(pixels);
    }
    file.close();

    int numSamples = images.size();
    int numFeatures = 784;
    int numClasses = 10;

    // Build data matrix X and one-hot encoded y
    Matrix X(numSamples, numFeatures);
    Matrix Y(numSamples, numClasses);   // will be one-hot

    for (int i = 0; i < numSamples; ++i) {
        for (int j = 0; j < numFeatures; ++j) {
            X(i + 1, j + 1) = images[i][j];   // 1‑based indexing
        }
        Matrix oneHotRow = oneHotEncode(labels[i], numClasses);
        for (int j = 0; j < numClasses; ++j) {
            Y(i + 1, j + 1) = oneHotRow(1, j + 1);
        }
    }

    std::cout << "Loaded " << numSamples << " training examples.\n";

    // ===================== NETWORK ARCHITECTURE =====================
    HiddenLayer hidden(784, 128);   // 784 -> 128
    HiddenLayer output(128, 10);    // 128 -> 10

    double learningRate = 0.5;      // works well for full-batch with normalized data
    int epochs = 20;                // 20 passes through the whole dataset

    // ===================== TRAINING =====================
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // ----- Forward pass -----
        Matrix Z1 = hidden.forward(X);
        Matrix A1 = Z1.apply(relu);          // ReLU activation

        Matrix Z2 = output.forward(A1);
        Matrix A2 = Z2.softmax();            // probabilities (numSamples x 10)

        // ----- Loss (cross-entropy) -----
        double loss = 0.0;
        for (int i = 0; i < numSamples; ++i) {
            for (int j = 0; j < numClasses; ++j) {
                double pred = A2(i+1, j+1);
                if (pred < 1e-12) pred = 1e-12;   // avoid log(0)
                loss -= Y(i+1, j+1) * std::log(pred);
            }
        }
        loss /= numSamples;

        // ----- Accuracy (on training data, for monitoring) -----
        int correct = 0;
        for (int i = 0; i < numSamples; ++i) {
            double maxProb = -1.0;
            int predictedClass = -1;
            for (int j = 0; j < numClasses; ++j) {
                if (A2(i+1, j+1) > maxProb) {
                    maxProb = A2(i+1, j+1);
                    predictedClass = j;
                }
            }
            if (predictedClass == labels[i]) correct++;
        }
        double accuracy = (double)correct / numSamples * 100.0;

        std::cout << "Epoch " << epoch
                  << " | Loss: " << loss
                  << " | Accuracy: " << accuracy << "%\n";

        // ----- Backward pass (output layer) -----
        Matrix dZ2 = A2 - Y;            // gradient of loss w.r.t. Z2 (softmax + cross‑entropy)
        Matrix dA1 = output.backward(dZ2, learningRate);

        // ----- Backward pass (hidden layer, with ReLU) -----
        Matrix reluMask = Z1.apply(relu_derivative);   // 1 where Z1 > 0, else 0
        Matrix dZ1 = dA1.hadamard(reluMask);
        hidden.backward(dZ1, learningRate);
    }

    std::cout << "\nTraining complete.\n";

    // ===================== OPTIONAL: test on a few examples =====================
    // Print predictions for first 10 samples
    Matrix Z1_test = hidden.forward(X).apply(relu);
    Matrix A2_test = output.forward(Z1_test).softmax();
    std::cout << "\nSample predictions (first 10):\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "True: " << labels[i] << " Predicted: ";
        double maxProb = -1.0;
        int pred = -1;
        for (int j = 0; j < numClasses; ++j) {
            if (A2_test(i+1, j+1) > maxProb) {
                maxProb = A2_test(i+1, j+1);
                pred = j;
            }
        }
        std::cout << pred << " (prob " << maxProb << ")\n";
    }

    return 0;
}