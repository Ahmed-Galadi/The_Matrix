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

// // Activation functions (same as before)
// double relu(double x) { return x > 0 ? x : 0; }
// double relu_derivative(double x) { return x > 0 ? 1.0 : 0.0; }

// One‑hot encoding (same as before)
Matrix oneHotEncode(int label, int numClasses) {
    Matrix oneHot(1, numClasses, 0.0);
    oneHot(1, label + 1) = 1.0;
    return oneHot;
}

// Helper: load a CSV file (label, pixel1, pixel2, ..., pixel784)
// Pixels are assumed to be already normalized to [0,1] (like mydata.csv)
// or raw 0-255 (like mnist_train.csv). We provide a `normalize` flag.
void loadCSV(const std::string &filename,
             std::vector<std::vector<double>> &images,
             std::vector<int> &labels,
             bool normalize = false)
{
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Warning: Could not open " << filename << " – skipping.\n";
        return;
    }

    std::string header;
    std::getline(file, header);   // skip header (if any)

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        std::getline(ss, val, ',');
        int label = std::stoi(val);
        labels.push_back(label);

        std::vector<double> pixels;
        while (std::getline(ss, val, ',')) {
            double pixel = std::stod(val);
            if (normalize) pixel /= 255.0;   // scale MNIST pixels
            pixels.push_back(pixel);
        }
        if (pixels.size() == 784)
            images.push_back(pixels);
    }
    file.close();
}

int main() {
    // ===================== 1. Load pre-trained parameters =====================
    std::cout << "Loading pre-trained weights...\n";
    HiddenLayer hidden(784, 128);
    HiddenLayer output(128, 10);

    hidden.setWeights(Matrix::load("hidden_weights.txt"));
    hidden.setBiases(Matrix::load("hidden_biases.txt"));
    output.setWeights(Matrix::load("output_weights.txt"));
    output.setBiases(Matrix::load("output_biases.txt"));
    std::cout << "Pre-trained model loaded.\n\n";

    // ===================== 2. Load personalised data =====================
    std::vector<std::vector<double>> personalImages;
    std::vector<int> personalLabels;
    loadCSV("mydata.csv", personalImages, personalLabels, false);  // already normalized
    int numPersonal = personalImages.size();
    std::cout << "Personal samples loaded: " << numPersonal << "\n";

    if (numPersonal == 0) {
        std::cerr << "No personal data found (mydata.csv missing or empty). Exiting.\n";
        return 1;
    }

    // ===================== 3. (Optional) Load a small MNIST subset =====================
    std::vector<std::vector<double>> mnistImages;
    std::vector<int> mnistLabels;
    loadCSV("mnist_subset.csv", mnistImages, mnistLabels, true);  // needs normalization
    int numMnist = mnistImages.size();
    std::cout << "MNIST subset loaded: " << numMnist << " (optional)\n\n";

    // ===================== 4. Combine datasets =====================
    int totalSamples = numPersonal + numMnist;
    int numFeatures = 784;
    int numClasses = 10;

    Matrix X(totalSamples, numFeatures);
    Matrix Y(totalSamples, numClasses);
    int row = 1;   // 1‑based indexing

    // Copy personal data
    for (int i = 0; i < numPersonal; ++i) {
        for (int j = 0; j < numFeatures; ++j)
            X(row, j+1) = personalImages[i][j];
        Matrix oneHot = oneHotEncode(personalLabels[i], numClasses);
        for (int j = 0; j < numClasses; ++j)
            Y(row, j+1) = oneHot(1, j+1);
        ++row;
    }

    // Copy MNIST subset
    for (int i = 0; i < numMnist; ++i) {
        for (int j = 0; j < numFeatures; ++j)
            X(row, j+1) = mnistImages[i][j];
        Matrix oneHot = oneHotEncode(mnistLabels[i], numClasses);
        for (int j = 0; j < numClasses; ++j)
            Y(row, j+1) = oneHot(1, j+1);
        ++row;
    }

    // ===================== 5. Fine‑tuning loop =====================
    double lr = 0.001;        // small learning rate
    int epochs = 15;

    std::cout << "Starting fine-tuning (lr=" << lr << ", epochs=" << epochs << ")...\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        Matrix Z1 = hidden.forward(X);
        Matrix A1 = Z1.apply(relu);
        Matrix Z2 = output.forward(A1);
        Matrix A2 = Z2.softmax();

        // Loss (cross‑entropy)
        double loss = 0.0;
        for (int i = 0; i < totalSamples; ++i) {
            for (int j = 0; j < numClasses; ++j) {
                double pred = A2(i+1, j+1);
                if (pred < 1e-12) pred = 1e-12;
                loss -= Y(i+1, j+1) * std::log(pred);
            }
        }
        loss /= totalSamples;

        // Accuracy (on the combined training set)
        int correct = 0;
        std::vector<int> allLabels;
        allLabels.insert(allLabels.end(), personalLabels.begin(), personalLabels.end());
        allLabels.insert(allLabels.end(), mnistLabels.begin(), mnistLabels.end());

        for (int i = 0; i < totalSamples; ++i) {
            double maxProb = -1.0;
            int predicted = -1;
            for (int j = 0; j < numClasses; ++j) {
                if (A2(i+1, j+1) > maxProb) {
                    maxProb = A2(i+1, j+1);
                    predicted = j;
                }
            }
            if (predicted == allLabels[i]) ++correct;
        }
        double acc = 100.0 * correct / totalSamples;

        std::cout << "Epoch " << epoch << " | Loss: " << loss
                  << " | Train Acc: " << acc << "%\n";

        // Backward pass
        Matrix dZ2 = A2 - Y;
        dZ2 = dZ2 * (1.0 / totalSamples);   // average gradient
        Matrix dA1 = output.backward(dZ2, lr);

        Matrix reluMask = Z1.apply(relu_derivative);
        Matrix dZ1 = dA1.hadamard(reluMask);
        hidden.backward(dZ1, lr);
    }

    // ===================== 6. Save updated parameters =====================
    hidden.getWeights().save("hidden_weights.txt");
    hidden.getBiases().save("hidden_biases.txt");
    output.getWeights().save("output_weights.txt");
    output.getBiases().save("output_biases.txt");

    std::cout << "\nFine-tuned weights saved. Replace them in your Java project.\n";
    return 0;
}