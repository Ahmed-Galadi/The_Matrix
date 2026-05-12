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

// ---------- One-hot encoding ----------
Matrix oneHotEncode(int label, int numClasses) {
    Matrix oneHot(1, numClasses, 0.0);
    oneHot(1, label + 1) = 1.0;   // 1‑based indexing
    return oneHot;
}

// ---------- Main ----------
int main() {
    // ===================== LOAD TRAINING DATA =====================
    std::ifstream trainFile("../Datasets/train.csv");
    if (!trainFile) {
        std::cerr << "Cannot open train.csv\n";
        return 1;
    }

    std::string header;
    std::getline(trainFile, header);   // skip header

    std::vector<std::vector<double>> trainImages;
    std::vector<int> trainLabels;
    std::string line;

    while (std::getline(trainFile, line)) {
        std::stringstream ss(line);
        std::string val;
        std::getline(ss, val, ',');
        int label = std::stoi(val);
        trainLabels.push_back(label);

        std::vector<double> pixels;
        while (std::getline(ss, val, ',')) {
            pixels.push_back(std::stod(val) / 255.0);
        }
        trainImages.push_back(pixels);
    }
    trainFile.close();

    int numTrain = trainImages.size();
    int numFeatures = 784;
    int numClasses = 10;

    // Build training matrices X and Y
    Matrix X(numTrain, numFeatures);
    Matrix Y(numTrain, numClasses);
    for (int i = 0; i < numTrain; ++i) {
        for (int j = 0; j < numFeatures; ++j) {
            X(i + 1, j + 1) = trainImages[i][j];
        }
        Matrix oneHotRow = oneHotEncode(trainLabels[i], numClasses);
        for (int j = 0; j < numClasses; ++j) {
            Y(i + 1, j + 1) = oneHotRow(1, j + 1);
        }
    }

    std::cout << "Training samples: " << numTrain << "\n";

    // ===================== LOAD TEST DATA =====================
    std::ifstream testFile("../Datasets/test.csv");
    if (!testFile) {
        std::cerr << "Cannot open test.csv\n";
        return 1;
    }

    std::getline(testFile, header);
    // Check if test set has labels
    bool testHasLabels = (header.find("label") != std::string::npos);

    std::vector<std::vector<double>> testImages;
    std::vector<int> testLabels;   // only used if testHasLabels == true

    while (std::getline(testFile, line)) {
        std::stringstream ss(line);
        std::string val;
        if (testHasLabels) {
            std::getline(ss, val, ',');
            int label = std::stoi(val);
            testLabels.push_back(label);
        }
        std::vector<double> pixels;
        while (std::getline(ss, val, ',')) {
            pixels.push_back(std::stod(val) / 255.0);
        }
        testImages.push_back(pixels);
    }
    testFile.close();

    int numTest = testImages.size();
    Matrix X_test(numTest, numFeatures);
    for (int i = 0; i < numTest; ++i) {
        for (int j = 0; j < numFeatures; ++j) {
            X_test(i + 1, j + 1) = testImages[i][j];
        }
    }

    std::cout << "Test samples: " << numTest;
    if (testHasLabels) std::cout << " (with labels)";
    std::cout << "\n\n";

    // ===================== NETWORK =====================
    HiddenLayer hidden(784, 128);
    HiddenLayer output(128, 10);

    double lr = 0.25;
    int epochs = 200;

    // ===================== TRAINING LOOP =====================
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward pass (train)
        Matrix Z1 = hidden.forward(X);
        Matrix A1 = Z1.apply(relu);
        Matrix Z2 = output.forward(A1);
        Matrix A2 = Z2.softmax();

        // Loss (cross‑entropy)
        double loss = 0.0;
        for (int i = 0; i < numTrain; ++i) {
            for (int j = 0; j < numClasses; ++j) {
                double pred = A2(i + 1, j + 1);
                if (pred < 1e-12) pred = 1e-12;
                loss -= Y(i + 1, j + 1) * std::log(pred);
            }
        }
        loss /= numTrain;

        // Accuracy (train)
        int correct = 0;
        for (int i = 0; i < numTrain; ++i) {
            double maxProb = -1.0;
            int predictedClass = -1;
            for (int j = 0; j < numClasses; ++j) {
                if (A2(i + 1, j + 1) > maxProb) {
                    maxProb = A2(i + 1, j + 1);
                    predictedClass = j;
                }
            }
            if (predictedClass == trainLabels[i]) correct++;
        }
        double acc = 100.0 * correct / numTrain;

        std::cout << "Epoch " << epoch
                  << " | Loss: " << loss
                  << " | Train Acc: " << acc << "%\n";

        // Backward pass (output)
        Matrix dZ2 = A2 - Y;
        dZ2 = dZ2 * (1.0 / numTrain);
        Matrix dA1 = output.backward(dZ2, lr);

        // Backward pass (hidden + ReLU)
        Matrix reluMask = Z1.apply(relu_derivative);
        Matrix dZ1 = dA1.hadamard(reluMask);
        hidden.backward(dZ1, lr);
    }

    std::cout << "Training complete.\n\n";

    // ===================== EVALUATE ON TEST =====================
    Matrix Z1_test = hidden.forward(X_test);
    Matrix A1_test = Z1_test.apply(relu);
    Matrix Z2_test = output.forward(A1_test);
    Matrix A2_test = Z2_test.softmax();

    if (testHasLabels) {
        // Compute accuracy on test set
        int testCorrect = 0;
        for (int i = 0; i < numTest; ++i) {
            double maxProb = -1.0;
            int predictedClass = -1;
            for (int j = 0; j < numClasses; ++j) {
                if (A2_test(i + 1, j + 1) > maxProb) {
                    maxProb = A2_test(i + 1, j + 1);
                    predictedClass = j;
                }
            }
            if (predictedClass == testLabels[i]) testCorrect++;
        }
        double testAcc = 100.0 * testCorrect / numTest;
        std::cout << "Test Accuracy: " << testAcc << "%\n";
    } else {
        std::ofstream subFile("../Datasets/predictions.csv");
        subFile << "ImageId,Label\n";
        for (int i = 0; i < numTest; ++i) {
            double maxProb = -1.0;
            int predictedClass = -1;
            for (int j = 0; j < numClasses; ++j) {
                if (A2_test(i + 1, j + 1) > maxProb) {
                    maxProb = A2_test(i + 1, j + 1);
                    predictedClass = j;
                }
            }
            subFile << i + 1 << "," << predictedClass << "\n";
        }
        subFile.close();
        std::cout << "Predictions saved to predictions.csv\n";
    }

    // Save the hidden layer parameters
    hidden.getWeights().save("../Datasets/hidden_weights.txt");
    hidden.getBiases().save("../Datasets/hidden_biases.txt");

    // Save the output layer parameters
    output.getWeights().save("../Datasets/output_weights.txt");
    output.getBiases().save("../Datasets/output_biases.txt");

std::cout << "Model saved to ../Datasets/hidden_weights.txt, ../Datasets/hidden_biases.txt, ../Datasets/output_weights.txt, ../Datasets/output_biases.txt\n";
    return 0;
}