#include "HiddenLayer.hpp"

HiddenLayer::HiddenLayer(int inputSize, int outputSize) : inputSize(inputSize), outputSize(outputSize), weights(inputSize, outputSize), biases(1, outputSize, 0.0) {
    weights.randomize(-0.5, 0.5);
}

Matrix HiddenLayer::getWeights() const {
    return (weights);
}

Matrix HiddenLayer::getBiases() const {
    return (biases);
}

void HiddenLayer::setBiases(const Matrix &b) {
    biases = b;
}

void HiddenLayer::setWeights(const Matrix &w) {
    weights = w;
}

Matrix HiddenLayer::forward(const Matrix &input) {
    if (input.getColumns() != inputSize)
        throw (Matrix::DimensionsDiff());
    Matrix result = input * weights;
    for (int i = 0; i < result.getRows(); i++)
        for (int j = 0; j < result.getColumns(); j++)
            result(i+1, j+1) = result(i+1, j+1) + biases(1, j+1);
    lastInput = input;
    return (result);
}

Matrix HiddenLayer::backward(const Matrix &upstreamGrad, double learningRate) {
    Matrix inputGrad = upstreamGrad * weights.transpose();
    Matrix weightGrad = lastInput.transpose() * upstreamGrad;

    int batchSize = upstreamGrad.getRows();
    int numNeurons = outputSize;
    Matrix biasGrad(1, numNeurons, 0.0);
    for (int i = 0; i < batchSize; i++)
        for (int j = 0; j < numNeurons; j++) 
            biasGrad(1, j+1) = biasGrad(1, j+1) + upstreamGrad(i+1, j+1);
    weights = weights - (weightGrad * learningRate);
    biases  = biases  - (biasGrad * learningRate);
    return (inputGrad);
}
