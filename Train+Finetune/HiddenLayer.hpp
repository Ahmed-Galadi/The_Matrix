#pragma ones
#include "Matrix.hpp"

class HiddenLayer {
    private:
        int inputSize;
        int outputSize;
        Matrix weights;
        Matrix biases;
        Matrix lastInput;
    
    public:
        HiddenLayer(int inputSize, int outputsize);

        void setWeights(const Matrix &w);
        void setBiases(const Matrix &b);
        Matrix getWeights() const;
        Matrix getBiases() const;

        Matrix forward(const Matrix &input);  
        Matrix backward(const Matrix &upstreamGrad, double learningRate);
};