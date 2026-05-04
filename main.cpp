#include "Matrix.hpp"
#include <fstream>
#include <algorithm>
#include "helpers.h"

int main() {
    std::ifstream file("data.csv");
    if (!file) {
        std::cerr << "Cannot open file\n";
        return 1;
    }
    std::string garbageHeader;
    std::getline(file, garbageHeader);

    std::vector<double> milages;
    std::vector<double> prices;
    double km, price;
    char garbageComma;

    while (file >> km >> garbageComma >> price) {
        milages.push_back(km);
        prices.push_back(price);
    }
    file.close();

    // m is the size of the vectors
    int m = milages.size();

    // Matrixes
    Matrix X(m, 2);
    Matrix y(m,1);

    double maxMilage = *std::max_element(milages.begin(), milages.end());
    for (int i = 0; i < m; i++) {
        // row in the matrix is i+1
        X(i + 1, 1) = 1.0; // bias column
        X(i + 1, 2) = milages[i] / maxMilage; // milag column

        y(i + 1, 1) = prices[i]; // price
    }

    // -------- lenear regression ------------
    Matrix theta_linear(2, 1);
    double lr = 0.01;
    int iterations = 100000;
    for (int iter = 0; iter < iterations; iter++) {
        Matrix pred   = X * theta_linear;
        Matrix error  = pred - y;
        Matrix grad   = X.transpose() * error;
        grad = grad * (1.0 / m);// average
        theta_linear = theta_linear - (grad * lr);// update
        if (iter % 100 == 0) {
            double mse = 0;
            for (int i = 0; i < m; i++)
                mse += error(i+1, 1) * error(i+1, 1);
            std::cout << "MSE at iteration " << iter << ": " << mse/m << "\n";
        }
    }

    std::cout << "this is θ₀ -> " << theta_linear(1,1) << "\n";
    std::cout << "this is θ₁ -> " << theta_linear(2,1) << "\n";


    // -------------- logistic regression ----------
    // compute average price
    double sum = 0;
    for (int i = 0; i < m; ++i)
        sum += prices[i];
    double avgPrice = sum / m;

    // create binary labels
    Matrix y_binary(m, 1);
    for (int i = 0; i < m; ++i) {
        if (prices[i] > avgPrice)
            y_binary(i+1, 1) = 1.0;
        else
            y_binary(i+1, 1) = 0.0;
    }

    Matrix theta_logistic(2, 1);
    for (int iter = 0; iter < iterations; iter++) {
        Matrix z = X * theta_logistic;
        Matrix pred = z.apply(sigmoid);
        Matrix error = pred - y_binary;
        Matrix grad = X.transpose() * error;
        grad = grad * (1.0 / m);
        theta_logistic = theta_logistic - (grad * lr);
    }
    
    for (int testIndex = 0; testIndex < m; ++testIndex) {
        double testMileage = milages[testIndex] / maxMilage;
        double z = theta_logistic(1,1) + theta_logistic(2,1) * testMileage;
        double prob = sigmoid(z);
        int predicted = (prob >= 0.5) ? 1 : 0;
        int actual = (prices[testIndex] > avgPrice) ? 1 : 0;
        std::cout << "Car " << testIndex 
                << " | mileage: " << milages[testIndex]
                << " | prob: " << prob
                << " | predicted: " << predicted
                << " | actual: " << actual
                << " | " << (predicted == actual ? "OK" : "WRONG") << "\n";
    }
    std::ofstream out("output_data.csv");
    out << "mileage,price,linear_pred,prob_expensive,actual_label\n";

    for (int i = 0; i < m; ++i) {
        double scaledMileage = milages[i] / maxMilage;
    
        // Linear regression prediction
        double linearPred = theta_linear(1,1) + theta_linear(2,1) * scaledMileage;
    
        // Logistic regression prediction (probability)
        double zLog = theta_logistic(1,1) + theta_logistic(2,1) * scaledMileage;
        double probExpensive = sigmoid(zLog);
    
        // Actual label (1 = expensive, 0 = cheap)
        int actualLabel = (prices[i] > avgPrice) ? 1 : 0;
    
        out << milages[i] << ","
            << prices[i] << ","
            << linearPred << ","
            << probExpensive << ","
            << actualLabel << "\n";
    }
    out.close();
    std::cout << "Data written to output_data.csv\n";
}