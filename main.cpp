#include "Matrix.hpp"
#include <fstream>

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
    double maxMilage = 240000.0; // i will get the max from milages later (TODO)
    for (int i = 0; i < m; i++) {
        // row in the matrix is i+1
        X(i + 1, 1) = 1.0; // bias column
        X(i + 1, 2) = milages[i] / maxMilage; // milag column

        y(i + 1, 1) = prices[i]; // price
    }


    Matrix theta(2, 1);
    double lr = 0.00000001;//very small learning rate (i will adjust it later)
    int iterations = 1000;

    for (int iter = 0; iter < iterations; ++iter) {
        Matrix pred   = X * theta;
        Matrix error  = pred - y;
        Matrix grad   = X.transpose() * error;
        grad = grad * (1.0 / m);// average
        theta = theta - (grad * lr);// update
    }

    std::cout << "this is θ₀ -> " << theta(1,1) << "\n";
    std::cout << "this is θ₁ -> " << theta(2,1) << "\n";
}