#include "Matrix.hpp"
#include <random>
#include <iomanip>
#include <string>
#include <cmath>
#include <fstream>


Matrix::Matrix() : rows(0), columns(0) {}

Matrix::Matrix(int rows, int columns, double initVal) : rows(rows), columns(columns) {
	matrix.resize(rows, std::vector<double>(columns, initVal));
}

Matrix::Matrix(const Matrix &other) {
	*this = other;
}

Matrix &Matrix::operator=(const Matrix &other) {
	if (this != &other) {
		matrix = other.matrix;
		rows = other.rows;
		columns = other.columns;
	}
	return (*this);
}

Matrix Matrix::operator+(const Matrix &other) const {
    if (columns != other.columns || rows != other.rows)
        throw DimensionsDiff();
    Matrix result(rows, columns);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < columns; j++)
            result.matrix[i][j] = matrix[i][j] + other.matrix[i][j];
    return (result);
}

Matrix Matrix::operator-(const Matrix &other) const {
    if (columns != other.columns || rows != other.rows)
        throw DimensionsDiff();
    Matrix result(rows, columns);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < columns; j++)
            result.matrix[i][j] = matrix[i][j] - other.matrix[i][j];
    return (result);
}

Matrix Matrix::operator*(const Matrix &other) const {
	if (columns != other.rows)
        throw DimensionsDiff();
	Matrix result(rows, other.columns, 0.0);
	// Triple loop multiplication
    for (int i = 0; i < rows; ++i) {                // for each row of A
        for (int j = 0; j < other.columns; j++) {   // for each column of B
            double sum = 0.0;
            for (int k = 0; k < columns; k++) {     // common dimension (columns of A = rows of B)
                sum += matrix[i][k] * other.matrix[k][j];
            }
            result.matrix[i][j] = sum; // store result
        }
    }

    return (result);
}


Matrix Matrix::operator*(double scale) const {
	Matrix result(rows, columns);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < columns; j++)
			result.matrix[i][j] = matrix[i][j] * scale;
	return (result);
}

double Matrix::operator()(int rowNum, int colNum) const{
	if (rowNum < 1 || colNum < 1 
		|| rowNum > rows || colNum > columns)
		throw (std::out_of_range("Matrix index out of range"));
	return (matrix[rowNum-1][colNum-1]);
}

double &Matrix::operator()(int rowNum, int colNum) {
	if (rowNum < 1 || colNum < 1 
		|| rowNum > rows || colNum > columns)
		throw (std::out_of_range("Matrix index out of range"));
	return (matrix[rowNum-1][colNum-1]);
}

Matrix Matrix::load(const std::string &filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Cannot open " + filename);
    int r, c;
    in >> r >> c;
    Matrix mat(r, c, 0.0);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            in >> mat(i+1, j+1);
    return mat;
}


size_t Matrix::getRows() const {
    return (rows);
}

size_t Matrix::getColumns() const {
    return (columns);
}

Matrix Matrix::transpose() const {
    if (rows == 0 && columns == 0)
        return Matrix();
    Matrix result(columns, rows);
    for (int i = 0; i < rows; i++) 
        for (int j = 0; j < columns; j++) 
            result.matrix[j][i] = matrix[i][j];
    return (result);
}

Matrix Matrix::hadamard(const Matrix &other) const {
    if (rows != other.rows || columns != other.columns)
        throw (DimensionsDiff());
    Matrix results(rows, columns);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < columns; j++)
            results.matrix[i][j] = matrix[i][j] * other.matrix[i][j];
    return (results);
}

Matrix Matrix::softmax() const {
    Matrix result(rows, columns);   // same size, filled with 0.0

    for (int i = 0; i < rows; ++i) {
        double maxVal = matrix[i][0];
        for (int j = 1; j < columns; ++j) {
            if (matrix[i][j] > maxVal)
                maxVal = matrix[i][j];
        }
        double sumExp = 0.0;
        for (int j = 0; j < columns; ++j) {
            double shifted = matrix[i][j] - maxVal;
            result.matrix[i][j] = std::exp(shifted);   // temporarily store exp
            sumExp += result.matrix[i][j];
        }
        for (int j = 0; j < columns; ++j)
            result.matrix[i][j] /= sumExp;
    }
    return (result);
}

void Matrix::randomize(double min, double max) {
	static std::mt19937 rng(std::random_device{}());
	std::uniform_real_distribution<double> dist(min, max);
	for (int i = 0; i < rows; i++) 
        for (int j = 0; j < columns; j++) 
            matrix[i][j] = dist(rng);
}

// void Matrix::print() const {
// 	for (int i = 0 ; i < rows; i++) {
// 		for (int j = 0; j < columns; j++)
// 			std::cout << matrix[i][j] << ",";
// 		std::cout << "\n";
// 	}
// 	std::cout << std::endl; 
// }

void Matrix::print() const {
    if (rows == 0 || columns == 0) {
        std::cout << "(empty matrix)\n";
        return;
    }

    // ----- Calculate column widths -----
    std::vector<int> colWidths(columns, 0);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            // Convert number to string with enough precision (adjust as needed)
            std::ostringstream oss;
            oss << std::setprecision(6) << std::fixed << matrix[i][j];
            int length = oss.str().length();
            if (length > colWidths[j])
                colWidths[j] = length;
        }
    }

    // ----- Helper for repeated spaces -----
    auto spaces = [](int n) { return std::string(n, ' '); };

    // ----- Print top border -----
    // Compute total inner width: sum of column widths + (columns-1) spaces between columns
    int innerWidth = 0;
    for (int w : colWidths) innerWidth += w;
    if (columns > 0) innerWidth += (columns - 1);   // spaces between columns

    std::cout << "┌" << spaces(innerWidth) << "┐\n";

    // ----- Print each row -----
    for (int i = 0; i < rows; ++i) {
        std::cout << "│";
        for (int j = 0; j < columns; ++j) {
            // Print number right-aligned into its column width
            std::cout << std::setw(colWidths[j]) << std::right;
            // one decimal place? The user used %g-like format earlier, but we can keep flexible.
            // here we use fixed with precision from the calculation.
            std::ostringstream oss;
            oss << std::setprecision(6) << std::fixed << matrix[i][j];
            std::cout << oss.str();
            if (j < columns - 1)
                std::cout << " ";
        }
        std::cout << "│\n";
    }

    // ----- Print bottom border -----
    std::cout << "└" << spaces(innerWidth) << "┘\n";
}

Matrix Matrix::apply(double (*func)(double)) const {
    Matrix result(rows, columns);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < columns; j++)
            result.matrix[i][j] = func(matrix[i][j]);
    return (result);
}

const char *Matrix::DimensionsDiff::what() const throw() {
	return ("Error: Matrix Dimensions Difference -> operation cant be completed!");
}



void Matrix::save(const std::string &filename) const {
    std::ofstream out(filename);
    out << rows << " " << columns << "\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            out << matrix[i][j];
            if (j < columns - 1) out << " ";
        }
        out << "\n";
    }
    out.close();
}