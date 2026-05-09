#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>

class Matrix {
	private:
		std::vector<std::vector<double>> matrix;
		int rows;
		int columns;
	
	public:
		Matrix();
		Matrix(int rows, int columns, double initVal = 0.0);
		Matrix(const Matrix &other);

		Matrix &operator=(const Matrix &other);
		Matrix operator+(const Matrix &other) const;
		Matrix operator-(const Matrix &other) const;
		Matrix operator*(const Matrix &other) const;
		Matrix operator*(double scale) const;
		double operator()(int rowNum, int colNum) const;
		double &operator()(int rowNum, int colNum);

		void addRows(const std::vector<double> &row);
		void addColumns(const std::vector<double> &column);

		std::vector<double> getRow(int rowNum) const;
		std::vector<double> getColumn(int colNum) const;
		
		size_t getRows() const;
		size_t getColumns() const;

		class DimensionsDiff : public std::exception {
			public:
				virtual const char *what() const throw();
		};

		Matrix transpose() const;
		void randomize(double min = -1.0, double max = 1.0);
		void print() const;	

		Matrix apply(double (*func)(double)) const;
		Matrix hadamard(const Matrix &other) const;
		Matrix softmax() const;
};
