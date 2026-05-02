#include "Matrix.hpp"

int main() {
    Matrix a(3, 4);
    a.randomize(-2.0, 2.0);
    a.print();

    Matrix b(4, 3);
    b.randomize(-2.0,2.0);
    b.print();

    Matrix c = b * a;
    c.print();
}