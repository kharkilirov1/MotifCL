#include <motifcl/motifcl.hpp>

#include <iostream>

int main() {
    motifcl::Shape shape({2, 3});
    if (shape.numel() != 6) return 1;
    std::cout << "MotifCL consumer smoke passed\n";
    return 0;
}
