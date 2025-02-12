#include "ND.hpp"
#include <vector>

int main() {
    ND<int> a = ND<int>::ones({1, 3});
    ND<int> b = ND<int>::ones({3, 1});
    std::cout << a + b << "\n";
}

