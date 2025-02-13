#include "ND.hpp"
#include <vector>

int main() {
    ND<int> a = ND<int>::arange(0, 24);
    ND<int> b = ND<int>::arange(0, 24);
    a.reshape({6, 1, 4, 1}); 
    b.reshape({6, 4, 1, 1});
    ND<int> z = a + b;
    std::cout << z << "\n";
    for(size_t x : a.strides()) std::cout << x << " ";
}

