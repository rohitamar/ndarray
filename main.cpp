#include "ND.hpp"
#include <chrono>
#include <vector>

using namespace std::chrono;

int main() {
    ND<int> a = ND<int>::arange(0, 1000000000);
    ND<int> b = ND<int>::arange(0,        500);
    a.reshape({50, 20, 10, 1000, 10, 10});
    b.reshape({50,  1, 10,    1,  1,  1});

    // ND<int> a = ND<int>::arange(0, 300);
    // ND<int> b = ND<int>::arange(0, 3);
    // a.reshape({100, 3});
    // b.reshape({1, 3});

    auto start = steady_clock::now();
    ND<int> c = a + b;
    auto end = steady_clock::now();
    auto ms = duration_cast<milliseconds>(end - start).count();
    std::cout << ms << "\n";
}

