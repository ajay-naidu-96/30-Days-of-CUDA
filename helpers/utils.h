#include <iostream>
using namespace std;

template<typename T>
void print(const T* data, size_t rows, size_t cols=1, const std::string &name="") {

    std::cout << "Matrix : " << name << endl;

    for (size_t r=0; r < rows; r++) {
        std::cout << " [ " ;
        for (size_t c=0; c < cols; c++) {
            cout << data[r*cols+c] ;
            if (c+1 < cols) {
                cout << ", ";
            }

        }
        std::cout << " ]" << endl;
    }
}
