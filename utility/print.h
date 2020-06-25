#include <string>
#include <iostream>

#include <Eigen/Eigen>

void print_vector(const Eigen::VectorXd &vec)
{
    std::cout << "[";
    for (int i{0}; i < vec.size(); ++i)
    {
        std::cout << vec(i);
        if (i != vec.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]\n";
}
