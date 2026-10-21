#pragma once

#include <vector>
#include <string>
#include <span>
#include "dataset_utility.h"

namespace ds
{

struct Matrix
{
    std::size_t n_rows;
    std::size_t n_cols;
    std::vector<std::string> cols;
    std::vector<double> data;

    std::span<const double> operator[](std::size_t i) const;
    std::span<double> operator[](std::size_t i);
    const double& at(std::size_t i, std::size_t j) const;
    double& at(std::size_t i, std::size_t j);
};
Matrix load_binary_matrix(const std::string& path);
Matrix build_matrix(std::vector<EncodedColumn> encoded_data, std::size_t rows);

}



