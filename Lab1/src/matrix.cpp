#include <fstream>
#include <limits>
#include "matrix.h"
#include "dataset.h"
#include "exc.h"

namespace ds
{

Matrix build_matrix(std::vector<EncodedColumn> encoded_data, std::size_t rows)
{
    Matrix m = {rows, encoded_data.size(), 
        std::vector<std::string>(encoded_data.size()), 
        std::vector<double>(encoded_data.size() * rows)};
    for (int j = 0; j < m.n_cols; ++j)
    {
        m.cols[j] = encoded_data[j].name;
        for (int i = 0; i < rows; ++i)
            m.data[i*m.n_cols + j] = encoded_data[j].values[i];
    }
    return m;
}

Matrix load_binary_matrix(const std::string& path)
{
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in) throw std::runtime_error("binary input file not found.");

    Matrix m;

    in.read(reinterpret_cast<char*>(&m.n_rows), sizeof(std::size_t));
    in.read(reinterpret_cast<char*>(&m.n_cols), sizeof(std::size_t));
    if (in.fail())
        throw std::runtime_error("binary reading error.");
    if (m.n_cols != 0 &&
        m.n_rows > std::numeric_limits<std::size_t>::max() / m.n_cols)
        throw std::runtime_error("binary matrix dimensions are too large.");
    m.cols.resize(m.n_cols);
    m.data.resize(m.n_rows * m.n_cols);
    for (auto& name : m.cols)
    {
        std::size_t len;
        in.read(reinterpret_cast<char*>(&len), sizeof(std::size_t));
        if (in.fail())
            throw std::runtime_error("binary reading error.");
        name.resize(len);
        in.read(name.data(), len);
        if (in.fail())
            throw std::runtime_error("binary reading error.");
    }
    in.read(reinterpret_cast<char*>(m.data.data()), m.n_rows*m.n_cols*sizeof(double));
    if (in.fail()) throw std::runtime_error("binary reading error.");

    return m;
}

std::span<const double> Matrix::operator[](std::size_t i) const
{   return std::span<const double>(data.data() + i*n_cols, n_cols); }

std::span<double> Matrix::operator[](std::size_t i)
{   return std::span<double>(data.data() + i*n_cols, n_cols); }

const double& Matrix::at(std::size_t i, std::size_t j) const
{
    if (i < 0 || i >= n_rows ||
        j < 0 || j >= n_cols) throw exc::out_of_matrix("out of matrix bounds.");
    return data[i*n_cols + j];
}

double& Matrix::at(std::size_t i, std::size_t j)
{
    if (i < 0 || i >= n_rows ||
        j < 0 || j >= n_cols) throw exc::out_of_matrix("out of matrix bounds.");
    return data[i*n_cols + j];
}


}