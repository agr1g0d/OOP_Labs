#pragma once

#include <iostream>
#include <variant>
#include <vector>
#include <optional>
#include <unordered_map>

namespace ds{

const char SPLIT = ',';
const std::string MISSING_VALUE = "?";
const int _UNIQUE = 2;
const double EPS = 1e-9;
const std::string FILLED_MISSING_VALUE = "__MISSING__";
const int MAX_HISTOGRAM_LINE = 50;
inline int NUMERIC_HISTOGRAM_INTERVALS = 10;

enum class FillStrategy { median, mean, zero }; 
enum class ScaleStrategy { min_max, standartisation, robust_scaling };
enum class EncodeStrategy { one_hot, ordinal };

inline FillStrategy FILL_STRATEGY;
inline ScaleStrategy SCALE_STRATEGY;
inline EncodeStrategy ENCODE_STRATEGY;

struct NumStat
{
    double min;
    double max;
    double mean;
    double var;
    double stand;
    double q[5];
};
struct CatStat
{
    std::vector<std::pair<std::string, int>> categories;
};
struct Stat
{
    int missing;
    std::variant<NumStat, CatStat> stat;
};
struct NumericColumn
{
    std::vector<std::optional<double>> values;
};
struct CategirialColumn
{
    std::vector<std::optional<std::string>> values;
};
using Column = std::variant<NumericColumn, CategirialColumn>;
struct EncodedColumn
{
    std::string name;
    std::vector<double> values;
};
struct Matrix
{
    std::size_t n_rows;
    std::size_t n_cols;
    std::vector<std::string> cols;
    std::vector<double> data;
};
Matrix load_binary_matrix(const std::string& path);


class Dataset
{
private:
    std::unordered_map<std::string, Column> columns;
    std::vector<std::string> features_sorted;
    std::vector<Stat> stats;
    int records;

    Matrix build_matrix(std::vector<EncodedColumn> encoded_data, std::size_t rows);
    void calculate_stats();
    void scale_nums();
    std::vector<EncodedColumn> one_hot() const;
    std::vector<EncodedColumn> ordinal() const;

public:
    Dataset();
    Dataset(const std::string& path);
    void load_csv(const std::string& path);
    const Column& operator[](const std::string& s) const;
    const Column& operator[](int i) const;
    void print_info() const;
    void print_stat_by_i(int i) const;
    void fill_missing();
    inline int size() const { return  columns.size(); }
    void save_binary_matrix(const std::string& path);

private:
    struct ColumnInfo
    {
        std::string name;
        bool is_num;
        int unique;
        std::vector<double> unique_vals;
        int missing;
        ColumnInfo() : is_num(1), unique(0), missing(0) 
        { unique_vals.reserve(2); }
    };
};



}


