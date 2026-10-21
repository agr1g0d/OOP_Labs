#pragma once

#include <iostream>
#include <vector>
#include <unordered_map>
#include "dataset_utility.h"
#include "matrix.h"


namespace ds{

using Column = std::variant<NumericColumn, CategirialColumn>;

class Dataset
{
private:
    std::unordered_map<std::string, Column> columns;
    std::vector<std::string> features_sorted;
    std::vector<Stat> stats;
    int records;
    std::string target_feature;


    void calculate_stats();
    void scale_nums();
    std::vector<EncodedColumn> one_hot() const;
    std::vector<EncodedColumn> ordinal() const;

public:
    Dataset(const std::string& path, 
        const std::string& target_feature = "Class/ASD");
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


