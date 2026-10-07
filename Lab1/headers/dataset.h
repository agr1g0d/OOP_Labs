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
const int NUMERIC_HISTOGRAM_INTERVALS = 8;
const int MAX_HISTOGRAM_LINE = 50; 

struct NumericColumn
{
    std::vector<std::optional<double>> values;
    int missing;
    NumericColumn() : missing(0) {}
};

struct CategirialColumn
{
    std::vector<std::optional<std::string>> values;
    int missing;
    CategirialColumn() : missing(0) {}
};

using Column = std::variant<NumericColumn, CategirialColumn>;

class Dataset
{
private:
    std::unordered_map<std::string, Column> columns;
    std::vector<std::string> features_sorted;
    int records;
public:
    Dataset(const std::string& path);
    void print_info() const;
    void print_stat_by_i(int i) const;
    const Column& operator[](const std::string& s) const;
    const Column& operator[](int i) const;
    inline int size() const { return  columns.size(); }

private:
    struct ColumnInfo
    {
        std::string name;
        bool is_num;
        int unique;
        std::vector<double> unique_vals;

        ColumnInfo() : is_num(1), unique(0) 
        { unique_vals.reserve(2); }
    };
};



}


