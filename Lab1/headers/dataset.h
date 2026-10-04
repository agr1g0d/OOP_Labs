#include <iostream>
#include <variant>
#include <vector>
#include <optional>
#include <unordered_map>

namespace ds{

const char SPLIT = ',';
const std::string MISSING_VALUE = "?";
const int _UNIQUE = 2;

enum TypeColumn
{
    None,
    Numeric,
    Categorial
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

class Dataset
{
private:
    std::unordered_map<std::string, Column> columns;
public:
    Dataset(std::string path);
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


