#pragma once

#include <string>
#include <vector>
#include <variant>
#include <optional>

namespace ds
{

const char SPLIT = ',';
const std::string MISSING_VALUE = "?";
const int _UNIQUE = 2;
const double EPS = 1e-9;
const std::string FILLED_MISSING_VALUE = "__MISSING__";
const int MAX_HISTOGRAM_LINE = 50;

enum class FillStrategy { median, mean, zero }; 
enum class ScaleStrategy { min_max, standartisation, robust_scaling };
enum class EncodeStrategy { one_hot, ordinal };

inline int NUMERIC_HISTOGRAM_INTERVALS;
inline FillStrategy FILL_STRATEGY;
inline ScaleStrategy SCALE_STRATEGY;
inline EncodeStrategy ENCODE_STRATEGY;
inline double TRAIN_PART;

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
struct EncodedColumn
{
    std::string name;
    std::vector<double> values;
};


}