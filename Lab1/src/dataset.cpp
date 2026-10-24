#include "dataset.h"
#include <iostream>
#include <fstream>
#include <exception>
#include <string_view>
#include <charconv>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <tuple>
#include <vector>
#include <cmath>
#include <limits>

namespace ds{

inline void pop_r(std::string& s)
{
    if (!s.empty() && s.back() == '\r')
        s.pop_back();
}

Dataset::Dataset(const std::string& path, 
    const std::string& target_feature)
        : target_feature(target_feature)
{
    load_csv(path);
    for (const std::string& s : features_sorted)
        if (s == target_feature) return;
    throw std::invalid_argument("dataset has no given target feature.");
}

void Dataset::load_csv(const std::string& path)
{
    if (!columns.empty()) columns.clear();
    if (!features_sorted.empty()) features_sorted.clear();
    if (!stats.empty()) stats.clear();
    

    std::ifstream file(path, std::ios::in);
    if (!file.is_open()) throw std::runtime_error("error while opening file: " + path);
    std::vector<Dataset::ColumnInfo> header;
    int features;

    std::string header_line;
    if (!std::getline(file, header_line) || header_line.empty())
        throw std::runtime_error("CSV is empty\n");
    pop_r(header_line);
    if (header_line.empty())
        throw std::runtime_error("CSV is empty\n");
    for (char c : header_line)
        if (c == SPLIT)
            header.push_back(Dataset::ColumnInfo());
    header.push_back(Dataset::ColumnInfo());
    features = header.size();
    columns.reserve(features);
    stats.reserve(features);


    std::string s;
    while (std::getline(file, s))         //check column types;
    {
        if (s.empty()) break;
        pop_r(s);
        std::string_view sv(s);

        const char* p = sv.begin(), *pprev = sv.begin();
        int i = 0;
        for (; p != sv.end(); ++p)
        {
            if (*p == SPLIT)
            {
                if (i >= features) throw std::runtime_error("amount of features more than in the header\n");
                double d;
                auto [ptr, ec] = std::from_chars(pprev, p, d);
                if (ec==std::errc{} && ptr==p)
                {
                    if (header[i].is_num)
                    {
                        if (header[i].unique <= _UNIQUE 
                            && std::find(header[i].unique_vals.begin(), 
                            header[i].unique_vals.end(),
                            d) == header[i].unique_vals.end())
                        {
                            header[i].unique++;
                            header[i].unique_vals.push_back(d);
                        }
                    }   
                } else if (s.substr((pprev - sv.begin()), (p-pprev)) != MISSING_VALUE)
                {
                    header[i].is_num = false;
                } 
                pprev = p; ++pprev;
                ++i;
            }
        }
        double d;
        auto [ptr, ec] = std::from_chars(pprev, p, d);
        if (ec==std::errc{} && ptr==p)
        {
            if (header[i].is_num)
            {
                if (header[i].unique <= _UNIQUE 
                    && std::find(header[i].unique_vals.begin(), 
                    header[i].unique_vals.end(),
                    d) == header[i].unique_vals.end())
                {
                    header[i].unique++;
                    header[i].unique_vals.push_back(d);
                }
            }   
        } else if (s.substr((pprev - sv.begin()), (p-pprev)) != MISSING_VALUE)
        {
            header[i].is_num = false;
        } 
        ++i;
        if (i != features) throw std::runtime_error("amount of features less than in the header");
    }

    file.clear();
    file.seekg(0);
    for (int i = 0; i < features; ++i)
    {
        char last = (i == features-1) ? '\n' : SPLIT;
        std::getline(file, header[i].name, last);
        pop_r(header[i].name);
        if (header[i].is_num && header[i].unique > _UNIQUE)
        {
            columns[header[i].name] = NumericColumn();
        }
        else
        {
            columns[header[i].name] = CategirialColumn();
        }
    }        
    while (!file.eof())
    {
        for (int i = 0; i < features; ++i)
        {
            char last = (i == features-1) ? '\n' : SPLIT;
            std::getline(file, s, last);
            if  (s.empty()) break;
            pop_r(s);
            if (std::holds_alternative<NumericColumn>(columns[header[i].name]))
            {
                if (s==MISSING_VALUE)
                {
                    std::get<NumericColumn>(columns[header[i].name]).values.push_back(std::nullopt);
                    header[i].missing++;
                }
                else                  
                    std::get<NumericColumn>(columns[header[i].name]).values.push_back(std::stod(s));
            } else
            {
                if (s==MISSING_VALUE) 
                {
                    std::get<CategirialColumn>(columns[header[i].name]).values.push_back(std::nullopt);
                    header[i].missing++;
                }
                else                  
                    std::get<CategirialColumn>(columns[header[i].name]).values.push_back(s);
            }
        }
    }
    int prev_records;
    if (std::holds_alternative<NumericColumn>(columns.begin()->second)) 
        prev_records = std::get<NumericColumn>(columns.begin()->second).values.size();
    else
        prev_records = std::get<CategirialColumn>(columns.begin()->second).values.size();
    if (prev_records == 0)
        throw std::runtime_error("CSV has no records\n");

    for (auto it = columns.begin(); it != columns.end(); ++it)
    {
        if (std::holds_alternative<NumericColumn>(it->second)) 
            records = std::get<NumericColumn>(it->second).values.size();
        else                                       
            records = std::get<CategirialColumn>(it->second).values.size();
        if (records != prev_records) throw std::runtime_error("columns have different sizes\n");
        prev_records = records;
    }

    std::sort(header.begin(), header.end(), 
        [](const ColumnInfo& left, const ColumnInfo& right)
        { return left.name < right.name; });
    
    
    for (const auto& f : header)
    {
        bool numeric = f.is_num && f.unique > _UNIQUE;
        if (numeric)
            stats.emplace_back(f.missing, NumStat());
        else
            stats.emplace_back(f.missing, CatStat());
        features_sorted.emplace_back(f.name);
    }
}

void Dataset::print_info() const
{
    std::vector<std::pair<std::string, const Column*>> ordered_columns;
    ordered_columns.reserve(columns.size());
    for (const auto& [name, column] : columns)
        ordered_columns.emplace_back(name, &column);
    std::sort(ordered_columns.begin(), ordered_columns.end(),
              [](const auto& left, const auto& right) {
                  return left.first < right.first;
              });

    std::cout << "Objects: " << records
              << "\nFeatures: " << columns.size() << "\n\n";
    std::cout << std::left
              << std::setw(4) << "#"
              << std::setw(18) << "Feature"
              << std::setw(14) << "Type"
              << "Missed\n";

    int total_missing = 0;
    int columns_with_missing = 0;
    for (std::size_t index = 0; index < ordered_columns.size(); ++index)
    {
        const auto& [name, column] = ordered_columns[index];
        const bool is_numeric = std::holds_alternative<NumericColumn>(*column);
        const int missing = stats[index].missing;

        total_missing += missing;
        if (missing > 0)
            ++columns_with_missing;

        const double missing_percent =
            records == 0 ? 0.0 : 100.0 * missing / records;
        std::cout << std::left
                  << std::setw(4) << index+1
                  << std::setw(18) << name
                  << std::setw(14)
                  << (is_numeric ? "numeric" : "categorial")
                  << missing << " ("
                  << std::fixed << std::setprecision(2)
                  << missing_percent << "%)\n";
    }

    std::cout << "\nTotall missed: " << total_missing
              << " in " << columns_with_missing << " features\n";
}

void Dataset::fit()
{
    for (int i = 0; i < columns.size(); ++i)
    {
        auto& col = operator[](i);
        int missing = stats[i].missing;
        if (std::holds_alternative<NumericColumn>(col))
        {
            auto& [min, max, mean, var, stand, q] = std::get<NumStat>(stats[i].stat);
            std::vector<std::optional<double>> vals = std::get<NumericColumn>(col).values;
            std::sort(vals.begin(), vals.end(),
                [](const std::optional<double>& left,
                const std::optional<double>& right)
                {
                    return !left.has_value() || (right.has_value() && (left.value() < right.value()));
                });
            min = vals[missing].value();
            max = vals.back().value();
    
            mean = var = stand = 0;
            for (auto& j : vals) mean += j.value_or(0);
            mean /= (records - missing);
    
            for (auto& j : vals)
                if (j.has_value())
                    var += std::pow(j.value() - mean, 2);
            var /= (records - missing - 1);
            stand = std::pow(var, 0.5);
    
            double p[5] = {0.05, 0.25, 0.5, 0.75, 0.95};
            for (int j = 0; j < 5; ++j)
            {
                double pos = (records - 1 - missing)*p[j] + 1.0;
                if (std::fabs(pos - std::round(pos)) < EPS)
                    q[j] = vals[(int)pos + missing].value();
                else
                    q[j] = vals[static_cast<int>(pos) + missing].value()
                        + (vals[static_cast<int>(pos) + missing + 1].value()
                        - vals[static_cast<int>(pos) + missing].value()
                        ) * (pos - std::trunc(pos));
            }
        } else
        {
            std::vector<std::optional<std::string>> vals = std::get<CategirialColumn>(col).values;
            auto& categories = std::get<CatStat>(stats[i].stat).categories;
    
            for (const auto& c : vals)
            {
                if (!c.has_value()) continue;
                const auto& it = std::find_if(categories.begin(), categories.end(), 
                    [&c](const std::pair<std::string, int>& p){return p.first == c.value();});
                if (it == categories.end())
                    categories.push_back({c.value(), 1});
                else
                    it->second++;            
            }
        }
    }
}

void Dataset::fit_another(const Dataset& other)
{
    for (std::size_t i = 0; i < stats.size(); ++i)
        stats[i].stat = other.stats[i].stat;
}

void Dataset::print_stat_by_i(int i) const
{
    const Column& col = operator[](i);
    int missing = stats[i].missing;
    if (std::holds_alternative<NumericColumn>(col))
    {
        const auto& [min, max, mean, var, stand, q] = std::get<NumStat>(stats[i].stat);
        std::vector<std::optional<double>> vals = std::get<NumericColumn>(col).values;
        std::sort(vals.begin(), vals.end(),
                [](const std::optional<double>& left,
                const std::optional<double>& right)
                {
                    return !left.has_value() || (right.has_value() && (left.value() < right.value()));
                });

        std::cout << "\nFeature " << i+1 << ": " << features_sorted[i]
                  << " (numeric)\n\n"
                  << "  missing: " << missing << '\n'
                  << "  min: " << min << "    max: " << max << '\n'
                  << "  mean: " << mean << "   var: " << var << "\n\n"
                  << "  q05: " << q[0] << '\n'
                  << "  q25: " << q[1] << '\n'
                  << "  q50: " << q[2] << '\n'
                  << "  q75: " << q[3] << '\n'
                  << "  q95: " << q[4] << "\n\n";

        double segment = (max-min) / NUMERIC_HISTOGRAM_INTERVALS;
        double bound = min;
        std::vector<std::tuple<double, double, int>> histogram(NUMERIC_HISTOGRAM_INTERVALS);
        for (auto& c : histogram)
        {
            c = {bound, bound + segment, 0};
            bound += segment;
        }
        auto it = histogram.begin();
        for (const auto& d : vals)
        {
            if (!d.has_value()) continue;
            for (;it != histogram.end() && d.value() >= std::get<1>(*it); ++it);
            if (it == histogram.end()) 
                ++std::get<2>(histogram.back());
            else ++std::get<2>(*it);
        }
        
        std::cout << "  Histogram:\n";
        for (const auto& [left, right, frequency] : histogram)
        {
            std::string bracket = std::tuple{left, right, frequency}==histogram.back() 
                    ? "] " : ") "; 
            std::ostringstream interval;
            interval << '[' << std::fixed << std::setprecision(2)
                     << left << ", " << right << bracket;
            const std::string bar(std::lround(
                static_cast<double>(frequency) / records
                * MAX_HISTOGRAM_LINE), '#');

            std::cout << "  " << std::left << std::setw(24) << interval.str()
                      << std::setw(MAX_HISTOGRAM_LINE + 2) << bar
                      << std::right << std::setw(6) << frequency << '\n';
        }
    } else
    {
        const std::vector<std::optional<std::string>> vals = std::get<CategirialColumn>(col).values;
        const auto& categories = std::get<CatStat>(stats[i].stat).categories;

        std::cout << "\nFeature " << i+1 << ": " << features_sorted[i]
                  << " (categorial)\n\n"
                  << "  missing: " << missing << "\n\n"
                  << "  Categories:\n";

        for (const auto& [category, frequency] : categories)
            std::cout << "    " << std::left << std::setw(24) << category << ": "
                      << frequency << '\n';

        std::cout << "\n  Histogram:\n";
        for (const auto& [category, frequency] : categories)
        {
            const std::string bar(std::lround(
                static_cast<double>(frequency) / records
                * MAX_HISTOGRAM_LINE), '#');
            std::cout << "    " << std::left << std::setw(24) << category
                      << std::setw(MAX_HISTOGRAM_LINE + 2) << bar
                      << std::right << std::setw(6) << frequency << '\n';
        }
    }
}

std::vector<EncodedColumn> Dataset::transform()
{
    using ES = EncodeStrategy;
    fill_missing();
    scale_nums();
    std::vector<EncodedColumn> (Dataset::*encode)()const;
    switch (ENCODE_STRATEGY)
    {
        case ES::one_hot: encode = &Dataset::one_hot; break;
        case ES::ordinal: encode = &Dataset::ordinal; break;
        default: throw std::runtime_error("invalid encode strategy.");
    }
    return (this->*encode)();
}

void Dataset::save_binary_matrix(const std::string& path)
{
    
    Matrix m = build_matrix(transform(), records);

    std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Output file not found.");

    std::size_t rows = m.n_rows, cols = m.n_cols;
    out.write(reinterpret_cast<const char*>(&rows), sizeof(std::size_t));               //rows
    out.write(reinterpret_cast<const char*>(&cols), sizeof(std::size_t));               //cols
    for(auto& name : m.cols)
    {
        std::size_t len = name.length();
        out.write(reinterpret_cast<const char*>(&len), sizeof(std::size_t));            //column name length
        out.write(name.data(), len);                                                    //column name
    }
    out.write(reinterpret_cast<const char*>(m.data.data()),                             //data
        m.data.size() * sizeof(double));
    if (out.fail()) throw std::runtime_error("binary output error.");
}

const Column& Dataset::operator[](const std::string& s) const
{
    if (!columns.contains(s)) throw std::out_of_range("dataset has no this feature");
    return columns.at(s);
}

const Column& Dataset::operator[](int i) const
{
    if (i < 0 || i >= columns.size()) throw std::out_of_range("index is out of bounds");
    return columns.at(features_sorted[i]);
}

void Dataset::fill_missing()
{
    using FS = FillStrategy;
    double miss;
    
    for (int i = 0; i < stats.size(); ++i)
    {
        if (stats[i].missing == 0) continue;
        if (features_sorted[i] == target_feature) continue;
        if (std::holds_alternative<NumStat>(stats[i].stat))
        {
            for (auto& x : std::get<NumericColumn>(columns[features_sorted[i]]).values)
                if (!x.has_value()) 
                    switch (FILL_STRATEGY)
                    {
                        case FS::mean:
                            x = std::get<NumStat>(stats[i].stat).mean;
                            break;
                        case FS::median:
                            x = std::get<NumStat>(stats[i].stat).q[2];
                            break;
                        case FS::zero:
                            x = 0.0;
                            break;
                        default: throw std::runtime_error("unknown fill strategy\n");
                    }
        } else
        {
            for (auto& x : std::get<CategirialColumn>(columns[features_sorted[i]]).values)
                if (!x.has_value()) x = FILLED_MISSING_VALUE;
        }
    }
}

void Dataset::scale_nums()
{
    using SS = ScaleStrategy;
    for (int i = 0; i < stats.size(); ++i)
    {
        if (std::holds_alternative<CategirialColumn>(columns[features_sorted[i]])) continue;
        if (features_sorted[i] == target_feature) continue;
        auto& col = std::get<NumericColumn>(columns[features_sorted[i]]).values;
        auto& stat = std::get<NumStat>(stats[i].stat);
        for (auto& v : col)
            switch(SCALE_STRATEGY)
            {
                case SS::min_max:
                    v = (v.value() - stat.min) / (stat.max - stat.min);
                    break;
                case SS::standartisation:
                    v = (v.value() - stat.mean) / stat.stand;
                    break;
                case SS::robust_scaling:
                    v = (v.value() - stat.q[2]) / (stat.q[3] - stat.q[1]);
                    break;
                default: throw std::runtime_error("unknown scale strategy.");
            }
    }
}

std::vector<EncodedColumn> Dataset::one_hot() const
{
    std::vector<EncodedColumn> result;
    for (int i = 0; i < stats.size(); ++i)
    {
        if (features_sorted[i] == target_feature) continue;
        const auto& col = operator[](i);
        if (std::holds_alternative<NumericColumn>(col))
        {
            EncodedColumn ecol = {features_sorted[i], {}};
            for (const auto& v : std::get<NumericColumn>(col).values)
                ecol.values.push_back(v.value());
            result.push_back(std::move(ecol));
        } else
        {
            auto cats = std::get<CatStat>(stats[i].stat).categories;
            if (stats[i].missing) cats.push_back({FILLED_MISSING_VALUE, stats[i].missing});
            for (auto [cat, n] : cats)
            {
                EncodedColumn ecol = {features_sorted[i] + "_" + cat, {}};
                for (const auto& v : std::get<CategirialColumn>(col).values)
                    ecol.values.push_back(v.value() == cat ? 1.0 : 0.0);
                result.push_back(std::move(ecol));
            }
        }
    }
    return result;
}

std::vector<EncodedColumn> Dataset::ordinal() const
{
    std::vector<EncodedColumn> result;
    for (int i = 0; i < stats.size(); ++i)
    {
        if (features_sorted[i] == target_feature) continue;
        const auto& col = operator[](i);
        if (std::holds_alternative<NumericColumn>(col))
        {
            EncodedColumn ecol = {features_sorted[i], {}};
            for (const auto& v : std::get<NumericColumn>(col).values)
                ecol.values.push_back(v.value());
            result.push_back(std::move(ecol));
        } else
        {
            auto cats = std::get<CatStat>(stats[i].stat).categories;
            EncodedColumn ecol = {features_sorted[i], {}};
            if (stats[i].missing) cats.push_back({FILLED_MISSING_VALUE, stats[i].missing});
            for (const auto& v : std::get<CategirialColumn>(col).values)
                for (int j = 0; j < cats.size(); ++j)
                {
                    const std::string& cat = cats[j].first; 
                    if (cat == v.value())
                        ecol.values.push_back(j);
                }
            result.push_back(ecol);
        }
    }
    return result;
}

std::pair<Dataset, Dataset> Dataset::train_test_split(double train_part) const
{
    if (train_part > 1 || train_part <= 0)
        throw std::invalid_argument("invalid train part argument.");
    std::size_t train = records * train_part;
    std::unordered_map<std::string, Column> train_data;
    std::unordered_map<std::string, Column> test_data;
    train_data.reserve(size());
    test_data.reserve(size());

    for (const auto& [name, col] : columns)
    {
        if (std::holds_alternative<NumericColumn>(col))
        {
            train_data[name] = test_data[name] = NumericColumn();
            std::get<NumericColumn>(train_data[name]).values.reserve(train);
            std::get<NumericColumn>(test_data[name]).values.reserve(records - train);
        } else
        {
            train_data[name] = test_data[name] = CategirialColumn();
            std::get<CategirialColumn>(train_data[name]).values.reserve(train);
            std::get<CategirialColumn>(test_data[name]).values.reserve(records - train);
        }
        
    }
    for (std::size_t i = 0; i < train; ++i)
        for (auto [name, col] : columns)
            if (std::holds_alternative<NumericColumn>(col))
                 std::get<NumericColumn>(train_data[name]).values
                    .emplace_back(std::get<NumericColumn>(col).values[i]);
            else
                std::get<CategirialColumn>(train_data[name]).values
                    .emplace_back(std::get<CategirialColumn>(col).values[i]);
    for (std::size_t i = train; i < records; ++i)
        for (auto [name, col] : columns)
            if (std::holds_alternative<NumericColumn>(col))
                 std::get<NumericColumn>(test_data[name]).values
                    .emplace_back(std::get<NumericColumn>(col).values[i]);
            else
                std::get<CategirialColumn>(test_data[name]).values
                    .emplace_back(std::get<CategirialColumn>(col).values[i]);
    return std::make_pair(
            Dataset(std::move(train_data), target_feature), 
            Dataset(std::move(test_data), target_feature)
        );
}

Dataset::Dataset(std::unordered_map<std::string, Column> data,
        std::string target_feature)
    : columns(std::move(data)), target_feature(target_feature)
{
    if (std::holds_alternative<NumericColumn>(columns.begin()->second))
        records = std::get<NumericColumn>(columns.begin()->second).values.size();
    else
        records = std::get<CategirialColumn>(columns.begin()->second).values.size();
    features_sorted.reserve(size());
    stats.reserve(size());
    for (auto [name, col] : columns)
        features_sorted.emplace_back(name);
    std::sort(features_sorted.begin(), features_sorted.end());

    for (const std::string& name : features_sorted)
    {
        if (std::holds_alternative<NumericColumn>(columns[name]))
        {
            stats.emplace_back(0,NumStat());
            const auto& values = std::get<NumericColumn>(columns[name]).values;
            for (const auto& v : values)
                if (!v.has_value())
                    ++stats.back().missing;
        }
        else
        {
            stats.emplace_back(0,CatStat());
            const auto& values = std::get<CategirialColumn>(columns[name]).values;
            for (const auto& v : values)
                if (!v.has_value())
                    ++stats.back().missing;
        }
    }

    
}


}