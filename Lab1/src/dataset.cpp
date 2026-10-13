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

Dataset::Dataset() {}

Dataset::Dataset(const std::string& path)
{
    load_csv(path);
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
    features_sorted.reserve(features);
    std::sort(header.begin(), header.end(), [](const auto& left, const auto& right)
        { return left.name < right.name; });
    for (int i = 0; i < features; ++i)
    {
        features_sorted.emplace_back(header[i].name);
        bool numeric = header[i].is_num && header[i].unique > _UNIQUE;
        if (numeric)
            stats.emplace_back(header[i].missing, NumStat());
        else
            stats.emplace_back(header[i].missing, CatStat());
    }

    calculate_stats();
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

void Dataset::calculate_stats()
{
    for (int i = 0; i < columns.size(); ++i)
    {
        auto& col = operator[](i);
        int missing = stats[i].missing;
        if (std::holds_alternative<NumericColumn>(col))
        {
            auto& [min, max, mean, var, q] = std::get<NumStat>(stats[i].stat);
            std::vector<std::optional<double>> vals = std::get<NumericColumn>(col).values;
            std::sort(vals.begin(), vals.end(),
                [](const std::optional<double>& left,
                const std::optional<double>& right)
                {
                    return !left.has_value() || (right.has_value() && (left.value() < right.value()));
                });
            min = vals[missing].value();
            max = vals.back().value();
    
            mean = var = 0;
            for (auto& j : vals) mean += j.value_or(0);
            mean /= (records - missing);
    
            for (auto& j : vals)
                if (j.has_value())
                    var += std::pow(j.value() - mean, 2);
            var /= (records - missing - 1);
    
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

void Dataset::print_stat_by_i(int i) const
{
    const Column& col = operator[](i);
    int missing = stats[i].missing;
    if (std::holds_alternative<NumericColumn>(col))
    {
        const auto& [min, max, mean, var, q] = std::get<NumStat>(stats[i].stat);
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

void Dataset::save_binary_matrix(const std::string& path)
{
    fill_missing();
    scale_nums();
    std::vector<EncodedColumn> data = std::move(one_hot());
    Matrix m = build_matrix(data, records);

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
    for (int i = 0; i < stats.size(); ++i)
    {
        if (stats[i].missing == 0) continue;
        if (std::holds_alternative<NumStat>(stats[i].stat))
        {
            for (auto& x : std::get<NumericColumn>(columns[features_sorted[i]]).values)
                if (!x.has_value()) x = std::get<NumStat>(stats[i].stat).q[2];
        } else
        {
            for (auto& x : std::get<CategirialColumn>(columns[features_sorted[i]]).values)
                if (!x.has_value()) x = FILLED_MISSING_VALUE;
        }
    }
}

void Dataset::scale_nums()
{
    for (int i = 0; i < stats.size(); ++i)
    {
        if (std::holds_alternative<CategirialColumn>(columns[features_sorted[i]])) continue;
        auto& col = std::get<NumericColumn>(columns[features_sorted[i]]).values;
        auto& stat = std::get<NumStat>(stats[i].stat);
        for (auto& v : col)
            v = (v.value() - stat.min) / (stat.max - stat.min);
    }
}

std::vector<EncodedColumn> Dataset::one_hot() const
{
    std::vector<EncodedColumn> result;
    for (int i = 0; i < stats.size(); ++i)
    {
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

Matrix Dataset::build_matrix(std::vector<EncodedColumn> encoded_data, std::size_t rows)
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

}