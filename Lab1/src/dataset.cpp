#include "dataset.h"
#include <iostream>
#include <fstream>
#include <exception>
#include <string_view>
#include <charconv>
#include <algorithm>
#include <iomanip>
#include <vector>

namespace ds{

inline void pop_r(std::string& s)
{
    if (!s.empty() && s.back() == '\r')
        s.pop_back();
}

Dataset::Dataset(std::string path)
{
    std::ifstream file(path, std::ios::in);
    if (!file.is_open()) throw std::runtime_error("error while opening file: " + path);
    if (file.eof()) throw std::runtime_error("CSV is empty\n");
    
    std::vector<Dataset::ColumnInfo> header;
    int features;

    char c; file.get(c);
    for (; c != '\n'; file.get(c))
        if (c == SPLIT)
            header.push_back(Dataset::ColumnInfo());
    header.push_back(Dataset::ColumnInfo());
    features = header.size();

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

    columns.reserve(features);
    file.clear();
    file.seekg(0);
    for (int i = 0; i < features; ++i)
    {
        char last = (i == features-1) ? '\n' : SPLIT;
        std::getline(file, header[i].name, last);
        pop_r(header[i].name);
        if (header[i].is_num && header[i].unique > _UNIQUE)
            columns[header[i].name] = NumericColumn();
        else
            columns[header[i].name] = CategirialColumn();
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
                    std::get<NumericColumn>(columns[header[i].name]).missing++;
                }
                else                  
                    std::get<NumericColumn>(columns[header[i].name]).values.push_back(std::stod(s));
            } else
            {
                if (s==MISSING_VALUE) 
                {
                    std::get<CategirialColumn>(columns[header[i].name]).missing++;
                    std::get<CategirialColumn>(columns[header[i].name]).values.push_back(std::nullopt);
                }
                else                  
                    std::get<CategirialColumn>(columns[header[i].name]).values.push_back(s);
            }
        }
    }
    int prevsize;
    if (std::holds_alternative<NumericColumn>(columns.begin()->second)) 
        prevsize = std::get<NumericColumn>(columns.begin()->second).values.size();
    else
        prevsize = std::get<CategirialColumn>(columns.begin()->second).values.size();

    for (auto it = columns.begin(); it != columns.end(); ++it)
    {
        if (std::holds_alternative<NumericColumn>(it->second)) 
            size = std::get<NumericColumn>(it->second).values.size();
        else                                       
            size = std::get<CategirialColumn>(it->second).values.size();
        if (size != prevsize) throw std::runtime_error("columns have different sizes\n");
        prevsize = size;
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

    std::cout << "Objects: " << size
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
        const int missing = is_numeric
            ? std::get<NumericColumn>(*column).missing
            : std::get<CategirialColumn>(*column).missing;

        total_missing += missing;
        if (missing > 0)
            ++columns_with_missing;

        const double missing_percent =
            size == 0 ? 0.0 : 100.0 * missing / size;
        std::cout << std::left
                  << std::setw(4) << index
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

}