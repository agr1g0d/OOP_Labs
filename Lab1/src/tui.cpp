#include <iostream>
#include <fstream>
#include <exception>
#include <limits>
#include <string>
#include <cxxopts.hpp>
#include "dataset.h"


namespace fs
{
    std::string BIN_PATH;
    std::string CSV_PATH;
}


void print_menu()
{
    std::cout << "\n=== Menu ===\n"
              << "1 - Print dataset summary\n"
              << "2 - Print feature statistics\n"
              << "3 - Preprocess and write dataset\n"
              << "0 - Exit\n"
              << "Choose an action: ";
}

void print_statistics_placeholder(const ds::Dataset& dataset)
{
    while (1)
    {
        std::cout << "\nEnter the feature index or 0 to return:\n";
        int i;
        std::cin >> i;
        if (std::cin.fail())
        {
            std::cout << "Value must be a number.\n";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            if (std::cin.eof())
                return;
            continue;
        }
        if (i > dataset.size() || i < 0)
        {
            std::cout << "Index is out of range.\n";
            continue;
        }
        if (!i) return;
        dataset.print_stat_by_i(i-1);

    }
}

void preprocess_and_write(ds::Dataset& dataset)
{
    dataset.save_binary_matrix(fs::BIN_PATH);
    std::ifstream csv(fs::CSV_PATH, std::ios::binary | std::ios::ate);
    std::ifstream bin(fs::BIN_PATH, std::ios::binary | std::ios::ate);
    std::cout << "CSV:    " << csv.tellg() << " bytes\n";
    std::cout << "Binary: " << bin.tellg() << " bytes\n";
}

ds::FillStrategy parse_fill(std::string s)
{
    static const std::unordered_map<std::string, ds::FillStrategy> m = 
    {
        {"median", ds::FillStrategy::median},
        {"mean", ds::FillStrategy::mean},
        {"zero", ds::FillStrategy::zero}
    };
    auto it = m.find(s);
    if (it == m.end()) throw std::runtime_error("Unknown missings fill strategy type.\n");
    return it->second;
}
ds::ScaleStrategy parse_scale(std::string s)
{
    static const std::unordered_map<std::string, ds::ScaleStrategy> m = 
    {
        {"min-max", ds::ScaleStrategy::min_max},
        {"standartisation", ds::ScaleStrategy::standartisation},
        {"robust_scaling", ds::ScaleStrategy::robust_scaling}
    };
    auto it = m.find(s);
    if (it == m.end()) throw std::runtime_error("Unknown numeric scaling strategy type.\n");
    return it->second;
}
ds::EncodeStrategy parse_encoding(std::string s)
{
    static const std::unordered_map<std::string, ds::EncodeStrategy> m = 
    {
        {"one-hot", ds::EncodeStrategy::one_hot},
        {"ordinal", ds::EncodeStrategy::ordinal}
    };
    auto it = m.find(s);
    if (it == m.end()) throw std::runtime_error("Unknown numeric scaling strategy type.\n");
    return it->second;
}

int main(int argc, char* argv[])
{
    cxxopts::Options opts("OOP labs", "This program is SPbPU coding pracrice.\n");
    opts.add_options()
        ("p,path_to_csv","A path to CSV dataset file.",cxxopts::value<std::string>()
            ->default_value("../data/Autism-Adult-Data.csv"))
        ("s,path_to_save","A path to output file.",cxxopts::value<std::string>()
            ->default_value("../data/Autism-Adult-Data.bin"))
        ("i,intervals","How many intervals will have numeric diograms.",cxxopts::value<int>()
            ->default_value("10"))
        ("f,fill","Filling of missings numerics (mean/median/zero).",cxxopts::value<std::string>()
            ->default_value("median"))
        ("c,scale_numeric","The way numerics will scale (min-max/standartisation/robust_scaling).",cxxopts::value<std::string>()
            ->default_value("min-max"))
        ("e,encode_categoric","The way categorial features will encode (one-hot/ordinal).",cxxopts::value<std::string>()
            ->default_value("one-hot"))
        ("h,help","Print availible comands");
    
    auto parsed = opts.parse(argc, argv);
    if (parsed.count("h"))
    {
        std::cout << opts.help() << "\n";
        return 1;
    }
    fs::CSV_PATH = parsed["p"].as<std::string>();
    fs::BIN_PATH = parsed["s"].as<std::string>();
    ds::NUMERIC_HISTOGRAM_INTERVALS = parsed["i"].as<int>();
    try {
        ds::FILL_STRATEGY = parse_fill(parsed["f"].as<std::string>());
        ds::SCALE_STRATEGY = parse_scale(parsed["c"].as<std::string>());
        ds::ENCODE_STRATEGY = parse_encoding(parsed["e"].as<std::string>());
    } catch (const std::runtime_error& exc)
    {
        std::cerr  << exc.what() << "\n";
        return 1;
    }

    try {
        ds::Dataset dataset(fs::CSV_PATH);
        bool running = true;

        while (running)
        {
            print_menu();

            char action;
            if (!(std::cin >> action))
            {
                std::cout << "\nInput finished.\n";
                break;
            }
            switch (action)
            {
                case '1':
                    dataset.print_info();
                    break;
                case '2':
                    print_statistics_placeholder(dataset);
                    break;
                case '3':
                    preprocess_and_write(dataset);
                    running = false;
                    break;
                case '0':
                    std::cout << "\nExiting.\n";
                    running = false;
                    break;
                default:
                    std::cout << "\nUnknown command. Choose an item from the menu.\n";
                    break;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "Ошибка: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
