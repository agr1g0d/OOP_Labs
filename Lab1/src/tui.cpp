#include <iostream>
#include <algorithm>
#include <fstream>
#include <exception>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <variant>
#include <cxxopts.hpp>
#include "dataset.h"
#include "model.h"


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
              << "4 - Train kNN and predict test data\n"
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

struct TargetValues
{
    std::vector<double> encoded;
    std::vector<std::string> displayed;
    bool categorical;
};

TargetValues encode_target(
    const ds::Dataset& dataset,
    const std::map<std::string, double>* category_codes = nullptr)
{
    const ds::Column& target = dataset.get_target();
    TargetValues result;
    if (std::holds_alternative<ds::NumericColumn>(target))
    {
        result.categorical = false;
        for (const auto& value : std::get<ds::NumericColumn>(target).values)
        {
            if (!value.has_value())
                throw std::invalid_argument("target contains missing values.");
            result.encoded.push_back(value.value());
            std::ostringstream displayed;
            displayed << value.value();
            result.displayed.push_back(displayed.str());
        }
    }
    else
    {
        result.categorical = true;
        std::map<std::string, double> local_categories;
        const auto& categories = category_codes == nullptr
            ? local_categories
            : *category_codes;
        for (const auto& value : std::get<ds::CategirialColumn>(target).values)
        {
            if (!value.has_value())
                throw std::invalid_argument("target contains missing values.");
            if (category_codes == nullptr)
                local_categories.emplace(value.value(), local_categories.size());
        }

        for (const auto& value : std::get<ds::CategirialColumn>(target).values)
        {
            const auto& category = value.value();
            result.encoded.push_back(categories.at(category));
            result.displayed.push_back(category);
        }
    }
    return result;
}

std::unique_ptr<model::IDistance> choose_distance(model::DistanceType distance)
{
    if (distance == model::DistanceType::euclid)
        return std::make_unique<model::EuclidDistance>();
    if (distance == model::DistanceType::manhattan)
        return std::make_unique<model::ManhattanDistance>();
    if (distance == model::DistanceType::chebyshev)
        return std::make_unique<model::ChebyshevDistance>();
    throw std::invalid_argument("Unknown distance metric.");
}

void train_and_predict(const ds::Dataset& dataset)
{
    auto distance = choose_distance(model::DISTANCE);
    auto [train, test] = dataset.train_test_split(ds::TRAIN_PART);
    train.fit();
    test.fit_another(train);

    std::map<std::string, double> category_codes;
    const ds::Column& target = dataset.get_target();
    if (std::holds_alternative<ds::CategirialColumn>(target))
    {
        for (const auto& value : std::get<ds::CategirialColumn>(target).values)
        {
            if (!value.has_value())
                throw std::invalid_argument("target contains missing values.");
            category_codes.emplace(value.value(), category_codes.size());
        }
    }
    const TargetValues train_target = encode_target(
        train, category_codes.empty() ? nullptr : &category_codes);
    const TargetValues test_target = encode_target(
        test, category_codes.empty() ? nullptr : &category_codes);
    const ds::Matrix train_features =
        ds::build_matrix(train.transform(), train_target.encoded.size());
    const ds::Matrix test_features =
        ds::build_matrix(test.transform(), test_target.encoded.size());

    std::unique_ptr<model::IModel> estimator;
    if (train_target.categorical)
        estimator = std::make_unique<model::KnnClassifier>(
            model::K, *distance, model::WEIGHTING);
    else
        estimator = std::make_unique<model::KnnRegressor>(
            model::K, *distance, model::WEIGHTING);

    estimator->fit(train_features, train_target.encoded);
    const std::vector<double> predictions = estimator->predict(test_features);
    std::mt19937 generator(std::random_device{}());
    while (true)
    {
        std::cout << "\n1 - Print five random predictions\n"
                  << "0 - Return to main menu\n"
                  << "Choose an action: ";
        char action;
        if (!(std::cin >> action) || action == '0')
            return;
        if (action != '1')
        {
            std::cout << "Unknown command. Choose an item from the menu.\n";
            continue;
        }

        const std::size_t sample_size = std::min<std::size_t>(5, predictions.size());
        std::vector<std::size_t> sample_indices(predictions.size());
        std::iota(sample_indices.begin(), sample_indices.end(), 0);
        std::shuffle(sample_indices.begin(), sample_indices.end(), generator);
        std::cout << "\nPredictions for " << sample_size
                  << " random test objects:\n"
                  << std::left << std::setw(12) << "Prediction"
                  << "Actual\n";
        for (std::size_t sample = 0; sample < sample_size; ++sample)
        {
            const std::size_t i = sample_indices[sample];
            if (train_target.categorical)
            {
                const auto predicted_code = static_cast<std::size_t>(predictions[i]);
                std::string predicted = std::to_string(predictions[i]);
                for (const auto& [category, code] : category_codes)
                    if (static_cast<std::size_t>(code) == predicted_code)
                        predicted = category;
                std::cout << std::left << std::setw(12) << predicted
                          << test_target.displayed[i] << '\n';
            }
            else
            {
                std::cout << std::left << std::setw(12) << predictions[i]
                          << test_target.displayed[i] << '\n';
            }
        }
    }
}

int check_col_intervals(int val)
{
    if (val <= 0)
        throw std::invalid_argument("Invalid histogram interval argement.\n");
    return val;
}
double check_train_part(double val)
{
    if (val < 0 || val > 1)
        throw std::invalid_argument("Invalid train part argement.\n");
    return val;
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
    if (it == m.end()) throw std::invalid_argument("Unknown missings fill strategy type.\n");
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
    if (it == m.end()) throw std::invalid_argument("Unknown numeric scaling strategy type.\n");
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
    if (it == m.end()) throw std::invalid_argument("Unknown numeric scaling strategy type.\n");
    return it->second;
}
int check_k(int val)
{
    if (val <= 0)
        throw std::invalid_argument("k must be greater than zero.\n");
    return val;
}
model::DistanceType parse_distance(std::string s)
{
    static const std::unordered_map<std::string, model::DistanceType> m =
    {
        {"euclid", model::DistanceType::euclid},
        {"manhattan", model::DistanceType::manhattan},
        {"chebyshev", model::DistanceType::chebyshev}
    };
    auto it = m.find(s);
    if (it == m.end())
        throw std::invalid_argument("Unknown distance metric type.\n");
    return it->second;
}
model::WeightingType parse_weighting(std::string s)
{
    static const std::unordered_map<std::string, model::WeightingType> m =
    {
        {"equal", model::WeightingType::equal},
        {"reverse", model::WeightingType::reverse}
    };
    auto it = m.find(s);
    if (it == m.end())
        throw std::invalid_argument("Unknown weighting type.\n");
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
        ("t,train_part","The part of dataset thar will be used to train model. The value is in [0,1].",cxxopts::value<double>()
            ->default_value("0.8"))
        ("k,k_nearest_neighbours","K in the kNN algorithm.",cxxopts::value<int>()
            ->default_value("30"))
        ("d,distance","The way of distance computing (euclid/manhattan/shebyshev).",cxxopts::value<std::string>()
            ->default_value("euclid"))
        ("w,weighting","The way of weighting neighbours in kNN (equal/reverse).",cxxopts::value<std::string>()
            ->default_value("reverse"))
        ("h,help","Print availible comands");
    
    auto parsed = opts.parse(argc, argv);
    if (parsed.count("h"))
    {
        std::cout << opts.help() << "\n";
        return 0;
    }
    try {
        fs::CSV_PATH = parsed["p"].as<std::string>();
        fs::BIN_PATH = parsed["s"].as<std::string>();
        ds::NUMERIC_HISTOGRAM_INTERVALS = check_col_intervals(parsed["i"].as<int>());
        ds::TRAIN_PART = check_train_part(parsed["t"].as<double>());
        ds::FILL_STRATEGY = parse_fill(parsed["f"].as<std::string>());
        ds::SCALE_STRATEGY = parse_scale(parsed["c"].as<std::string>());
        ds::ENCODE_STRATEGY = parse_encoding(parsed["e"].as<std::string>());
        model::K = check_k(parsed["k"].as<int>());
        model::DISTANCE = parse_distance(parsed["d"].as<std::string>());
        model::WEIGHTING = parse_weighting(parsed["w"].as<std::string>());
    } catch (const std::invalid_argument& exc)
    {
        std::cerr  << exc.what() << "\n";
        return 1;
    }

    try {
        ds::Dataset dataset(fs::CSV_PATH);
        dataset.fit();
        
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
                case '4':
                    train_and_predict(dataset);
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
