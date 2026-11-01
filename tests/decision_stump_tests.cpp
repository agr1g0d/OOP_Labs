#include "exc.h"
#include "dataset.h"
#include "matrix.h"
#include "model.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
struct TestFailure : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

void check(bool condition, const std::string& message)
{
    if (!condition)
        throw TestFailure(message);
}

void check_close(double actual, double expected, const std::string& message)
{
    check(std::abs(actual - expected) < 1e-9, message);
}

template <typename Exception, typename Function>
void check_throws(Function&& function, const std::string& message)
{
    try
    {
        function();
    }
    catch (const Exception&)
    {
        return;
    }
    catch (const std::exception& error)
    {
        throw TestFailure(message + " (unexpected exception: " + error.what() + ")");
    }
    throw TestFailure(message + " (no exception)");
}

ds::Matrix make_matrix(std::size_t rows, std::size_t cols,
                       std::vector<double> data)
{
    check(data.size() == rows * cols, "invalid test matrix data");
    return {rows, cols, std::vector<std::string>(cols), std::move(data)};
}

struct Split
{
    std::size_t feature;
    double threshold;
    double score;
};

Split naive_best_split(const ds::Matrix& features,
                       std::span<const double> target,
                       model::SplitCriterion criterion,
                       std::size_t min_samples_leaf)
{
    Split best{0, 0.0, -1.0};
    bool found = false;
    for (std::size_t j = 0; j < features.n_cols; ++j)
    {
        for (std::size_t candidate = 0; candidate < features.n_rows; ++candidate)
        {
            const double threshold = features[candidate][j];
            std::vector<std::size_t> left;
            std::vector<std::size_t> right;
            for (std::size_t i = 0; i < features.n_rows; ++i)
            {
                if (features[i][j] <= threshold)
                    left.push_back(i);
                else
                    right.push_back(i);
            }
            if (left.size() < min_samples_leaf
                || right.size() < min_samples_leaf)
                continue;

            double score = 0.0;
            if (criterion == model::SplitCriterion::Variance)
            {
                double left_sum = 0.0;
                double right_sum = 0.0;
                for (const std::size_t i : left)
                    left_sum += target[i];
                for (const std::size_t i : right)
                    right_sum += target[i];
                score = left_sum * left_sum / left.size()
                    + right_sum * right_sum / right.size();
            }
            else
            {
                std::unordered_map<double, std::size_t> left_counts;
                std::unordered_map<double, std::size_t> right_counts;
                for (const std::size_t i : left)
                    ++left_counts[target[i]];
                for (const std::size_t i : right)
                    ++right_counts[target[i]];
                for (const auto& [category, count] : left_counts)
                    score += static_cast<double>(count * count) / left.size();
                for (const auto& [category, count] : right_counts)
                    score += static_cast<double>(count * count) / right.size();
            }

            if (!found || score > best.score)
            {
                found = true;
                best = {j, threshold, score};
            }
        }
    }
    check(found, "naive stump must find a valid split");
    return best;
}

double split_score(const ds::Matrix& features,
                   std::span<const double> target,
                   model::SplitCriterion criterion,
                   std::size_t feature,
                   double threshold)
{
    std::vector<std::size_t> left;
    std::vector<std::size_t> right;
    for (std::size_t i = 0; i < features.n_rows; ++i)
        (features[i][feature] <= threshold ? left : right).push_back(i);
    if (left.empty() || right.empty())
        return -1.0;

    double score = 0.0;
    if (criterion == model::SplitCriterion::Variance)
    {
        double left_sum = 0.0;
        double right_sum = 0.0;
        for (const auto i : left)
            left_sum += target[i];
        for (const auto i : right)
            right_sum += target[i];
        score = left_sum * left_sum / left.size()
            + right_sum * right_sum / right.size();
    }
    else
    {
        std::unordered_map<double, std::size_t> left_counts;
        std::unordered_map<double, std::size_t> right_counts;
        for (const auto i : left)
            ++left_counts[target[i]];
        for (const auto i : right)
            ++right_counts[target[i]];
        for (const auto [category, count] : left_counts)
            score += static_cast<double>(count * count) / left.size();
        for (const auto [category, count] : right_counts)
            score += static_cast<double>(count * count) / right.size();
    }
    return score;
}

void check_stump_matches_naive(const ds::Matrix& features,
                               std::span<const double> target,
                               model::SplitCriterion criterion,
                               std::size_t min_samples_leaf)
{
    const Split expected = naive_best_split(
        features, target, criterion, min_samples_leaf);
    model::DecisionStump stump(criterion, min_samples_leaf);
    stump.fit(features, target);
    const double actual_score = split_score(
        features, target, criterion, stump.split_feature(),
        stump.split_threshold());
    check_close(actual_score, expected.score,
                "stump selected a split with a different score than naive implementation");
    check(stump.split_feature() == expected.feature,
          "stump selected a different feature than naive implementation");
}

void test_split_search()
{
    const ds::Matrix features = make_matrix(
        6, 2, {0.0, 0.0, 1.0, 0.0, 2.0, 0.0,
               3.0, 0.0, 4.0, 0.0, 5.0, 0.0});
    const std::vector<double> regression_target{0.0, 0.0, 0.0, 10.0, 10.0, 10.0};
    const std::vector<double> classification_target{0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    check_stump_matches_naive(
        features, regression_target, model::SplitCriterion::Variance, 1);
    check_stump_matches_naive(
        features, classification_target, model::SplitCriterion::Gini, 1);
}

void test_random_comparison()
{
    std::mt19937 generator(42);
    std::uniform_real_distribution<double> value(-10.0, 10.0);
    for (std::size_t trial = 0; trial < 20; ++trial)
    {
        constexpr std::size_t rows = 24;
        constexpr std::size_t columns = 3;
        std::vector<double> data(rows * columns);
        for (double& item : data)
            item = value(generator);
        std::vector<double> regression_target(rows);
        std::vector<double> classification_target(rows);
        for (std::size_t i = 0; i < rows; ++i)
        {
            regression_target[i] = static_cast<double>((i * 7 + trial) % 11);
            classification_target[i] = i % 3;
        }
        const ds::Matrix features = make_matrix(rows, columns, std::move(data));
        for (std::size_t min_leaf : {1U, 2U, 5U})
        {
            check_stump_matches_naive(
                features, regression_target,
                model::SplitCriterion::Variance, min_leaf);
            check_stump_matches_naive(
                features, classification_target,
                model::SplitCriterion::Gini, min_leaf);
        }
    }
}

void test_predictions()
{
    const ds::Matrix features = make_matrix(
        6, 1, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0});
    const std::vector<double> target{1.0, 1.0, 1.0, 9.0, 9.0, 9.0};
    model::DecisionStump stump(model::SplitCriterion::Variance, 2);
    stump.fit(features, target);
    check(stump.split_feature() == 0, "stump must select the only feature");
    check_close(stump.split_threshold(), 2.5, "stump threshold is incorrect");
    const auto predictions = stump.predict(features);
    for (std::size_t i = 0; i < 3; ++i)
        check_close(predictions[i], 1.0, "left leaf prediction is incorrect");
    for (std::size_t i = 3; i < 6; ++i)
        check_close(predictions[i], 9.0, "right leaf prediction is incorrect");
}

void test_errors_and_boundaries()
{
    const ds::Matrix train = make_matrix(4, 1, {0.0, 1.0, 2.0, 3.0});
    const std::vector<double> target{0.0, 0.0, 1.0, 1.0};
    check_throws<exc::invalid_min_samples_leaf>(
        [&] { model::DecisionStump stump(model::SplitCriterion::Gini, 0); },
        "zero min_samples_leaf must be rejected");
    check_throws<exc::invalid_train_sample>(
        [&] {
            model::DecisionStump stump(model::SplitCriterion::Gini, 1);
            stump.fit(make_matrix(4, 1, {1.0, 1.0, 1.0, 1.0}), target);
        },
        "constant features must be rejected");
    check_throws<exc::invalid_min_samples_leaf>(
        [&] {
            model::DecisionStump stump(model::SplitCriterion::Gini, 3);
            stump.fit(train, target);
        },
        "impossible leaf size must be rejected");
    check_throws<exc::predict_before_fit>(
        [&] {
            model::DecisionStump stump(model::SplitCriterion::Gini, 1);
            (void)stump.predict(train);
        },
        "stump prediction before fit must be rejected");
    check_throws<exc::invalid_prediction_sample>(
        [&] {
            model::DecisionStump stump(model::SplitCriterion::Gini, 1);
            stump.fit(train, target);
            (void)stump.predict(make_matrix(1, 2, {0.0, 1.0}));
        },
        "stump prediction with wrong feature count must be rejected");
}

void test_duplicate_values()
{
    const ds::Matrix features = make_matrix(
        6, 1, {0.0, 0.0, 1.0, 1.0, 2.0, 2.0});
    const std::vector<double> target{0.0, 0.0, 1.0, 1.0, 0.0, 0.0};
    model::DecisionStump stump(model::SplitCriterion::Gini, 1);
    stump.fit(features, target);
    check(stump.split_threshold() == 0.5 || stump.split_threshold() == 1.5,
          "stump must not place threshold between equal feature values");
}

void test_training_time_on_dataset()
{
    ds::FILL_STRATEGY = ds::FillStrategy::median;
    ds::SCALE_STRATEGY = ds::ScaleStrategy::robust_scaling;
    ds::ENCODE_STRATEGY = ds::EncodeStrategy::one_hot;
    ds::Dataset dataset(DATASET_PATH);
    dataset.fit();
    const auto target_column = dataset.get_target();
    const auto& categories =
        std::get<ds::CategirialColumn>(target_column).values;
    std::map<std::string, double> category_codes;
    for (const auto& value : categories)
    {
        check(value.has_value(), "source dataset target contains missing values");
        category_codes.emplace(value.value(), category_codes.size());
    }

    std::vector<double> target;
    target.reserve(categories.size());
    for (const auto& value : categories)
        target.push_back(category_codes.at(value.value()));
    const ds::Matrix features =
        ds::build_matrix(dataset.transform(), target.size());
    constexpr std::size_t min_samples_leaf = 10;
    constexpr std::size_t repetitions = 3;

    using clock = std::chrono::steady_clock;
    double fast_total_ms = 0.0;
    double naive_total_ms = 0.0;
    Split fast_split{};
    Split naive_split{};
    for (std::size_t repetition = 0; repetition < repetitions; ++repetition)
    {
        const auto fast_start = clock::now();
        model::DecisionStump stump(model::SplitCriterion::Gini,
                                   min_samples_leaf);
        stump.fit(features, target);
        const auto fast_end = clock::now();
        fast_total_ms += std::chrono::duration<double, std::milli>(
            fast_end - fast_start).count();
        fast_split = {
            stump.split_feature(), stump.split_threshold(), 0.0};

        const auto naive_start = clock::now();
        naive_split = naive_best_split(
            features, target, model::SplitCriterion::Gini,
            min_samples_leaf);
        const auto naive_end = clock::now();
        naive_total_ms += std::chrono::duration<double, std::milli>(
            naive_end - naive_start).count();
    }

    const double fast_average_ms = fast_total_ms / repetitions;
    const double naive_average_ms = naive_total_ms / repetitions;
    std::cout << "Decision stump timing on source dataset ("
              << features.n_rows << " rows, " << features.n_cols << " features):\n"
              << "  fast:  " << fast_average_ms << " ms\n"
              << "  naive: " << naive_average_ms << " ms\n"
              << "  speedup: " << naive_average_ms / fast_average_ms << "x\n";
    check(fast_split.feature == naive_split.feature,
          "fast and naive dataset splits use different features");
}
}

int main()
{
    try
    {
        test_split_search();
        test_random_comparison();
        test_predictions();
        test_errors_and_boundaries();
        test_duplicate_values();
        test_training_time_on_dataset();
    }
    catch (const TestFailure& failure)
    {
        std::cerr << "FAIL: " << failure.what() << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }

    std::cout << "All decision stump tests passed.\n";
    return 0;
}
