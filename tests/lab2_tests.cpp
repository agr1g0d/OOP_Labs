#include "exc.h"
#include "matrix.h"
#include "model.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
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

void test_distances()
{
    const std::vector<double> a{1.0, -2.0, 4.0};
    const std::vector<double> b{4.0, 2.0, -2.0};
    model::EuclidDistance euclid;
    model::ManhattanDistance manhattan;
    model::ChebyshevDistance chebyshev;

    check_close(euclid.compute(a, a), 0.0, "distance to itself must be zero");
    check_close(euclid.compute(a, b), std::sqrt(61.0), "euclidean distance is incorrect");
    check_close(manhattan.compute(a, b), 13.0, "manhattan distance is incorrect");
    check_close(chebyshev.compute(a, b), 6.0, "chebyshev distance is incorrect");

    const std::vector<double> empty;
    check_close(euclid.compute(empty, empty), 0.0, "empty euclidean vectors must be supported");
    check_throws<exc::incompatible_vectors>(
        [&] { (void)euclid.compute(a, empty); },
        "different vector dimensions must be rejected");
    check_throws<exc::incompatible_vectors>(
        [&] { (void)manhattan.compute(a, empty); },
        "different manhattan vector dimensions must be rejected");
    check_throws<exc::incompatible_vectors>(
        [&] { (void)chebyshev.compute(a, empty); },
        "different chebyshev vector dimensions must be rejected");
}

void test_matrix_bounds()
{
    ds::Matrix matrix = make_matrix(2, 2, {1.0, 2.0, 3.0, 4.0});
    check_close(matrix.at(0, 0), 1.0, "matrix element lookup is incorrect");
    matrix.at(1, 1) = 5.0;
    check_close(matrix.at(1, 1), 5.0, "matrix element assignment is incorrect");

    check_throws<exc::out_of_matrix>(
        [&] { (void)matrix.at(2, 0); },
        "row index at the end must be rejected");
    check_throws<exc::out_of_matrix>(
        [&] { (void)matrix.at(0, 2); },
        "column index at the end must be rejected");
    check_throws<exc::out_of_matrix>(
        [&] { (void)matrix.at(std::numeric_limits<std::size_t>::max(), 0); },
        "maximum row index must be rejected");
    check_throws<exc::out_of_matrix>(
        [&] { (void)matrix.at(0, std::numeric_limits<std::size_t>::max()); },
        "maximum column index must be rejected");
}

void test_model_errors()
{
    model::EuclidDistance distance;
    const ds::Matrix train = make_matrix(2, 1, {0.0, 1.0});
    const ds::Matrix empty_train = make_matrix(0, 1, {});
    const ds::Matrix wrong_features = make_matrix(1, 2, {0.0, 1.0});
    const std::vector<double> target{10.0, 20.0};

    check_throws<exc::invalid_k>(
        [&] { model::KnnRegressor model(0, distance, model::WeightingType::equal); },
        "zero k must be rejected");
    check_throws<exc::invalid_k>(
        [&] {
            model::KnnRegressor model(3, distance, model::WeightingType::equal);
            model.fit(train, target);
        },
        "k larger than the training sample must be rejected at fit");

    model::KnnRegressor too_large_k(3, distance, model::WeightingType::equal);
    check_throws<exc::empty_train_sample>(
        [&] { too_large_k.fit(empty_train, {}); },
        "empty training sample must be rejected");
    model::KnnRegressor target_size_check(1, distance, model::WeightingType::equal);
    check_throws<exc::invalid_target>(
        [&] { target_size_check.fit(train, std::span<const double>(target.data(), 1)); },
        "target with a wrong size must be rejected");

    model::KnnRegressor not_fitted(1, distance, model::WeightingType::equal);
    check_throws<exc::predict_before_fit>(
        [&] { (void)not_fitted.predict(train); },
        "prediction before fit must be rejected");

    model::KnnRegressor fitted(1, distance, model::WeightingType::equal);
    fitted.fit(train, target);
    check_throws<exc::invalid_prediction_sample>(
        [&] { (void)fitted.predict(wrong_features); },
        "prediction with a wrong feature count must be rejected");
    check(fitted.predict(make_matrix(0, 1, {})).empty(),
          "prediction on an empty sample must return an empty result");
}

void test_regressor()
{
    model::EuclidDistance distance;
    const ds::Matrix train = make_matrix(3, 1, {0.0, 1.0, 2.0});
    const ds::Matrix queries = make_matrix(2, 1, {0.0, 1.5});
    const std::vector<double> target{0.0, 10.0, 20.0};

    model::KnnRegressor equal(2, distance, model::WeightingType::equal);
    equal.fit(train, target);
    const auto equal_result = equal.predict(queries);
    check(equal_result.size() == 2, "regressor result size is incorrect");
    check_close(equal_result[0], 5.0, "equal-weight regression is incorrect");
    check_close(equal_result[1], 15.0, "equal-weight regression tie is incorrect");

    model::KnnRegressor reverse(2, distance, model::WeightingType::reverse);
    reverse.fit(train, target);
    const auto reverse_result = reverse.predict(queries);
    check_close(reverse_result[0], 0.0, "reverse weighting must handle an exact match");
    check_close(reverse_result[1], 15.0, "reverse-weight regression is incorrect");
}

void test_classifier()
{
    model::ManhattanDistance distance;
    const ds::Matrix train = make_matrix(4, 1, {0.0, 1.0, 3.0, 4.0});
    const ds::Matrix queries = make_matrix(2, 1, {0.5, 3.0});
    const std::vector<double> target{2.0, 1.0, 1.0, 2.0};

    model::KnnClassifier equal(2, distance, model::WeightingType::equal);
    equal.fit(train, target);
    const auto equal_result = equal.predict(queries);
    check_close(equal_result[0], 1.0, "classifier tie must choose the least category");
    check_close(equal_result[1], 1.0, "classifier exact-neighbour result is incorrect");

    model::KnnClassifier reverse(2, distance, model::WeightingType::reverse);
    reverse.fit(train, target);
    const auto reverse_result = reverse.predict(queries);
    check_close(reverse_result[0], 1.0, "reverse-weight classification is incorrect");
    check_close(reverse_result[1], 1.0, "reverse classification must handle exact matches");
}
} // namespace

int main()
{
    try
    {
        test_distances();
        test_matrix_bounds();
        test_model_errors();
        test_regressor();
        test_classifier();
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

    std::cout << "All Lab 2 tests passed.\n";
    return 0;
}
