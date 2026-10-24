#include "dataset.h"
#include "matrix.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
namespace fs = std::filesystem;

struct TestFailure : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

void check(bool condition, const std::string& message)
{
    if (!condition)
        throw TestFailure(message);
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

class TemporaryFile
{
public:
    explicit TemporaryFile(std::string suffix)
        : path(fs::temp_directory_path() /
               ("oop_labs_dataset_test_" + std::to_string(counter++) + suffix))
    {
    }

    ~TemporaryFile()
    {
        std::error_code error;
        fs::remove(path, error);
    }

    const fs::path& get() const { return path; }

private:
    inline static unsigned counter = 0;
    fs::path path;
};

void write_file(const fs::path& path, const std::string& contents)
{
    std::ofstream file(path);
    check(file.good(), "failed to create temporary test file");
    file << contents;
    check(file.good(), "failed to write temporary test file");
}

void check_same_matrix(const ds::Matrix& left, const ds::Matrix& right)
{
    check(left.n_rows == right.n_rows, "matrix row count changed after reading");
    check(left.n_cols == right.n_cols, "matrix column count changed after reading");
    check(left.cols == right.cols, "matrix column names changed after reading");
    check(left.data.size() == right.data.size(), "matrix data size changed after reading");
    for (std::size_t i = 0; i < left.data.size(); ++i)
        check(std::memcmp(&left.data[i], &right.data[i], sizeof(double)) == 0,
              "matrix value changed after reading");
}

void test_invalid_paths(const fs::path& csv_path)
{
    check_throws<std::runtime_error>(
        [] { ds::Dataset dataset("/path/that/does/not/exist.csv"); },
        "missing CSV must be rejected");
    check_throws<std::runtime_error>(
        [] { ds::load_binary_matrix("/path/that/does/not/exist.bin"); },
        "missing binary file must be rejected");

    TemporaryFile empty(".csv");
    write_file(empty.get(), "");
    check_throws<std::runtime_error>(
        [&] { ds::Dataset dataset(empty.get()); },
        "empty CSV must be rejected");

    TemporaryFile header_only(".csv");
    write_file(header_only.get(), "a,b\n");
    check_throws<std::runtime_error>(
        [&] { ds::Dataset dataset(header_only.get()); },
        "CSV without records must be rejected");

    TemporaryFile too_many(".csv");
    write_file(too_many.get(), "a,b\n1,2,3\n");
    check_throws<std::runtime_error>(
        [&] { ds::Dataset dataset(too_many.get()); },
        "row with too many fields must be rejected");

    TemporaryFile too_few(".csv");
    write_file(too_few.get(), "a,b\n1\n");
    check_throws<std::runtime_error>(
        [&] { ds::Dataset dataset(too_few.get()); },
        "row with too few fields must be rejected");

    TemporaryFile uneven(".csv");
    write_file(uneven.get(), "a,b\n1,2\n3\n");
    check_throws<std::runtime_error>(
        [&] { ds::Dataset dataset(uneven.get()); },
        "CSV with uneven columns must be rejected");

    ds::Dataset dataset(csv_path);
    dataset.fit();
    check_throws<std::out_of_range>(
        [&] { (void)dataset["not a feature"]; },
        "unknown feature name must be rejected");
    check_throws<std::out_of_range>(
        [&] { (void)dataset[-1]; },
        "negative feature index must be rejected");
    check_throws<std::out_of_range>(
        [&] { (void)dataset[dataset.size()]; },
        "feature index past the end must be rejected");
    check_throws<std::runtime_error>(
        [&] { dataset.save_binary_matrix("/path/that/does/not/exist/output.bin"); },
        "binary output to an unavailable path must be rejected");

    TemporaryFile truncated(".bin");
    write_file(truncated.get(), "short");
    check_throws<std::runtime_error>(
        [&] { (void)ds::load_binary_matrix(truncated.get()); },
        "truncated binary file must be rejected");
}

void test_binary_round_trip(const fs::path& csv_path)
{
    ds::Dataset dataset(csv_path);
    dataset.fit();
    TemporaryFile binary(".bin");
    dataset.save_binary_matrix(binary.get());

    const ds::Matrix first = ds::load_binary_matrix(binary.get());
    const ds::Matrix second = ds::load_binary_matrix(binary.get());

    check(first.n_rows == 704, "unexpected number of dataset rows");
    check(first.n_cols > 0, "encoded matrix must have columns");
    check(first.cols.size() == first.n_cols, "matrix names do not match column count");
    check(first.data.size() == first.n_rows * first.n_cols,
          "matrix data does not match its dimensions");
    for (const auto& value : first.data)
        check(std::isfinite(value), "matrix contains a non-finite value");
    check_same_matrix(first, second);
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: dataset_tests <csv path>\n";
        return 2;
    }

    try
    {
        test_invalid_paths(argv[1]);
        test_binary_round_trip(argv[1]);
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

    std::cout << "All dataset tests passed.\n";
    return 0;
}
