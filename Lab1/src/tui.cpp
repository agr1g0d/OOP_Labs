#include <iostream>
#include <filesystem>
#include <string>
#include "dataset.h"
using namespace std;

string CSV_PATH = "../data/Autism-Adult-Data.csv";

int main()
{
    std::cout << std::filesystem::current_path() << "\n\n";

    ds::Dataset dataset(CSV_PATH);
    dataset.print_info();
    return 0;
}
