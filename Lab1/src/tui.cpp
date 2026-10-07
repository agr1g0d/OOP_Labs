#include <iostream>
#include <exception>
#include <limits>
#include <string>
#include "dataset.h"

namespace
{
constexpr const char* CSV_PATH = "../data/Autism-Adult-Data.csv";

void print_menu()
{
    std::cout << "\n=== Menu ===\n"
              << "1 - Print dataset summary\n"
              << "2 - Print feature statistics\n"
              << "3 - Preprocess and exit\n"
              << "0 - Exit without preprocessing\n"
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
        dataset.print_stat_by_i(i);

    }
}

void preprocess_and_exit_placeholder()
{
    std::cout << "\n[Place for preprocessing and dataset saving implementation]\n";
}
} // namespace

int main()
{
    try {
        ds::Dataset dataset(CSV_PATH);
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
                    preprocess_and_exit_placeholder();
                    running = false;
                    break;
                case '0':
                    std::cout << "\nExiting without preprocessing.\n";
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
