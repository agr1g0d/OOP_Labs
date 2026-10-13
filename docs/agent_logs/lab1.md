# Журнал работы с агентом — лабораторная работа 1

## Эпизод 1. Настройка CMake

### Запрос

> Составь `CMakeLists.txt` так, чтобы из любых `.cpp`-файлов можно было
> включать заголовочные файлы без указания пути. `.cpp`-файлы лежат в папках
> `LabN/src/`, а `.h`-файлы — в `LabN/headers/` (в данный момент есть только
> `Lab1/`, но в дальнейшем появятся `Lab2`, `Lab3` и т. д.). Пока что `main()`
> находится в `Lab1/src/tui.cpp`.

### Результат

Агент сгенерировал `CMakeLists.txt` с настройкой каталогов исходных файлов и
заголовков.

### Решение

Результат принят и использован в проекте.

### Проверка

Сборка проекта выполнена через CMake.

## Эпизод 2. Черновой текстовый интерфейс

### Запрос

> Прочитай файл условий лабораторной и сделай простой текстовый пользовательский
> интерфейс в `main()` — например, через чтение символа с клавиатуры и
> последующую обработку через `switch-case`. Нужны функции для вывода сводки,
> вывода статистики и выхода с предобработкой. Сам функционал не реализуй:
> оставь только интерфейс и места, куда я потом помещу реализацию.

### Результат

Агент предложил меню и заготовки обработчиков:

```cpp
void print_menu()
{
    std::cout << "\n=== Меню ===\n"
              << "1 - Вывести сводку по датасету\n"
              << "2 - Вывести статистику по признаку\n"
              << "3 - Выполнить предобработку и выйти\n"
              << "0 - Выйти без предобработки\n"
              << "Выберите действие: ";
}

void print_statistics_placeholder()
{
    std::cout << "\n[Место для реализации статистики по выбранному признаку]\n";
}

void preprocess_and_exit_placeholder()
{
    std::cout << "\n[Место для реализации предобработки и сохранения датасета]\n";
}

int main()
{
    try
    {
        ds::Dataset dataset(CSV_PATH);
        bool running = true;

        while (running)
        {
            print_menu();

            char action;
            if (!(std::cin >> action))
            {
                std::cout << "\nВвод завершён.\n";
                break;
            }

            switch (action)
            {
                case '1':
                    dataset.print_info();
                    break;
                case '2':
                    print_statistics_placeholder();
                    break;
                case '3':
                    preprocess_and_exit_placeholder();
                    running = false;
                    break;
                case '0':
                    std::cout << "\nВыход без предобработки.\n";
                    running = false;
                    break;
                default:
                    std::cout << "\nНеизвестная команда. Выберите пункт меню.\n";
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
```

### Правки

- Меню переделано на английский язык.
- Заготовки оставлены до реализации соответствующего функционала.

### Проверка

Проверена логика переходов между пунктами меню и выхода из программы.

## Эпизод 3. Вывод сводной информации о датасете

### Запрос

> В методе `print_info()` файла `dataset.cpp` допиши функционал вывода
> информации о датасете согласно третьему пункту условий лабораторной работы 1.

### Результат

Агент предложил вывод количества объектов и признаков, таблицу признаков с их
типами и количеством пропусков, а также итоговое количество пропусков.

### Правки

Вывод переведён на английский язык.

### Проверка

Проверены сортировка признаков, подсчёт пропусков и вывод процента пропущенных
значений.

## Эпизод 4. Статистика по признаку и гистограммы

### Запрос

> Реализуй вывод статистики по признаку в методе `print_stat_by_i(int i)`.
> Для числового признака выведи количество пропусков, минимум, максимум,
> среднее, дисперсию, квантили и гистограмму. Для категориального признака
> выведи количество пропусков, категории, их частоты и гистограмму.

### Результат

Агент предложил заготовку вывода для числовых и категориальных признаков:

```cpp
const Column& col = operator[](i);

if (std::holds_alternative<NumericColumn>(col))
{
    int missing;
    double min, max, mean, var;
    double q05, q25, q50, q75, q95;

    std::cout << "\nFeature " << i << ": " << features_sorted[i]
              << " (numeric)\n\n"
              << "  missing: " << missing << '\n'
              << "  min: " << min << "    max: " << max << '\n'
              << "  mean: " << mean << "   var: " << var << "\n\n"
              << "  q05: " << q05 << '\n'
              << "  q25: " << q25 << '\n'
              << "  q50: " << q50 << '\n'
              << "  q75: " << q75 << '\n'
              << "  q95: " << q95 << "\n\n";

    // Границы интервалов и частоты рассчитываются вызывающим кодом.
    std::vector<std::pair<std::string, int>> histogram;
    for (const auto& [interval, frequency] : histogram)
    {
        std::cout << "  " << std::left << std::setw(18) << interval
                  << frequency << '\n';
    }
}
else
{
    int missing;
    std::vector<std::pair<std::string, int>> categories;

    std::cout << "\nFeature " << i << ": " << features_sorted[i]
              << " (categorical)\n\n"
              << "  missing: " << missing << "\n\n"
              << "  Categories:\n";

    for (const auto& [category, frequency] : categories)
    {
        std::cout << "    " << std::left << std::setw(24) << category
                  << frequency << '\n';
    }

    std::cout << "\n  Histogram:\n";
    for (const auto& [category, frequency] : categories)
    {
        std::cout << "    " << std::left << std::setw(24) << category
                  << std::string(frequency, '#') << "  "
                  << frequency << '\n';
    }
}
```

### Правки

- Вывод гистограмм переделан.
- Исправлено обозначение категориального признака: `categorical`.

### Проверка

Проверен формат вывода статистики для числовых и категориальных признаков.

## Эпизод 5. Тесты

### Запрос

> Напиши тесты для проекта. Папка с тестами должна быть `./tests`, тесты должны
> запускаться через CMake. Тесты должны проверять некорректные вводы везде, где
> это возможно.

### Результат

Агент добавил файл [dataset_tests.cpp](../../tests/dataset_tests.cpp) и изменил
`CMakeLists.txt`, подключив тесты к CTest:

```cmake
enable_testing()

add_executable(dataset_tests
    tests/dataset_tests.cpp
    Lab1/src/dataset.cpp
)

target_include_directories(dataset_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/Lab1/headers"
)

add_test(
    NAME dataset_tests
    COMMAND dataset_tests
            "${CMAKE_CURRENT_SOURCE_DIR}/data/Autism-Adult-Data.csv"
)
```

### Решение

Тесты приняты и подключены к сборке проекта.

### Проверка

Тесты запускаются через CTest.
