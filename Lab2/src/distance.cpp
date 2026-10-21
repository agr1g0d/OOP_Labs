#include <vector>
#include <algorithm>
#include <iterator>
#include <cmath>
#include "distance.h"
#include "exc.h"

namespace model
{

double EuclidDistance::compute(
        std::span<const double> a,
        std::span<const double> b
    ) const
{
    if (a.size() != b.size()) throw exc::incompatible_vectors("Cannot compute distance of vectors of different dimensions.");
    double dist = 0;

    /* std::vector<std::pair<const double, const double>> ab(a.size());
    std::transform(a.begin(), a.end(), b.begin(),
        ab.begin(), [](auto ai, auto bi) 
            { return std::make_pair(ai, bi); } );   мне бы хотелось так, но увы 
                                                    думаю это не оптимально (два цикла вместо одного)
    for (auto [ai, bi] : ab)
        dist += pow(ai - bi, 2);
    dist = sqrt(dist);
    */
    for (int i = 0; i < a.size(); ++i)
        dist += pow(a[i] - b[i], 2);
    dist = sqrt(dist);
    return dist;
}

double ManhattanDistance::compute(
    std::span<const double> a,
    std::span<const double> b
) const
{
    if (a.size() != b.size()) throw exc::incompatible_vectors("Cannot compute distance of vectors of different dimensions.");
    double dist = 0;
    for (int i = 0; i < a.size(); ++i)
        dist += abs(a[i] - b[i]);
}

double ChebyshevDistance::compute(
    std::span<const double> a,
    std::span<const double> b
) const
{
    if (a.size() != b.size()) throw exc::incompatible_vectors("Cannot compute distance of vectors of different dimensions.");
    double dist = 0;
    for (int i = 0; i < a.size(); ++i)
        if (dist < abs(a[i] - b[i]))
            dist = abs(a[i] - b[i]);
    return dist;
}

}

