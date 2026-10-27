#include <span>

namespace model
{

enum class WeightingType
{
    equal,
    reverse
};
enum class DistanceType
{
    euclid,
    manhattan,
    chebyshev
};

class IDistance
{
public:
    virtual ~IDistance() = default;

    virtual double compute(
        std::span<const double> a,
        std::span<const double> b
    ) const = 0;
};

class EuclidDistance : public IDistance
{
public:
    double compute(
        std::span<const double> a,
        std::span<const double> b
    ) const override;
};

class ManhattanDistance : public IDistance
{
public:
    double compute(
        std::span<const double> a,
        std::span<const double> b
    ) const override;
};

class ChebyshevDistance : public IDistance
{
public:
    double compute(
        std::span<const double> a,
        std::span<const double> b
    ) const override;
};


}