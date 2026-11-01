#include "dataset.h"
#include "distance.h"

namespace model
{



inline int K;
inline std::size_t MIN_SAMPLES_LEAF;
inline WeightingType WEIGHTING;
inline DistanceType DISTANCE;

class IModel
{
public:
    virtual ~IModel() = default;

    virtual void fit(
        const ds::Matrix& features,
        std::span<const double> target
    ) = 0;

    virtual std::vector<double> predict(
        const ds::Matrix& features
    ) const = 0;
};

class Knn : public IModel
{
protected:
    ds::Matrix train_features;
    std::vector<double> train_target;
    const std::size_t k;
    const IDistance& dist;
    const WeightingType weighting;

public:
    Knn(std::size_t k, IDistance& dist, 
        WeightingType weighting
    );

    void fit(
        const ds::Matrix& features,
        std::span<const double> target
    ) override;

    std::vector<double> predict(
        const ds::Matrix& features
    ) const = 0;
};

class KnnClassifier : public Knn
{
public:
    KnnClassifier(std::size_t k, IDistance& dist, WeightingType weighting)
        : Knn(k, dist, weighting) {}

    std::vector<double> predict(
        const ds::Matrix& features
    ) const override;
};

class KnnRegressor : public Knn
{
public:
    KnnRegressor(std::size_t k, IDistance& dist, WeightingType weighting)
        : Knn(k, dist, weighting) {}

    std::vector<double> predict(
        const ds::Matrix& features
    ) const override;
};

enum class SplitCriterion { Variance, Gini };

struct Agregates
{
    std::size_t m;
    std::size_t m_L;
    std::span<const double> target;

    Agregates(std::span<const double> target);
    virtual ~Agregates() = default;
    virtual void reset() = 0;
    virtual double score() const = 0;
    virtual void move(std::size_t i) = 0;
};

struct AgregatesVariance : public Agregates
{
    double S;
    double S_L;
    AgregatesVariance(std::span<const double> target);
    virtual void reset() override;
    virtual double score() const override;
    virtual void move(std::size_t i) override;
};

struct AgregatesGini : public Agregates
{
    double A;
    double A_L;
    std::unordered_map<double, std::size_t> nc;
    std::unordered_map<double, std::size_t> nc_L;
    AgregatesGini(std::span<const double> target);
    virtual void reset() override;
    virtual double score() const override;
    virtual void move(std::size_t i) override;
};

class DecisionStump : public IModel
{
protected:
    ds::Matrix train_features;
    std::vector<double> train_target;

    SplitCriterion criterion;
    std::size_t min_samples_leaf;

    std::size_t best_j;
    double best_t;
    double left_prediction;
    double right_prediction;
public:
    DecisionStump(SplitCriterion criterion, 
        std::size_t min_samples_leaf);

    void fit(
        const ds::Matrix& features,
        std::span<const double> target
    ) override;

    std::vector<double> predict(
        const ds::Matrix& features
    ) const override;

    inline std::size_t split_feature() const
    { return best_j; }

    inline double split_threshold() const
    { return best_t; }
};


}