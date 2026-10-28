#include "dataset.h"
#include "distance.h"

namespace model
{



inline int K;
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

class DecisionStump : public IModel
{
    SplitCriterion criterion;
    std::size_t min_samples_leaf;

    std::size_t best_j;
    std::size_t best_t;
    double left_prefoction;
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

    std::size_t split_feature() const;

    double split_threshold() const;

};

}