#include <queue>
#include <unordered_map>
#include <memory>
#include <numeric>
#include "model.h"
#include "exc.h"

namespace model
{

Knn::Knn(std::size_t k, IDistance& dist, 
        WeightingType weighting
    ) :  k(k), dist(dist), weighting(weighting) 
{
    if (k < 1) throw exc::invalid_k("k is less than 1.");
}

void Knn::fit(
        const ds::Matrix& features,
        std::span<const double> target
    )
{
    // todo: прочитать про расрутку стека
    if (features.n_rows == 0)
        throw exc::invalid_train_sample("Train sample is empty.");
    if (k > features.n_rows)
        throw exc::invalid_k("k is greater than train sample.");
    if (features.n_rows  != target.size()) 
        throw exc::invalid_target("Target column is greater/less than features.");

    train_features = features;
    if (!train_target.empty())
        train_target.clear();
    train_target.reserve(target.size());
    for (const auto& t : target)
        train_target.emplace_back(t);
}

auto max_disance = [](
            const std::pair<double, std::size_t>& left,
            const std::pair<double, std::size_t>& right
) {
    if (left.first == right.first)
        return left.second < right.second;
    return (left.first < right.first); 
};

std::vector<double> KnnRegressor::predict(
        const ds::Matrix& features
    ) const
{
    if (train_target.empty())
        throw exc::predict_before_fit("predict method was called before fit.");
    if (features.n_cols != train_features.n_cols)
        throw exc::invalid_prediction_sample("prediction sample is incompitable with train sample.");

    std::vector<double> answers;
    answers.reserve(features.n_rows);
    for (std::size_t i = 0; i < features.n_rows; ++i)
    {
        std::priority_queue
        <
            std::pair<double, std::size_t>,
            std::vector<std::pair<double, std::size_t>>,
            decltype(max_disance)
        > neighbours;

        std::span<const double> row = features[i];

        for (std::size_t j = 0; j < train_features.n_rows; ++j)
        {
            std::span<const double> train_row = train_features[j];
            double distance = dist.compute(row, train_row);
            if (neighbours.size() != k)
                neighbours.push(std::make_pair(distance, j));
            else if (distance < neighbours.top().first)
            {
                neighbours.pop();
                neighbours.push(std::make_pair(distance, j));
            }
        }

        double result = 0;
        switch(weighting)
        {
            case WeightingType::equal:
                for (std::size_t j = 0; neighbours.size(); ++j)
                {
                    auto [_, ind] = neighbours.top();
                    neighbours.pop();
                    result += train_target[ind];
                }
                result /= k;
                break;
            case WeightingType::reverse: {
                double w = 0;
                bool coincidence = 0;
                for (std::size_t j = 0; neighbours.size(); ++j)
                {
                    auto [d, ind] = neighbours.top();
                    if (d)
                    {
                        result += train_target[ind] / d;
                        w += 1/d;
                    }
                    else
                    {
                        result = train_target[ind];
                        coincidence = 1;
                        break;
                    }
                    neighbours.pop();
                }
                if(!coincidence) result /= w;
                break;
            } default:
                throw std::invalid_argument("unsupported weighting type.");
        }
        answers.push_back(result);
    }

    return answers;
}

std::vector<double> KnnClassifier::predict(
        const ds::Matrix& features
    ) const
{
    if (train_target.empty())
        throw exc::predict_before_fit("predict method was called before fit.");
    if (features.n_cols != train_features.n_cols)
        throw exc::invalid_prediction_sample("prediction sample is incompitable with train sample.");

    std::vector<double> answers;
    answers.reserve(features.n_rows);
    for (std::size_t i = 0; i < features.n_rows; ++i)
    {
        std::priority_queue
        <
            std::pair<double, std::size_t>,
            std::vector<std::pair<double, std::size_t>>,
            decltype(max_disance)
        > neighbours;

        std::span<const double> row = features[i];

        for (std::size_t j = 0; j < train_features.n_rows; ++j)
        {
            std::span<const double> train_row = train_features[j];
            double distance = dist.compute(row, train_row);
            if (neighbours.size() != k)
                neighbours.push(std::make_pair(distance, j));
            else if (distance < neighbours.top().first)
            {
                neighbours.pop();
                neighbours.push(std::make_pair(distance, j));
            }
        }

        std::unordered_map<double, double> voting; // category-votes
        double winner;
        switch(weighting)
        {
            case WeightingType::equal:
                for (std::size_t j = 0; neighbours.size(); ++j)
                {
                    auto [_, ind] = neighbours.top();
                    voting[train_target[ind]] += 1;
                    neighbours.pop();
                }
                break;
            case WeightingType::reverse: {
                bool coincidence = 0;
                for (std::size_t j = 0; neighbours.size(); ++j)
                {
                    auto [d, ind] = neighbours.top();
                    if (d == 0.0)
                    {
                        if (!coincidence)
                        {
                            voting.clear();
                            coincidence = 1;
                        }
                        voting[train_target[ind]] += 1;
                    } else if (!coincidence)
                        voting[train_target[ind]] += (1 / d);
                    neighbours.pop();
                }
                break;
            } default:
                throw std::invalid_argument("unsupported weighting type.");
        }
        if (voting.empty()) throw std::runtime_error("voting system is broken.");
        winner = voting.begin()->first;
        for (auto [cat, vote] : voting)
            if (vote > voting[winner])
                winner = cat;
        for (auto [cat, vote] : voting)  // if some cat has save votes we choose the one with the least cat code
            if (vote == voting[winner] && cat < winner)
                winner = cat;
        answers.push_back(winner);
    }

    return answers;
}

Agregates::Agregates(std::span<const double> target) 
    : target(target)
{
    m = target.size();
    m_L = 0;
}
AgregatesVariance::AgregatesVariance(std::span<const double> target)
    : Agregates(target)
{
    S = 0;
    for (double d : target)
        S += d;
}
AgregatesGini::AgregatesGini(std::span<const double> target)
    : Agregates(target)
{
    A = 0.0;
    for (double d : target)
        ++nc[d];
    for (auto [c, n] : nc)
        A += n*n;
}
void AgregatesVariance::reset()
{
    S_L = 0;
    m_L = 0;
}
void AgregatesGini::reset()
{
    for (auto& [c, n] : nc_L)
        n = 0;
    A_L = 0.0;
    m_L = 0;
}
void AgregatesVariance::move(std::size_t i)
{
    S_L += target[i];
    m_L++;
}
void AgregatesGini::move(std::size_t i)
{
    double c = target[i];
    A_L += 1.0 + 2.0*nc_L[c];
    nc_L[c]++;
    m_L++;
}
double AgregatesVariance::score() const
{
    double S_R = S - S_L;
    std::size_t m_R = m - m_L;
    return S_L*S_L/m_L + S_R*S_R/m_R;
}
double AgregatesGini::score() const
{
    double A_R = 0.0;
    std::size_t m_R = m - m_L;
    for (auto [c, n] : nc)
    {
        const auto left_count = nc_L.find(c);
        const std::size_t n_L =
            left_count == nc_L.end() ? 0 : left_count->second;
        const double difference = static_cast<double>(n) - n_L;
        A_R += difference * difference;
    }
    return A_L/m_L + A_R/m_R;
}

DecisionStump::DecisionStump(SplitCriterion criterion, 
        std::size_t min_samples_leaf)
    : criterion(criterion), min_samples_leaf(min_samples_leaf)
    {
        if (min_samples_leaf < 1)
            throw exc::invalid_min_samples_leaf(
                "Sample leaf cant be less than one."
            );
    }

void DecisionStump::fit(const ds::Matrix& features,
        std::span<const double> target)
{
    if (features.n_rows == 0)
        throw exc::invalid_train_sample("Train sample is empty.");
    if (features.n_rows == 1)
        throw exc::invalid_train_sample("Train sample has only one object. Split is impossible.");
    if (features.n_rows  != target.size()) 
        throw exc::invalid_target("Target column is greater/less than features.");
    if (min_samples_leaf > target.size() / 2)
        throw exc::invalid_min_samples_leaf("Train sample split is impossible.");
    train_features = features;
    if (!train_target.empty())
        train_target.clear();
    train_target.reserve(target.size());
    for (const auto& t : target)
        train_target.emplace_back(t);

    std::unique_ptr<Agregates> agr;
    switch (criterion)
    {
        case SplitCriterion::Variance:
            agr = std::make_unique<AgregatesVariance>(train_target);
            break;
        case SplitCriterion::Gini: 
            agr = std::make_unique<AgregatesGini>(train_target);
            break;
        default: throw
            std::invalid_argument("unsupported split criterion.");
    }

    double best_score = -1.0;

    std::vector<size_t> order; 
    order.reserve(train_features.n_rows);
    for (std::size_t i = 0; i < train_features.n_rows; ++i) 
        order.push_back(i);

    for (std::size_t j = 0; j < train_features.n_cols; ++j)
    {
        std::sort(order.begin(), order.end(),
        [this, j](std::size_t l, std::size_t r)
        {
            return 
                this->train_features[l][j] < this->train_features[r][j];
        });
        agr->reset();
        for (std::size_t p = 0; p < train_features.n_rows-1; ++p)
        {
            std::size_t i = order[p];
            agr->move(i);
            if (train_features[order[p]][j] 
                    == train_features[order[p+1]][j])
                continue;
            if (agr->m_L < min_samples_leaf )
                continue;
            if ((agr->m - agr->m_L) < min_samples_leaf)
                break;
            double score = agr->score();
            if (score > best_score)
            {
                best_score = score;
                best_j = j;
                best_t = std::midpoint(
                    train_features[order[p]][j],
                    train_features[order[p+1]][j]
                );
            } // if scores are equal we choose the one with the least j
        }
    }
    if (best_score == -1.0)
        throw exc::invalid_train_sample("All features have constant value.");

    std::sort(order.begin(), order.end(),
    [this](std::size_t l, std::size_t r)
    {
        return 
            this->train_features[l][this->best_j] 
            < this->train_features[r][this->best_j];
    });
    left_prediction = right_prediction = 0;

    switch (criterion)
    {
        case SplitCriterion::Variance:
        {
            std::size_t p = 0;
            for (;train_features[order[p]][best_j] <= best_t; ++p)
                left_prediction += train_target[order[p]];
            left_prediction /= p;
            std::size_t q = p;
            for (;q != order.size(); ++q)
                right_prediction += train_target[order[q]];
            right_prediction /= (q-p);

            break;
        }
        case SplitCriterion::Gini: 
        {
            std::unordered_map<double, std::size_t> voting;
            std::size_t p = 0;
            for (;train_features[order[p]][best_j] <= best_t; ++p)
            {
                double c = train_target[order[p]];
                ++voting[c];
            }
            left_prediction = voting.begin()->first;
            for (auto [c, vote] : voting)
                if (vote > voting[left_prediction])
                    left_prediction = c;
            for (auto [c, vote] : voting)  // if some cat has save votes we choose the one with the least cat code
                if (vote == voting[left_prediction] && c < left_prediction)
                    left_prediction = c;
            voting.clear();
            std::size_t q = p;
            for (;q != order.size(); ++q)
            {
                double c = train_target[order[q]];
                ++voting[c];
            }
            right_prediction = voting.begin()->first;
            for (auto [c, vote] : voting)
                if (vote > voting[right_prediction])
                    right_prediction = c;
            for (auto [c, vote] : voting)  // if some cat has save votes we choose the one with the least cat code
                if (vote == voting[right_prediction] && c < right_prediction)
                    right_prediction = c;
            break;
        }
        default: throw
            std::invalid_argument("unsupported split criterion.");
    }
}

std::vector<double> DecisionStump::predict(
        const ds::Matrix& features
    ) const
{
    if (train_target.empty())
        throw exc::predict_before_fit("Predict method was called before fit.");
    if (features.n_cols != train_features.n_cols)
        throw exc::invalid_prediction_sample("Prediction sample is incompitable with train sample.");

    std::vector<double> result;
    result.reserve(features.n_rows);

    for (std::size_t i = 0; i < features.n_rows; ++i)
        if (features[i][best_j] <= best_t)    
            result.push_back(left_prediction);
        else
            result.push_back(right_prediction);
    return result;
}

}