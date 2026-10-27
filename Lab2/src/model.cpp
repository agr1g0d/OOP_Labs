#include <queue>
#include <unordered_map>
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
        throw exc::empty_train_sample("Train sample is empty.");
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

}