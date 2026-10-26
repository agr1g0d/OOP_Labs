#include <stdexcept>

namespace exc
{

struct incompatible_vectors : std::invalid_argument
{
    using std::invalid_argument::invalid_argument;
};

struct invalid_k : std::invalid_argument
{
    using std::invalid_argument::invalid_argument;
};

struct predict_before_fit : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

struct empty_train_sample : std::invalid_argument
{
    using std::invalid_argument::invalid_argument;
};

struct invalid_prediction_sample : std::invalid_argument
{
    using std::invalid_argument::invalid_argument;
};

struct out_of_matrix : std::out_of_range
{
    using std::out_of_range::out_of_range;
};

struct invalid_target : std::invalid_argument
{
    using std::invalid_argument::invalid_argument;
};

    
}
