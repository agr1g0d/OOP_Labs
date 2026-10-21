#include <stdexcept>

namespace exc
{

struct incompatible_vectors : std::invalid_argument
{
    using std::invalid_argument::invalid_argument;
};

struct out_of_matrix : std::out_of_range
{
    using std::out_of_range::out_of_range;
};
    
}
