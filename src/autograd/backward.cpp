#include <motifcl/autograd/backward.hpp>

namespace motifcl::autograd {

void backward(Tensor loss) { loss.backward(); }

} // namespace motifcl::autograd
