#include <cassert>
#include <motifcl/motifcl.hpp>
#include "test_utils.hpp"

int main() {
    motifcl::Shape s{2, 3, 4};
    assert(s.ndim() == 3);
    assert(s.numel() == 24);
    assert(s[1] == 3);
    auto st = motifcl::contiguous_strides(s);
    assert(st[0] == 12 && st[1] == 4 && st[2] == 1);
}
