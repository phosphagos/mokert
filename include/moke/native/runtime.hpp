#pragma once
#include "moke/runtime/common.hpp"

namespace moke {
template <class Status>
void check_status(Status status);

void sync_device();
} // namespace moke
