
#include <time.h>
#include <unistd.h>
#include <dmlc/logging.h>
#include <cstdio>
#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <vector>

#include <mxnet/engine.h>
#include "../src/engine/engine_impl.h"
#include <dmlc/timer.h>

/**
 * present the following workload
 *  n = reads.size()
 *  data[write] = (data[reads[0]] + ... data[reads[n]]) / n
 *  std::this_thread::sleep_for(std::chrono::microsecons(time));
 */
struct Workload {