#ifndef TEST_CONVEX_CU_
#define TEST_CONVEX_CU_

#include "../include/cuda_gtest_plugin.h"
#include "../include/convex_iou_cuda_kernel.cuh"

CUDA_TEST(convex_iou, cross)
{
    Point o(0, 0), a(1, 0), b(0, 1);
    float result = cross(o, a, b);
    ASSERT_FLOAT_EQ(1.f, result);
}

#endif