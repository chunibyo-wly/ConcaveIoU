#ifndef TEST_CONCAVE_CU_
#define TEST_CONCAVE_CU_

#include <random>

#include "../include/concave_iou_cuda_kernel.cuh"
#include "../include/cuda_gtest_plugin.h"

CUDA_TEST(concave_iou, fast_mod) {
    int mod = fast_mod(10, 6);
    ASSERT_EQ(mod, 4);
}

CUDA_TEST(concave_iou, sum) {
    double a[5] = {0.123, 1.2, 3.166, 0.111, 2.1};
    double result = sum(a, 5);
    ASSERT_FLOAT_EQ(6.7, result);
}

CUDA_TEST(concave_iou, circumradius) {
    double result = circumradius(0., 0., 0., 1., 1., 0.);
    ASSERT_FLOAT_EQ(result, 0.5);

    ASSERT_FLOAT_EQ(circumradius(0., 0., -1., 0., 1., 0.),
                    std::numeric_limits<double>::max());
}

CUDA_TEST(concave_iou, circumcenter) {
    double x, y;
    circumcenter(0., 0., 0., 1., 1., 0., x, y);
    ASSERT_FLOAT_EQ(x, 0.5);
    ASSERT_FLOAT_EQ(y, 0.5);
}

CUDA_TEST(concave_iou, check_pts_equal) {
    ASSERT_EQ(
        check_pts_equal(1. + std::numeric_limits<double>::epsilon(), 0., 1, 0.),
        true);
    ASSERT_EQ(check_pts_equal(1. + 2 * std::numeric_limits<double>::epsilon(),
                              0., 1, 0.),
              false);
}

#endif