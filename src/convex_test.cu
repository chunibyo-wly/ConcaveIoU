#ifndef TEST_CONVEX_CU_
#define TEST_CONVEX_CU_

#include <random>

#include "../include/convex_iou_cuda_kernel.cuh"
#include "../include/cuda_gtest_plugin.h"

CUDA_TEST(convex_iou, cross) {
    Point o(0, 0), a(1, 0), b(0, 1);
    float result = cross(o, a, b);
    ASSERT_FLOAT_EQ(1.f, result);
}

TEST(convex_iou, Jarvis) {
    Point p[100];
    p[0] = Point(0, 0), p[1] = Point(0, 1), p[2] = Point(1, 0),
    p[3] = Point(1, 1);
    for (size_t i = 4; i < 100; ++i) {
        p[i].x = (double)rand() / INT_MAX, p[i].y = (double)rand() / INT_MAX;
    }
    int n_poly = 100;
    Jarvis(p, n_poly);
    ASSERT_EQ(4, n_poly);
}

TEST(convex_iou, Jarvis_and_index) {
    int n_poly = 15;

    Point p[15], p_copy[15];
    p[0] = Point(0, 0), p[1] = Point(0, 1), p[2] = Point(1, 0),
    p[3] = Point(1, 1), p[14] = Point(0.5, 2);
    for (size_t i = 4; i < n_poly - 1; ++i) {
        p[i].x = (double)rand() / INT_MAX, p[i].y = (double)rand() / INT_MAX;
    }
    for (size_t i = 0; i < n_poly; ++i)
        p_copy[i].x = p[i].x, p_copy[i].y = p[i].y;

    int points_to_convex_ind[15];
    for (size_t i = 0; i < n_poly; ++i)
        points_to_convex_ind[i] = -1;
    Jarvis_and_index(p, n_poly, points_to_convex_ind);

    ASSERT_EQ(5, n_poly);
    for (size_t i = 0; i < n_poly; ++i) {
        ASSERT_FLOAT_EQ(p[i].x, p_copy[points_to_convex_ind[i]].x);
        ASSERT_FLOAT_EQ(p[i].y, p_copy[points_to_convex_ind[i]].y);
    }
}

#endif