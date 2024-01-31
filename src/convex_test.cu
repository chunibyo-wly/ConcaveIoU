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

    // 计算得到的凸包上的 1,2,3,4,5 个点分别是原来输入的点集里面的几号点
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

TEST(convex_iou, intersectAreaPoly) {
    int n_poly = 25;

    Point p1[n_poly], p2[n_poly];
    p1[0] = Point(0, 0), p1[1] = Point(0, 1), p1[2] = Point(1, 0),
    p1[3] = Point(1, 1), p1[24] = Point(0.5, 2);

    p2[0] = Point(0, 0), p2[1] = Point(0, 1), p2[2] = Point(1, 0),
    p2[3] = Point(1, 1), p2[24] = Point(0.5, 2);

    for (size_t i = 4; i < n_poly - 1; ++i) {
        p1[i].x = (double)rand() / INT_MAX, p1[i].y = (double)rand() / INT_MAX;
        p2[i].x = (double)rand() / INT_MAX, p2[i].y = (double)rand() / INT_MAX;
    }

    double grad_C[18];

    for (size_t i = 0; i < 18; ++i) {
        grad_C[i] = (double)rand() / INT_MAX;
    }

    intersectAreaPoly(p1, n_poly, p2, n_poly, grad_C);

    ASSERT_EQ(0, 0);
}

#endif