#ifndef TEST_CONCAVE_CU_
#define TEST_CONCAVE_CU_

#include <filesystem>
#include <iostream>
#include <random>
#include <stdio.h>
#include <vector>

#include "../include/concave_iou_cuda_kernel.cuh"
#include "../include/cuda_gtest_plugin.h"
#include "../include/delaunator.hpp"

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

CUDA_TEST(concave_iou, While) {
    int x = 0, y = 0;
    while (x = 0, y++ < 20) {
    }
    ASSERT_EQ(y, 21);
}

HOST_DEVICE_INLINE bool compare(std::size_t i, std::size_t j) { return i < j; }

CUDA_TEST(concave_iou, bubble_sort) {
    std::size_t ids[10];
    int n = 10;
    for (int i = 0; i < n; ++i)
        ids[i] = n - i;

    for (std::size_t i = 0; i < n; i++) {
        for (std::size_t j = 0; j < n - i - 1; j++) {
            if (compare(ids[j + 1], ids[j]))
                swap_size_t(ids[j], ids[j + 1]);
        }
    }
    for (int i = 0; i < n - 1; ++i) {
        ASSERT_EQ(ids[i] < ids[i + 1], true);
    }
}

CUDA_TEST(concave_iou, orient) {
    ASSERT_EQ(orient(0., 0., 0., 1., 1., 0.), false);
    ASSERT_EQ(orient(0., 0., 1., 0., 0., 1.), true);
}

inline void validate(const std::vector<double> &coords) {
    delaunator::Delaunator d(coords);

    // validate halfedges
    for (std::size_t i = 0; i < d.halfedges.size(); i++) {
        const auto i2 = d.halfedges[i];
        GTEST_ASSERT_EQ(!static_cast<bool>((i2 != delaunator::INVALID_INDEX) &&
                                           (d.halfedges[i2] != i)),
                        true);
    }

    // validate triangulation
    double hull_area = d.get_hull_area();
    std::vector<double> triangles_areas;

    for (size_t i = 0; i < d.triangles.size(); i += 3) {
        const double ax = coords[2 * d.triangles[i]];
        const double ay = coords[2 * d.triangles[i] + 1];
        const double bx = coords[2 * d.triangles[i + 1]];
        const double by = coords[2 * d.triangles[i + 1] + 1];
        const double cx = coords[2 * d.triangles[i + 2]];
        const double cy = coords[2 * d.triangles[i + 2] + 1];
        triangles_areas.push_back(
            std::fabs((by - ay) * (cx - bx) - (bx - ax) * (cy - by)));
    }
    double triangles_area = delaunator::sum(triangles_areas);
    EXPECT_FLOAT_EQ(triangles_area, hull_area);
}

const int MAX_J = 500;

CUDA_TEST(concave_iou, Delaunator1) {
    std::size_t number = 14;
    double cudaArray[MAX_J] = {516, 661, 426, 539, 273, 525, 204,
                               694, 747, 750, 369, 793, 454, 390};

    // TODO: code duplication

    Delaunator d(cudaArray, number);
    for (std::size_t i = 0; i < d.halfedges_size; ++i) {
        const auto i2 = d.halfedges[i];
        ASSERT_EQ(
            !static_cast<bool>((i2 != INVALID_INDEX) && (d.halfedges[i2] != i)),
            true);
    }

    // validate triangulation
    double hull_area = d.get_hull_area();
    double *triangles_areas = new double[d.triangles_size];
    std::size_t triangles_areas_size = 0;
    for (std::size_t i = 0; i < d.triangles_size; i += 3) {
        const double ax = cudaArray[2 * d.triangles[i]];
        const double ay = cudaArray[2 * d.triangles[i] + 1];
        const double bx = cudaArray[2 * d.triangles[i + 1]];
        const double by = cudaArray[2 * d.triangles[i + 1] + 1];
        const double cx = cudaArray[2 * d.triangles[i + 2]];
        const double cy = cudaArray[2 * d.triangles[i + 2] + 1];
        triangles_areas[triangles_areas_size++] =
            std::fabs((by - ay) * (cx - bx) - (bx - ax) * (cy - by));
    }
    double triangles_area = sum(triangles_areas, triangles_areas_size);
    ASSERT_FLOAT_EQ(triangles_area, hull_area);
    // TODO: code duplication
}

CUDA_TEST(concave_iou, Delaunator2) {
    std::size_t number = 34;
    double cudaArray[MAX_J] = {4,
                               1,
                               3.7974166882130675,
                               2.0837249985614585,
                               3.2170267516619773,
                               3.0210869309396715,
                               2.337215067329615,
                               3.685489874065187,
                               1.276805078389906,
                               3.9872025288851036,
                               0.17901102978375127,
                               3.885476929518457,
                               -0.8079039091377689,
                               3.3940516818407187,
                               -1.550651407188842,
                               2.5792964886320684,
                               -1.9489192990517052,
                               1.5512485534497125,
                               -1.9489192990517057,
                               0.44875144655029087,
                               -1.5506514071888438,
                               -0.5792964886320653,
                               -0.8079039091377715,
                               -1.394051681840717,
                               0.17901102978374794,
                               -1.8854769295184561,
                               1.276805078389902,
                               -1.987202528885104,
                               2.337215067329611,
                               -1.6854898740651891,
                               3.217026751661974,
                               -1.021086930939675,
                               3.7974166882130653,
                               -0.08372499856146409};

    // TODO: code duplication

    Delaunator d(cudaArray, number);
    for (std::size_t i = 0; i < d.halfedges_size; ++i) {
        const auto i2 = d.halfedges[i];
        ASSERT_EQ(
            !static_cast<bool>((i2 != INVALID_INDEX) && (d.halfedges[i2] != i)),
            true);
    }

    // validate triangulation
    double hull_area = d.get_hull_area();
    double *triangles_areas = new double[d.triangles_size];
    std::size_t triangles_areas_size = 0;
    for (std::size_t i = 0; i < d.triangles_size; i += 3) {
        const double ax = cudaArray[2 * d.triangles[i]];
        const double ay = cudaArray[2 * d.triangles[i] + 1];
        const double bx = cudaArray[2 * d.triangles[i + 1]];
        const double by = cudaArray[2 * d.triangles[i + 1] + 1];
        const double cx = cudaArray[2 * d.triangles[i + 2]];
        const double cy = cudaArray[2 * d.triangles[i + 2] + 1];
        triangles_areas[triangles_areas_size++] =
            std::fabs((by - ay) * (cx - bx) - (bx - ax) * (cy - by));
    }
    double triangles_area = sum(triangles_areas, triangles_areas_size);
    ASSERT_FLOAT_EQ(triangles_area, hull_area);
    // TODO: code duplication
}

CUDA_TEST(concave_iou, Delaunator3) {
    std::size_t number = 158;
    double cudaArray[MAX_J] = {66.103648384371410,
                               68.588612471664760,
                               146.680713462100413,
                               121.680713462100428,
                               128.868896560467447,
                               117.261797559041411,
                               66.103648384371439,
                               68.588612471664774,
                               169.552139667571992,
                               146.133776538276890,
                               126.629392246050883,
                               181.111404660392082,
                               74.434448280233709,
                               78.630898779520691,
                               121.111404660392054,
                               153.370607753949116,
                               98.888595339607888,
                               186.629392246050855,
                               52.660668968140221,
                               63.178539267712423,
                               85.321337936280443,
                               86.357078535424832,
                               129.615705608064615,
                               173.901806440322616,
                               91.522409349774278,
                               162.346331352698143,
                               137.240951282800551,
                               112.240951282800537,
                               93.370607753949116,
                               158.888595339607917,
                               175,
                               150,
                               124.142135623730908,
                               184.142135623730979,
                               96.208227592327205,
                               94.083258291328988,
                               98.888595339607988,
                               153.370607753949059,
                               117.982006904420700,
                               109.535617803137270,
                               116.194470264303831,
                               108.267043413376910,
                               54.324378061245710,
                               62.306334965997713,
                               30.886889656046740,
                               47.726179755904141,
                               107.095117248373952,
                               101.809438047233129,
                               38.892261948632665,
                               52.594841299088443,
                               146.680713462100413,
                               121.680713462100399,
                               95.857864376269077,
                               155.857864376269020,
                               54.324378061245703,
                               62.306334965997706,
                               137.240951282800551,
                               112.240951282800551,
                               161.529565528607690,
                               140.440336826753821,
                               90.384294391935398,
                               166.098193559677383,
                               113.220729676874285,
                               93.717722494332946,
                               77.882918707497154,
                               74.870889977331813,
                               50,
                               60,
                               85.321337936280457,
                               86.357078535424847,
                               41.773779312093481,
                               55.452359511808289,
                               89.662189030622869,
                               81.153167482998867,
                               101.441459353748570,
                               87.435444988665906,
                               124.142135623730965,
                               155.857864376269048,
                               172.416455184654381,
                               148.166516582657948,
                               63.547558624186912,
                               70.904719023616522,
                               150.642675872560943,
                               132.714157070849694,
                               109.999999999999928,
                               190,
                               128.477590650225721,
                               177.653668647301827,
                               90,
                               169.999999999999943,
                               128.477590650225749,
                               162.346331352698200,
                               156.120475641400275,
                               131.120475641400275,
                               90.384294391935384,
                               173.901806440322502,
                               95.857864376268992,
                               184.142135623730894,
                               77.882918707497140,
                               74.870889977331799,
                               139.755786216514195,
                               124.987977314945553,
                               130,
                               170,
                               102.346331352698129,
                               188.477590650225693,
                               41.773779312093481,
                               55.452359511808282,
                               91.522409349774235,
                               177.653668647301714,
                               27.784523897265298,
                               45.189682598176865,
                               126.629392246050912,
                               158.888595339607974,
                               106.098193559677355,
                               189.615705608064587,
                               52.660668968140200,
                               63.178539267712395,
                               74.434448280233681,
                               78.630898779520677,
                               106.098193559677469,
                               150.384294391935384,
                               117.653668647301728,
                               188.477590650225749,
                               125,
                               100,
                               38.892261948632565,
                               52.594841299088379,
                               52.660668968140228,
                               63.178539267712416,
                               129.615705608064615,
                               166.098193559677440,
                               20,
                               40,
                               117.653668647301813,
                               151.522409349774278,
                               161.529565528607662,
                               140.440336826753821,
                               63.547558624186969,
                               70.904719023616564,
                               127.801189103500675,
                               102.801189103500675,
                               89.662189030622840,
                               81.153167482998853,
                               102.346331352698243,
                               151.522409349774250,
                               93.370607753949059,
                               181.111404660391968,
                               113.901806440322502,
                               189.615705608064615,
                               121.111404660391997,
                               186.629392246050940,
                               113.901806440322587,
                               150.384294391935384,
                               110.000000000000028,
                               150,
                               165.560237820700137,
                               140.560237820700137};

    // TODO: code duplication
    Delaunator d(cudaArray, number);
    for (std::size_t i = 0; i < d.halfedges_size; ++i) {
        const auto i2 = d.halfedges[i];
        ASSERT_EQ(
            !static_cast<bool>((i2 != INVALID_INDEX) && (d.halfedges[i2] != i)),
            true);
    }

    // validate triangulation
    double hull_area = d.get_hull_area();
    double *triangles_areas = new double[d.triangles_size];
    std::size_t triangles_areas_size = 0;
    for (std::size_t i = 0; i < d.triangles_size; i += 3) {
        const double ax = cudaArray[2 * d.triangles[i]];
        const double ay = cudaArray[2 * d.triangles[i] + 1];
        const double bx = cudaArray[2 * d.triangles[i + 1]];
        const double by = cudaArray[2 * d.triangles[i + 1] + 1];
        const double cx = cudaArray[2 * d.triangles[i + 2]];
        const double cy = cudaArray[2 * d.triangles[i + 2] + 1];
        triangles_areas[triangles_areas_size++] =
            std::fabs((by - ay) * (cx - bx) - (bx - ax) * (cy - by));
    }
    double triangles_area = sum(triangles_areas, triangles_areas_size);
    ASSERT_FLOAT_EQ(triangles_area, hull_area);
    // TODO: code duplication
}

CUDA_TEST(concave_iou, concave1) {
    std::size_t number = 14;
    double cudaArray[MAX_J] = {516, 661, 426, 539, 273, 525, 204,
                               694, 747, 750, 369, 793, 454, 390};
    double outArray[MAX_J] = {0};
    std::size_t outNumber = 0;
    concavehull(cudaArray, number, outArray, outNumber);
}

#endif