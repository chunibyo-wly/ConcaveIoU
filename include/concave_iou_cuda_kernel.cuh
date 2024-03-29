// Copyright (c) OpenMMLab. All rights reserved
#ifndef CONCAVE_IOU_CUDA_KERNEL_CUH
#define CONCAVE_IOU_CUDA_KERNEL_CUH

#include "convex_iou_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

#define HOST_DEVICE __host__ __device__
#define HOST_DEVICE_INLINE HOST_DEVICE __forceinline__

// Delaunator
// https://github.com/delfrrr/delaunator-
// https://mapbox.github.io/delaunator/

HOST_DEVICE_INLINE std::size_t next_halfedge(std::size_t e) {
    return (e % 3 == 2) ? e - 2 : e + 1;
}

HOST_DEVICE_INLINE std::size_t prev_halfedge(std::size_t e) {
    return (e % 3 == 0) ? e + 2 : e - 1;
}

HOST_DEVICE_INLINE void swap_size_t(std::size_t &x, std::size_t &y) {
    std::size_t tmp = x;
    x = y;
    y = tmp;
}

HOST_DEVICE_INLINE void swap_double(double &x, double &y) {
    double tmp = x;
    x = y;
    y = tmp;
}

template <typename scalar_t>
HOST_DEVICE_INLINE void _swap(scalar_t &x, scalar_t &y) {
    scalar_t tmp = x;
    x = y;
    y = tmp;
}

// https://stackoverflow.com/questions/33333363/built-in-mod-vs-custom-mod-function-improve-the-performance-of-modulus-op/33333636#33333636
HOST_DEVICE_INLINE size_t fast_mod(const size_t i, const size_t c) {
    return i >= c ? i % c : i;
}

// Kahan and Babuska summation, Neumaier variant; accumulates less FP error
template <typename scalar_t>
HOST_DEVICE_INLINE scalar_t sum(const scalar_t *x, const int n) {
    scalar_t sum = x[0];
    scalar_t err = 0.0;

    for (size_t i = 1; i < n; i++) {
        const scalar_t k = x[i];
        const scalar_t m = sum + k;
        err += std::fabs(sum) >= std::fabs(k) ? sum - m + k : k - m + sum;
        sum = m;
    }
    return sum + err;
}

template <typename scalar_t>
HOST_DEVICE_INLINE scalar_t dist(const scalar_t ax, const scalar_t ay,
                                 const scalar_t bx, const scalar_t by) {
    const scalar_t dx = ax - bx;
    const scalar_t dy = ay - by;
    return dx * dx + dy * dy;
}

template <typename scalar_t>
HOST_DEVICE_INLINE scalar_t circumradius(const scalar_t ax, const scalar_t ay,
                                         const scalar_t bx, const scalar_t by,
                                         const scalar_t cx, const scalar_t cy) {
    // 计算圆半径
    const scalar_t dx = bx - ax;
    const scalar_t dy = by - ay;
    const scalar_t ex = cx - ax;
    const scalar_t ey = cy - ay;

    const scalar_t bl = dx * dx + dy * dy;
    const scalar_t cl = ex * ex + ey * ey;
    const scalar_t d = dx * ey - dy * ex;

    const scalar_t x = (ey * bl - dy * cl) * 0.5 / d;
    const scalar_t y = (dx * cl - ex * bl) * 0.5 / d;

    if ((bl > 0.0 || bl < 0.0) && (cl > 0.0 || cl < 0.0) &&
        (d > 0.0 || d < 0.0)) {
        return x * x + y * y;
    } else {
        return std::numeric_limits<scalar_t>::max();
    }
}

template <typename scalar_t>
HOST_DEVICE_INLINE bool orient(const scalar_t px, const scalar_t py,
                               const scalar_t qx, const scalar_t qy,
                               const scalar_t rx, const scalar_t ry) {
    // ! 逆时针方向为真
    // (y2 - y1) * (x3 - x2) + (x2 - x1) * (y3 - y2)
    // https://www.geeksforgeeks.org/orientation-3-ordered-points/
    // To find orientation of ordered triplet (p1, p2, p3).
    return (qy - py) * (rx - qx) - (qx - px) * (ry - qy) < 0.0;
}

template <typename scalar_t>
HOST_DEVICE_INLINE void circumcenter(const scalar_t ax, const scalar_t ay,
                                     const scalar_t bx, const scalar_t by,
                                     const scalar_t cx, const scalar_t cy,
                                     scalar_t &x, scalar_t &y) {
    // 计算圆心
    const scalar_t dx = bx - ax;
    const scalar_t dy = by - ay;
    const scalar_t ex = cx - ax;
    const scalar_t ey = cy - ay;

    const scalar_t bl = dx * dx + dy * dy;
    const scalar_t cl = ex * ex + ey * ey;
    const scalar_t d = dx * ey - dy * ex;

    x = ax + (ey * bl - dy * cl) * 0.5 / d;
    y = ay + (dx * cl - ex * bl) * 0.5 / d;
}

template <typename scalar_t>
HOST_DEVICE_INLINE bool in_circle(const scalar_t ax, const scalar_t ay,
                                  const scalar_t bx, const scalar_t by,
                                  const scalar_t cx, const scalar_t cy,
                                  const scalar_t px, const scalar_t py) {
    // 判断 p 是否在 abc 组成的圆内
    const scalar_t dx = ax - px;
    const scalar_t dy = ay - py;
    const scalar_t ex = bx - px;
    const scalar_t ey = by - py;
    const scalar_t fx = cx - px;
    const scalar_t fy = cy - py;

    const scalar_t ap = dx * dx + dy * dy;
    const scalar_t bp = ex * ex + ey * ey;
    const scalar_t cp = fx * fx + fy * fy;

    return (dx * (ey * cp - bp * fy) - dy * (ex * cp - bp * fx) +
            ap * (ex * fy - ey * fx)) < 0.0;
}

constexpr std::size_t INVALID_INDEX = std::numeric_limits<std::size_t>::max();

template <typename scalar_t>
HOST_DEVICE_INLINE bool check_pts_equal(scalar_t x1, scalar_t y1, scalar_t x2,
                                        scalar_t y2) {
    return std::fabs(x1 - x2) <= std::numeric_limits<scalar_t>::epsilon() &&
           std::fabs(y1 - y2) <= std::numeric_limits<scalar_t>::epsilon();
}

// monotonically increases with real angle, but doesn't need expensive
// trigonometry
template <typename scalar_t>
HOST_DEVICE_INLINE double pseudo_angle(const scalar_t dx, const scalar_t dy) {
    // https://computergraphics.stackexchange.com/a/10523
    // TODO:
    // 好像是使用伪角可以做到和反三角函数一样的单调性，然后将三角函数对应的圆投影到
    // 0-1 区间。
    const scalar_t p = dx / (std::abs(dx) + std::abs(dy));
    return (dy > 0.0 ? 3.0 - p : 1.0 + p) / 4.0; // [0..1)
}

#define POINTS_NUMBER 500
const int MAX_TRIANGLES = POINTS_NUMBER < 3 ? 1 : 2 * POINTS_NUMBER - 5;

template <typename scalar_t> struct Delaunator {
  public:
    scalar_t coords[POINTS_NUMBER * 2];
    // 边映射到向量起点, 多对一
    std::size_t triangles[MAX_TRIANGLES * 3];
    std::size_t triangles_size;

    // 边对边
    std::size_t halfedges[MAX_TRIANGLES * 3];
    std::size_t halfedges_size;

    std::size_t hull_prev[POINTS_NUMBER * 2];
    std::size_t hull_next[POINTS_NUMBER * 2];
    // coords 顺序的第 i 个点对应的 halfedge
    // 中间量多边形第 i 个点对应的边
    std::size_t hull_tri[POINTS_NUMBER * 2];
    std::size_t hull_start;

    HOST_DEVICE_INLINE Delaunator(const scalar_t *in_coords,
                                  const std::size_t length);

    HOST_DEVICE_INLINE scalar_t get_hull_area();
    HOST_DEVICE_INLINE void get_hull_coords(scalar_t *hull_coords,
                                            std::size_t &hull_coords_size);
    HOST_DEVICE_INLINE void get_hull_points(std::size_t *hull_pts,
                                            std::size_t &hull_pts_size);
    HOST_DEVICE_INLINE scalar_t edge_length(std::size_t e);
    HOST_DEVICE_INLINE std::size_t get_interior_point(std::size_t e);

  private:
    std::size_t m_hash[POINTS_NUMBER];
    scalar_t m_center_x;
    scalar_t m_center_y;
    std::size_t m_hash_size;

    std::size_t m_edge_stack[MAX_TRIANGLES * 3];
    std::size_t m_edge_stack_size;

    HOST_DEVICE_INLINE std::size_t legalize(std::size_t a);
    HOST_DEVICE_INLINE std::size_t hash_key(const scalar_t x,
                                            const scalar_t y) const;
    HOST_DEVICE_INLINE std::size_t add_triangle(std::size_t i0, std::size_t i1,
                                                std::size_t i2, std::size_t a,
                                                std::size_t b, std::size_t c);
    HOST_DEVICE_INLINE void link(std::size_t a, std::size_t b);

    HOST_DEVICE_INLINE bool compare(std::size_t i, std::size_t j);
};

template <typename scalar_t>
HOST_DEVICE_INLINE Delaunator<scalar_t>::Delaunator(const scalar_t *in_coords,
                                                    const std::size_t length) {
    // intialize

    for (size_t i = 0; i < length; i++)
        coords[i] = in_coords[i];
    std::size_t n = length / 2;
    halfedges_size = 0;
    triangles_size = 0;
    m_edge_stack_size = 0;

    // initialize

    scalar_t max_x = std::numeric_limits<scalar_t>::min();
    scalar_t max_y = std::numeric_limits<scalar_t>::min();
    scalar_t min_x = std::numeric_limits<scalar_t>::max();
    scalar_t min_y = std::numeric_limits<scalar_t>::max();

    std::size_t ids[POINTS_NUMBER];

    // 坐标轴方向的矩形包围盒
    for (std::size_t i = 0; i < n; i++) {
        const scalar_t x = coords[2 * i];
        const scalar_t y = coords[2 * i + 1];

        if (x < min_x)
            min_x = x;
        if (y < min_y)
            min_y = y;
        if (x > max_x)
            max_x = x;
        if (y > max_y)
            max_y = y;

        ids[i] = i;
    }

    // 矩形包围盒中心
    const scalar_t cx = (min_x + max_x) / 2;
    const scalar_t cy = (min_y + max_y) / 2;
    scalar_t min_dist = std::numeric_limits<scalar_t>::max();

    // 到矩形包围盒中心最小的点
    std::size_t i0 = INVALID_INDEX;
    // 到 i0 最近的点
    std::size_t i1 = INVALID_INDEX;
    std::size_t i2 = INVALID_INDEX;

    // 计算每个点到矩形中心的距离，选出最小点 i0
    // pick a seed point close to the centroid
    for (std::size_t i = 0; i < n; i++) {
        const scalar_t d = dist(cx, cy, coords[2 * i], coords[2 * i + 1]);
        if (d < min_dist) {
            i0 = i;
            min_dist = d;
        }
    }

    const scalar_t i0x = coords[2 * i0];
    const scalar_t i0y = coords[2 * i0 + 1];

    min_dist = std::numeric_limits<scalar_t>::max();

    // 计算每个点到 i0 的距离，选出最小点 i1
    // find the point closest to the seed
    for (std::size_t i = 0; i < n; i++) {
        if (i == i0)
            continue;
        const scalar_t d = dist(i0x, i0y, coords[2 * i], coords[2 * i + 1]);
        if (d < min_dist && d > 0.0) {
            i1 = i;
            min_dist = d;
        }
    }

    scalar_t i1x = coords[2 * i1];
    scalar_t i1y = coords[2 * i1 + 1];

    scalar_t min_radius = std::numeric_limits<scalar_t>::max();

    // 遍历所有点，找出共圆后半径最小的点 i2
    // find the third point which forms the smallest circumcircle with the first
    // two
    for (std::size_t i = 0; i < n; i++) {
        if (i == i0 || i == i1)
            continue;

        const scalar_t r =
            circumradius(i0x, i0y, i1x, i1y, coords[2 * i], coords[2 * i + 1]);

        if (r < min_radius) {
            i2 = i;
            min_radius = r;
        }
    }

    if (!(min_radius < std::numeric_limits<scalar_t>::max())) {
        // TODO: throw CUDA error
        // throw std::runtime_error("not triangulation");
    }

    scalar_t i2x = coords[2 * i2];
    scalar_t i2y = coords[2 * i2 + 1];

    // 保证 i0 -> i1 -> i2 就是顺时针方向
    if (orient(i0x, i0y, i1x, i1y, i2x, i2y)) {
        _swap<std::size_t>(i1, i2);
        _swap<scalar_t>(i1x, i2x);
        _swap<scalar_t>(i1y, i2y);
    }

    // i0, i1, i2 三点共圆圆心
    circumcenter(i0x, i0y, i1x, i1y, i2x, i2y, m_center_x, m_center_y);

    // bubble sort
    // points number is little
    // sort the points by distance from the seed triangle circumcenter
    // 按照点到初始圆心距离排序
    for (std::size_t i = 0; i < n; i++) {
        for (std::size_t j = 0; j < n - i - 1; j++) {
            if (this->compare(ids[j + 1], ids[j]))
                swap_size_t(ids[j + 1], ids[j]);
        }
    }

    m_hash_size =
        static_cast<std::size_t>(std::llround(std::ceil(std::sqrt(n))));

    for (size_t i = 0; i < m_hash_size; i++) {
        m_hash[i] = INVALID_INDEX;
    }

    hull_start = i0;

    size_t hull_size = 3;

    hull_next[i0] = hull_prev[i2] = i1;
    hull_next[i1] = hull_prev[i0] = i2;
    hull_next[i2] = hull_prev[i1] = i0;

    hull_tri[i0] = 0;
    hull_tri[i1] = 1;
    hull_tri[i2] = 2;

    m_hash[hash_key(i0x, i0y)] = i0;
    m_hash[hash_key(i1x, i1y)] = i1;
    m_hash[hash_key(i2x, i2y)] = i2;

    add_triangle(i0, i1, i2, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX);
    scalar_t xp = std::numeric_limits<scalar_t>::quiet_NaN();
    scalar_t yp = std::numeric_limits<scalar_t>::quiet_NaN();

    for (std::size_t k = 0; k < n; k++) {
        // 遍历所有点，此时 ids 是从小到大的顺序
        // i 表示当前点
        const std::size_t i = ids[k];
        const scalar_t x = coords[2 * i];
        const scalar_t y = coords[2 * i + 1];

        // skip near-duplicate points
        // 保证当前点不和上一个点重合
        if (k > 0 && check_pts_equal(x, y, xp, yp))
            continue;
        // 备份当前点变成上一个点
        xp = x;
        yp = y;

        // skip seed triangle points
        // 跳过构成初始圆的三个点
        if (check_pts_equal(x, y, i0x, i0y) ||
            check_pts_equal(x, y, i1x, i1y) || check_pts_equal(x, y, i2x, i2y))
            continue;

        // find a visible edge on the convex hull using edge hash
        std::size_t start = 0;

        size_t key = hash_key(x, y);
        // 逆时针方向寻找距离角度最近的已加入集合中的点
        for (size_t j = 0; j < m_hash_size; j++) {
            start = m_hash[fast_mod(key + j, m_hash_size)];
            if (start != INVALID_INDEX && start != hull_next[start])
                break;
        }

        start = hull_prev[start];
        size_t e = start;
        size_t q;

        // q 永远是 e 的下一个点
        // ! 如果是顺时针就一直循环
        // ! 直到找到一组三点对 i -> e -> q 构成逆时针方向
        /*
         *          q
         *   +---->  +-----+
         *   |     | ^     |
         *   |     | |     |
         *   |     | |     v
         * i       | |       r
         *   ^     | |     |
         *   |     | |     |
         *   |     | |     |
         *   +---- v +<----+
         *          e
         */
        while (q = hull_next[e], !orient(x, y, coords[2 * e], coords[2 * e + 1],
                                         coords[2 * q], coords[2 * q + 1])) {
            // TODO: does it works in a same way as in JS
            e = q;
            if (e == start) {
                e = INVALID_INDEX;
                break;
            }
        }

        if (e == INVALID_INDEX)
            continue; // likely a near-duplicate point; skip it

        // add the first triangle from the point
        // i -> e -> q 逆时针方向, e -> i -> q 顺时针方向
        // e -> q 为两个三角形相邻边
        // 所以新三角形 △ieq 以 q 为起点的 halfedge 对应的是 △eqr 的以 e
        // 为起点的 halfedge
        std::size_t t = add_triangle(e, i, hull_next[e], INVALID_INDEX,
                                     INVALID_INDEX, hull_tri[e]);

        // t+2 对应的是新三角形以 hull_next[e] 或者说 q 为起点的 halfedge
        hull_tri[i] = legalize(t + 2);
        hull_tri[e] = t;
        hull_size++;

        // walk forward through the hull, adding more triangles and flipping
        // recursively
        // 遍历多边形外面的边，连接更多的三角形
        std::size_t next = hull_next[e];
        while (q = hull_next[next],
               orient(x, y, coords[2 * next], coords[2 * next + 1],
                      coords[2 * q], coords[2 * q + 1])) {
            t = add_triangle(next, i, q, hull_tri[i], INVALID_INDEX,
                             hull_tri[next]);
            hull_tri[i] = legalize(t + 2);
            hull_next[next] = next; // mark as removed
            hull_size--;
            next = q;
        }

        // walk backward from the other side, adding more triangles and flipping
        if (e == start) {
            while (q = hull_prev[e],
                   orient(x, y, coords[2 * q], coords[2 * q + 1], coords[2 * e],
                          coords[2 * e + 1])) {
                // TODO: 什么情况会进入这个循环
                t = add_triangle(q, i, e, INVALID_INDEX, hull_tri[e],
                                 hull_tri[q]);
                legalize(t + 2);
                hull_tri[q] = t;
                hull_next[e] = e; // mark as removed
                hull_size--;
                e = q;
            }
        }

        // update the hull indices
        hull_prev[i] = e;
        hull_start = e;
        hull_prev[next] = i;
        hull_next[e] = i;
        hull_next[i] = next;

        m_hash[hash_key(x, y)] = i;
        m_hash[hash_key(coords[2 * e], coords[2 * e + 1])] = e;
    }
}

template <typename scalar_t>
HOST_DEVICE_INLINE bool Delaunator<scalar_t>::compare(std::size_t i,
                                                      std::size_t j) {
    const scalar_t d1 = dist(coords[2 * i], coords[2 * i + 1], this->m_center_x,
                             this->m_center_y);
    const scalar_t d2 = dist(coords[2 * j], coords[2 * j + 1], this->m_center_x,
                             this->m_center_y);
    const scalar_t diff1 = d1 - d2;
    const scalar_t diff2 = coords[2 * i] - coords[2 * j];
    const scalar_t diff3 = coords[2 * i + 1] - coords[2 * j + 1];

    if (diff1 > 0.0 || diff1 < 0.0) {
        return diff1 < 0;
    } else if (diff2 > 0.0 || diff2 < 0.0) {
        return diff2 < 0;
    } else {
        return diff3 < 0;
    }
}

template <typename scalar_t>
HOST_DEVICE_INLINE std::size_t
Delaunator<scalar_t>::hash_key(const scalar_t x, const scalar_t y) const {
    const scalar_t dx = x - m_center_x;
    const scalar_t dy = y - m_center_y;
    return fast_mod(
        static_cast<std::size_t>(std::llround(std::floor(
            pseudo_angle(dx, dy) * static_cast<scalar_t>(m_hash_size)))),
        m_hash_size);
}

template <typename scalar_t>
HOST_DEVICE_INLINE void Delaunator<scalar_t>::link(const std::size_t a,
                                                   const std::size_t b) {
    // 建立无向图
    // halfedges[i] = j: 第 i 条边对应的第 j 条边
    // ! 强假设: 相邻两个三角形边按顺时针排列，重叠的两条边一定方向相反
    std::size_t s = halfedges_size;
    if (a == s) {
        halfedges[halfedges_size++] = b;
    } else if (a < s) {
        halfedges[a] = b;
    } else {
        // TODO: throw CUDA error
        // throw std::runtime_error("Cannot link edge");
    }

    if (b != INVALID_INDEX) {
        std::size_t s2 = halfedges_size;
        if (b == s2) {
            halfedges[halfedges_size++] = a;
        } else if (b < s2) {
            halfedges[b] = a;
        } else {
            // TODO: throw CUDA error
            // throw std::runtime_error("Cannot link edge");
        }
    }
}

template <typename scalar_t>
HOST_DEVICE_INLINE std::size_t
Delaunator<scalar_t>::add_triangle(std::size_t i0, std::size_t i1,
                                   std::size_t i2, std::size_t a, std::size_t b,
                                   std::size_t c) {
    // 输入的三个点 i0, i1, i2 保证为顺时针方向
    // 前三个参数为新三角形的三个点，会在这个函数里面映射到 halfedge
    // 后三个参数为已经记录过的 halfedge，与新的 halfedge 进行 link
    std::size_t t = triangles_size;
    triangles[triangles_size++] = i0; // t
    triangles[triangles_size++] = i1; // t + 1
    triangles[triangles_size++] = i2; // t + 2
    link(t, a);
    link(t + 1, b);
    link(t + 2, c);
    return t;
}

template <typename scalar_t>
HOST_DEVICE_INLINE std::size_t Delaunator<scalar_t>::legalize(std::size_t a) {
    // 第 a 条 halfedge
    std::size_t i = 0;
    std::size_t ar = 0;
    m_edge_stack_size = 0;

    // recursion eliminated with a fixed-size stack
    while (true) {
        // a 的对应边 b
        const size_t b = halfedges[a];

        /* if the pair of triangles doesn't satisfy the Delaunay condition
         * (p1 is inside the circumcircle of [p0, pl, pr]), flip them,
         * then do the same check/flip recursively for the new pair of triangles
         *
         *           pl                    pl
         *          /||\                  /  \
         *       al/ || \bl            al/    \a
         *        /  ||  \              /      \
         *       /  a||b  \    flip    /___ar___\
         *     p0\   ||   /p1   =>   p0\---bl---/p1
         *        \  ||  /              \      /
         *       ar\ || /br             b\    /br
         *          \||/                  \  /
         *           pr                    pr
         */

        // 三角形每三个一组，找到当前组的第一个
        const size_t a0 = 3 * (a / 3);
        ar = a0 + (a + 2) % 3;

        if (b == INVALID_INDEX) {
            if (i > 0) {
                i--;
                a = m_edge_stack[i];
                continue;
            } else {
                // i = INVALID_INDEX;
                break;
            }
        }

        const size_t b0 = 3 * (b / 3);
        const size_t al = a0 + (a + 1) % 3;
        const size_t bl = b0 + (b + 2) % 3;

        const std::size_t p0 = triangles[ar];
        const std::size_t pr = triangles[a];
        const std::size_t pl = triangles[al];
        const std::size_t p1 = triangles[bl];

        const bool illegal =
            in_circle(coords[2 * p0], coords[2 * p0 + 1], coords[2 * pr],
                      coords[2 * pr + 1], coords[2 * pl], coords[2 * pl + 1],
                      coords[2 * p1], coords[2 * p1 + 1]);
        if (illegal) {
            // p1 不能位于新三角形的外接圆中, 重新构造这两个三角形
            triangles[a] = p1;
            triangles[b] = p0;

            // 边重组的依据是一组三角形内边的顺序
            auto hbl = halfedges[bl];

            // edge swapped on the other side of the hull (rare); fix the
            // halfedge reference
            if (hbl == INVALID_INDEX) {
                // TODO: hull 指的是多边形上的边 ?
                std::size_t e = hull_start;
                do {
                    if (hull_tri[e] == bl) {
                        hull_tri[e] = a;
                        break;
                    }
                    e = hull_prev[e];
                } while (e != hull_start);
            }
            link(a, hbl);
            link(b, halfedges[ar]);
            link(ar, bl);
            std::size_t br = b0 + (b + 1) % 3;

            if (i < m_edge_stack_size) {
                m_edge_stack[i] = br;
            } else {
                m_edge_stack[m_edge_stack_size++] = br;
            }
            i++;

        } else {
            if (i > 0) {
                i--;
                a = m_edge_stack[i];
                continue;
            } else {
                break;
            }
        }
    }
    return ar;
}

template <typename scalar_t>
HOST_DEVICE_INLINE scalar_t Delaunator<scalar_t>::get_hull_area() {
    scalar_t hull_area[POINTS_NUMBER * 2];
    std::size_t hull_area_size = 0;

    std::size_t e = hull_start;
    do {
        hull_area[hull_area_size++] =
            ((coords[2 * e] - coords[2 * hull_prev[e]]) *
             (coords[2 * e + 1] + coords[2 * hull_prev[e] + 1]));
        e = hull_next[e];
    } while (e != hull_start);
    return sum(hull_area, hull_area_size);
}

template <typename scalar_t>
HOST_DEVICE_INLINE void
Delaunator<scalar_t>::get_hull_points(std::size_t *hull_pts,
                                      std::size_t &hull_pts_size) {
    hull_pts_size = 0;
    std::size_t point = hull_start;
    do {
        hull_pts[hull_pts_size++] = point;
        point = hull_next[point];
    } while (point != hull_start);

    // Wrap back around
    hull_pts[hull_pts_size++] = hull_start;
}

template <typename scalar_t>
HOST_DEVICE_INLINE void
Delaunator<scalar_t>::get_hull_coords(scalar_t *hull_coords,
                                      std::size_t &hull_coords_size) {
    std::size_t hull_pts[POINTS_NUMBER];
    std::size_t hull_pts_size = 0;
    get_hull_points(hull_pts, hull_pts_size);

    hull_coords_size = 0;
    for (std::size_t i = 0; i < hull_pts_size; ++i) {
        auto point = hull_pts[i];
        scalar_t x = coords[2 * point];
        scalar_t y = coords[2 * point + 1];
        hull_coords[hull_coords_size++] = x;
        hull_coords[hull_coords_size++] = y;
    }
}

template <typename scalar_t>
HOST_DEVICE_INLINE scalar_t Delaunator<scalar_t>::edge_length(std::size_t e_a) {
    size_t e_b = next_halfedge(e_a);

    scalar_t x_a = coords[2 * triangles[e_a]];
    scalar_t y_a = coords[2 * triangles[e_a] + 1];

    scalar_t x_b = coords[2 * triangles[e_b]];
    scalar_t y_b = coords[2 * triangles[e_b] + 1];

    return std::sqrt(std::pow(x_a - x_b, 2) + std::pow(y_a - y_b, 2));
}

template <typename scalar_t>
HOST_DEVICE_INLINE size_t
Delaunator<scalar_t>::get_interior_point(std::size_t e) {
    return triangles[next_halfedge(next_halfedge(e))];
}

// Delaunator

// Concave hull

template <typename scalar_t>
HOST_DEVICE_INLINE void sort_bheap(std::size_t *bheap, std::size_t bheap_size,
                                   scalar_t *edges_length) {
    for (std::size_t i = 0; i < bheap_size; ++i)
        bheap[i] = i;

    for (std::size_t i = 0; i < bheap_size; i++) {
        for (std::size_t j = 0; j < bheap_size - i - 1; j++) {
            if (edges_length[bheap[j + 1]] < edges_length[bheap[j]])
                swap_size_t(bheap[j + 1], bheap[j]);
        }
    }
}

HOST_DEVICE_INLINE bool exist(std::size_t *bpoints, std::size_t bpoints_size,
                              std::size_t c) {
    for (std::size_t i = 0; i < bpoints_size; ++i) {
        if (c == bpoints[i])
            return true;
    }
    return false;
}

template <typename scalar_t>
HOST_DEVICE_INLINE void
concavehull(const scalar_t *coords, const std::size_t coords_size,
            scalar_t *concaveCoords, std::size_t &concaveCoords_size,
            const scalar_t chi_factor = 0.1) {
    if (chi_factor < 0 || chi_factor > 1) {
        // TODO: cuda throw error
        // throw std::invalid_argument(
        //     "Chi factor must be between 0 and 1 inclusive");
    }

    Delaunator<scalar_t> d(coords, coords_size);

    // Determine initial points on outside hull
    // 历史点集合
    std::size_t bpoints[POINTS_NUMBER];
    std::size_t bpoints_size = 0;
    d.get_hull_points(bpoints, bpoints_size);

    // 用来排序的索引
    std::size_t bheap[POINTS_NUMBER];
    std::size_t bheap_size = bpoints_size;
    scalar_t max_len = std::numeric_limits<scalar_t>::min();
    scalar_t min_len = std::numeric_limits<scalar_t>::max();

    // 边以及对应的长度
    std::size_t edges[POINTS_NUMBER];
    scalar_t edges_length[POINTS_NUMBER];

    // 初始化
    for (std::size_t i = 0; i < bpoints_size; ++i) {
        // bheap[i] = i;
        edges[i] = d.hull_tri[bpoints[i]];
        edges_length[i] = d.edge_length(edges[i]);

        min_len = std::min(edges_length[i], min_len);
        max_len = std::max(edges_length[i], max_len);
    }
    sort_bheap(bheap, bheap_size, edges_length);

    // Determine length parameter
    scalar_t length_param = chi_factor * max_len + (1 - chi_factor) * min_len;

    // Iteratively add points to boundary by iterating over the triangles on the
    // hull
    while (bheap_size > 0) {
        // Get edge with the largest length
        std::size_t tmp = bheap[bheap_size - 1];
        bheap_size--;
        std::size_t e = edges[tmp];
        scalar_t len = edges_length[tmp];

        // Length of edge too small for our chi factor
        if (len <= length_param) {
            break;
        }

        // Find interior point given edge e (a -> b)
        //       e
        //  b <----- a
        //     \   /
        //  e_b \ / e_a
        //       c
        size_t c = d.get_interior_point(e);

        // Point already belongs to boundary
        if (exist(bpoints, bpoints_size, c))
            continue;

        // Get two edges connected to interior point
        //  c -> b
        size_t e_b = d.halfedges[next_halfedge(e)];
        //  a -> c
        size_t e_a = d.halfedges[next_halfedge(next_halfedge(e))];

        // Add edges to heap
        scalar_t len_a = d.edge_length(e_a);
        scalar_t len_b = d.edge_length(e_b);

        edges[bheap_size] = e_a, edges[bheap_size + 1] = e_b;
        edges_length[bheap_size] = len_a, edges_length[bheap_size + 1] = len_b;
        bheap_size += 2;
        sort_bheap(bheap, bheap_size, edges_length);

        // Update outer hull and connect new edges
        size_t a = d.triangles[e];
        size_t b = d.triangles[next_halfedge(e)];

        d.hull_next[c] = b;
        d.hull_prev[c] = a;
        d.hull_next[a] = d.hull_prev[b] = c;

        bpoints[bpoints_size++] = c;
    }

    d.get_hull_coords(concaveCoords, concaveCoords_size);
}

template <typename scalar_t>
HOST_DEVICE_INLINE float
concaveIoU(const scalar_t *predict, const std::size_t predict_size,
           const scalar_t *groundtruth, const std::size_t groundtruth_size,
           const scalar_t chi_factor = 0.1) {
    scalar_t predict_concave[POINTS_NUMBER];
    std::size_t predict_concave_size = 0;
    concavehull<scalar_t>(predict, predict_size, predict_concave,
                          predict_concave_size, chi_factor);

    Point ps1[POINTS_NUMBER], ps2[POINTS_NUMBER];

    int n1 = predict_concave_size / 2 - 1;
    for (std::size_t i = 0; i < n1; ++i)
        ps1[i].x = (double)predict_concave[2 * i],
        ps1[i].y = (double)predict_concave[2 * i + 1];

    int n2 = groundtruth_size / 2;
    for (std::size_t i = 0; i < n2; ++i)
        ps2[i].x = (double)groundtruth[2 * i],
        ps2[i].y = (double)groundtruth[2 * i + 1];

    double inter_area = intersectAreaO(ps1, n1, ps2, n2);
    double S_pred = area(ps1, n1);
    double union_area = fabs(S_pred) + fabs(area(ps2, n2)) - inter_area;
    double iou = inter_area / union_area;
    return (float)iou;
}

HOST_DEVICE_INLINE double polygon_area_grad2(Point *ps, int n,
                                             int *polygon_to_pred_index,
                                             int n_pred, double *grad_C) {
    ps[n] = ps[0];
    double partion_grad[4 * POINTS_NUMBER + 2];
    double res = 0;
    for (int i = 0; i < n; i++) {
        res += ps[i].x * ps[i + 1].y - ps[i].y * ps[i + 1].x;
        partion_grad[i * 4 + 2] = ps[i + 1].y;
        partion_grad[i * 4 + 3] = -ps[i + 1].x;
        if (i != n - 1) {
            partion_grad[i * 4 + 4] = -ps[i].y;
            partion_grad[i * 4 + 5] = ps[i].x;
        } else {
            partion_grad[0] = -ps[i].y;
            partion_grad[1] = ps[i].x;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n_pred; j++) {
            if (i == polygon_to_pred_index[j]) {
                grad_C[2 * polygon_to_pred_index[j + n_pred]] =
                    (partion_grad[i * 4] + partion_grad[i * 4 + 2]) / 2;
                break;
            }
        }
        for (int j = 0; j < n_pred; j++) {
            if (i == polygon_to_pred_index[j]) {
                grad_C[2 * polygon_to_pred_index[j + n_pred] + 1] =
                    (partion_grad[i * 4 + 1] + partion_grad[i * 4 + 1 + 2]) / 2;
                break;
            }
        }
    }

    return res / 2.0;
}

template <typename scalar_t>
HOST_DEVICE_INLINE float
concaveIoU(const scalar_t *predict, const std::size_t predict_size,
           const scalar_t *groundtruth, const std::size_t groundtruth_size,
           scalar_t *grad, const scalar_t chi_factor = 0.1) {
    // 带梯度的 iou
    scalar_t predict_concave[POINTS_NUMBER];
    std::size_t predict_concave_size = 0;
    concavehull<scalar_t>(predict, predict_size, predict_concave,
                          predict_concave_size, chi_factor);

    Point ps1[POINTS_NUMBER], ps2[POINTS_NUMBER];

    // 预测的凹包点
    int n1 = predict_concave_size / 2 - 1;
    for (std::size_t i = 0; i < n1; ++i)
        ps1[i].x = (double)predict_concave[2 * i],
        ps1[i].y = (double)predict_concave[2 * i + 1];
    reverse1(ps1, n1);

    // 真值
    int n2 = groundtruth_size / 2;
    for (std::size_t i = 0; i < n2; ++i)
        ps2[i].x = (double)groundtruth[2 * i],
        ps2[i].y = (double)groundtruth[2 * i + 1];

    int points_to_concave_ind[POINTS_NUMBER];
    for (int i = 0; i < POINTS_NUMBER; ++i)
        points_to_concave_ind[i] = -1;

    for (std::size_t i = 0; i < n1; ++i) {
        for (std::size_t j = 0; j < predict_size / 2; ++j) {
            auto p = Point(predict[j * 2], predict[j * 2 + 1]);
            if (point_same(ps1[i], p)) {
                points_to_concave_ind[i] = j;
                break;
            }
        }
    }

    int polygon_index_box_index[POINTS_NUMBER * 2];
    for (int i = 0; i < n1; i++) {
        polygon_index_box_index[i] = i;
        polygon_index_box_index[i + n1] = i;
    }

    // 预测值 - 面积
    double grad_A[POINTS_NUMBER * 2] = {0};
    // 预测值、真值交 - 面积
    double grad_AB[POINTS_NUMBER * 2] = {0};
    // 两个框的最小闭包区域 - 面积
    double grad_C[POINTS_NUMBER * 2] = {0};

    double inter_area = intersectAreaO(ps1, n1, ps2, n2, grad_AB);
    double S_pred =
        polygon_area_grad2(ps1, n1, polygon_index_box_index, n1, grad_A);
    if (S_pred < 0) {
        for (int i = 0; i < n1 * 2; i++) {
            grad_A[i] = -grad_A[i];
        }
    }
    double union_area = fabs(S_pred) + fabs(area(ps2, n2)) - inter_area;
    double iou = inter_area / union_area;

    for (int i = 0; i < n1; i++) {
        int grad_point = points_to_concave_ind[i];
        grad[2 * grad_point] =
            (float)((union_area + inter_area) / (union_area * union_area) *
                        grad_AB[2 * i] -
                    iou / union_area * grad_A[2 * i]);
        grad[2 * grad_point + 1] =
            (float)((union_area + inter_area) / (union_area * union_area) *
                        grad_AB[2 * i + 1] -
                    iou / union_area * grad_A[2 * i + 1]);
    }

    return (float)iou;
}

// Concave hull

// Cuda kernel

/**
 * @brief
 *
 * @param n_group : K 组
 * @param n_coords : 坐标点的数目
 * @param points : 坐标点
 * @param chi_factor : 生成凹包的参数
 * @param output : 输出
 * @return void
 */
__global__ void concave_hull_cuda_kernel(std::size_t n_group,
                                         std::size_t n_coords, double *points,
                                         double chi_factor, double *output) {
    CUDA_1D_KERNEL_LOOP(index, n_group) {
        // n_group 组点被拉长了，通过计算索引访问
        double *current_points = points + index * n_coords;
        // 每一行 n_coords+3 个元素
        // 每行第一个为输出的凹包的点坐标数目
        double *current_output = output + index * (n_coords + 3) + 1;
        std::size_t concaveCoords_size = 0;
        concavehull<double>(current_points, n_coords, current_output,
                            concaveCoords_size, chi_factor);
        output[index * (n_coords + 3)] = concaveCoords_size;
    }
}

template <typename T>
__global__ void
concave_iou_cuda_kernel(const T *predict, const std::size_t n_predict_points,
                        const std::size_t n_predict_group, const T *groundtruth,
                        const std::size_t n_groundtruth_points,
                        const std::size_t n_groundtruth_group,
                        const T chi_factor, T *ious) {
    CUDA_1D_KERNEL_LOOP(index, n_predict_group) {
        const T *cur_predict = predict + index * n_predict_points;
        for (std::size_t j = 0; j < n_groundtruth_group; ++j) {
            const T *cur_groundtruth = groundtruth + j * n_groundtruth_points;
            ious[index * n_groundtruth_group + j] =
                concaveIoU(cur_predict, n_predict_points, cur_groundtruth,
                           n_groundtruth_points, chi_factor);
        }
    }
}

template <typename T>
__global__ void concave_giou_cuda_kernel(const T *predict,
                                         const std::size_t n_predict_points,
                                         const std::size_t n_predict_group,
                                         const T *groundtruth,
                                         const std::size_t n_groundtruth_points,
                                         const std::size_t n_groundtruth_group,
                                         const T chi_factor, T *grad) {
    CUDA_1D_KERNEL_LOOP(index, n_predict_group) {
        const T *cur_predict = predict + index * n_predict_points;
        const T *cur_groundtruth = groundtruth + index * n_groundtruth_points;
        T *cur_grad = grad + index * (n_predict_points + 1);

        T iou = concaveIoU(cur_predict, n_predict_points, cur_groundtruth,
                           n_groundtruth_points, cur_grad, chi_factor);
        cur_grad[n_predict_points] = iou;
    }
}
// Cuda kernel

#endif // CONCAVE_IOU_CUDA_KERNEL_CUH