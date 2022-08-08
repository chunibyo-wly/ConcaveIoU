#define HOST_DEVICE __host__ __device__
#define HOST_DEVICE_INLINE HOST_DEVICE __forceinline__

// Delaunator
// https://github.com/delfrrr/delaunator-cpp

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

// https://stackoverflow.com/questions/33333363/built-in-mod-vs-custom-mod-function-improve-the-performance-of-modulus-op/33333636#33333636
HOST_DEVICE_INLINE size_t fast_mod(const size_t i, const size_t c) {
    return i >= c ? i % c : i;
}

// Kahan and Babuska summation, Neumaier variant; accumulates less FP error
HOST_DEVICE_INLINE double sum(const double *x, const int n) {
    double sum = x[0];
    double err = 0.0;

    for (size_t i = 1; i < n; i++) {
        const double k = x[i];
        const double m = sum + k;
        err += std::fabs(sum) >= std::fabs(k) ? sum - m + k : k - m + sum;
        sum = m;
    }
    return sum + err;
}

HOST_DEVICE_INLINE double dist(const double ax, const double ay,
                               const double bx, const double by) {
    const double dx = ax - bx;
    const double dy = ay - by;
    return dx * dx + dy * dy;
}

HOST_DEVICE_INLINE double circumradius(const double ax, const double ay,
                                       const double bx, const double by,
                                       const double cx, const double cy) {
    // 计算圆半径
    const double dx = bx - ax;
    const double dy = by - ay;
    const double ex = cx - ax;
    const double ey = cy - ay;

    const double bl = dx * dx + dy * dy;
    const double cl = ex * ex + ey * ey;
    const double d = dx * ey - dy * ex;

    const double x = (ey * bl - dy * cl) * 0.5 / d;
    const double y = (dx * cl - ex * bl) * 0.5 / d;

    if ((bl > 0.0 || bl < 0.0) && (cl > 0.0 || cl < 0.0) &&
        (d > 0.0 || d < 0.0)) {
        return x * x + y * y;
    } else {
        return std::numeric_limits<double>::max();
    }
}

HOST_DEVICE_INLINE bool orient(const double px, const double py,
                               const double qx, const double qy,
                               const double rx, const double ry) {
    return (qy - py) * (rx - qx) - (qx - px) * (ry - qy) < 0.0;
}

HOST_DEVICE_INLINE void circumcenter(const double ax, const double ay,
                                     const double bx, const double by,
                                     const double cx, const double cy,
                                     double &x, double &y) {
    // 计算圆心
    const double dx = bx - ax;
    const double dy = by - ay;
    const double ex = cx - ax;
    const double ey = cy - ay;

    const double bl = dx * dx + dy * dy;
    const double cl = ex * ex + ey * ey;
    const double d = dx * ey - dy * ex;

    x = ax + (ey * bl - dy * cl) * 0.5 / d;
    y = ay + (dx * cl - ex * bl) * 0.5 / d;
}

HOST_DEVICE_INLINE bool in_circle(const double ax, const double ay,
                                  const double bx, const double by,
                                  const double cx, const double cy,
                                  const double px, const double py) {
    const double dx = ax - px;
    const double dy = ay - py;
    const double ex = bx - px;
    const double ey = by - py;
    const double fx = cx - px;
    const double fy = cy - py;

    const double ap = dx * dx + dy * dy;
    const double bp = ex * ex + ey * ey;
    const double cp = fx * fx + fy * fy;

    return (dx * (ey * cp - bp * fy) - dy * (ex * cp - bp * fx) +
            ap * (ex * fy - ey * fx)) < 0.0;
}

constexpr double EPSILON = std::numeric_limits<double>::epsilon();
constexpr std::size_t INVALID_INDEX = std::numeric_limits<std::size_t>::max();

HOST_DEVICE_INLINE bool check_pts_equal(double x1, double y1, double x2,
                                        double y2) {
    return std::fabs(x1 - x2) <= EPSILON && std::fabs(y1 - y2) <= EPSILON;
}

// monotonically increases with real angle, but doesn't need expensive
// trigonometry
HOST_DEVICE_INLINE double pseudo_angle(const double dx, const double dy) {
    const double p = dx / (std::abs(dx) + std::abs(dy));
    return (dy > 0.0 ? 3.0 - p : 1.0 + p) / 4.0; // [0..1)
}

struct DelaunatorPoint {
    std::size_t i;
    double x;
    double y;
    std::size_t t;
    std::size_t prev;
    std::size_t next;
    bool removed;
};

#define POINTS_NUMBER 500
const int MAX_TRIANGLES = POINTS_NUMBER < 3 ? 1 : 2 * POINTS_NUMBER - 5;
struct Delaunator {
  public:
    double coords[POINTS_NUMBER * 2];
    std::size_t triangles[MAX_TRIANGLES * 3];
    std::size_t triangles_size;

    std::size_t halfedges[MAX_TRIANGLES * 3];
    std::size_t halfedges_size;

    std::size_t hull_prev[POINTS_NUMBER * 2];
    std::size_t hull_next[POINTS_NUMBER * 2];
    std::size_t hull_tri[POINTS_NUMBER * 2];
    std::size_t hull_start;

    HOST_DEVICE_INLINE Delaunator(double *in_coords, const std::size_t length);

    HOST_DEVICE_INLINE double get_hull_area();
    HOST_DEVICE_INLINE void get_hull_coords(double *hull_coords,
                                            std::size_t &hull_coords_size);
    HOST_DEVICE_INLINE void get_hull_points(std::size_t *hull_pts,
                                            std::size_t &hull_pts_size);
    HOST_DEVICE_INLINE double edge_length(std::size_t e);
    HOST_DEVICE_INLINE std::size_t get_interior_point(std::size_t e);

  private:
    std::size_t m_hash[POINTS_NUMBER];
    double m_center_x;
    double m_center_y;
    std::size_t m_hash_size;

    std::size_t m_edge_stack[MAX_TRIANGLES * 3];
    std::size_t m_edge_stack_size;

    HOST_DEVICE_INLINE std::size_t legalize(std::size_t a);
    HOST_DEVICE_INLINE std::size_t hash_key(const double x,
                                            const double y) const;
    HOST_DEVICE_INLINE std::size_t add_triangle(std::size_t i0, std::size_t i1,
                                                std::size_t i2, std::size_t a,
                                                std::size_t b, std::size_t c);
    HOST_DEVICE_INLINE void link(std::size_t a, std::size_t b);

    HOST_DEVICE_INLINE bool compare(std::size_t &i, std::size_t &j);
};

HOST_DEVICE_INLINE Delaunator::Delaunator(double *in_coords,
                                          const std::size_t length) {
    // intialize

    for (size_t i = 0; i < length; i++)
        coords[i] = in_coords[i];
    std::size_t n = length / 2;
    halfedges_size = 0;
    triangles_size = 0;
    m_edge_stack_size = 0;

    // initialize

    double max_x = std::numeric_limits<double>::min();
    double max_y = std::numeric_limits<double>::min();
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();

    std::size_t ids[POINTS_NUMBER];

    for (std::size_t i = 0; i < n; i++) {
        const double x = coords[2 * i];
        const double y = coords[2 * i + 1];

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

    const double cx = (min_x + max_x) / 2;
    const double cy = (min_y + max_y) / 2;
    double min_dist = std::numeric_limits<double>::max();

    std::size_t i0 = INVALID_INDEX;
    std::size_t i1 = INVALID_INDEX;
    std::size_t i2 = INVALID_INDEX;

    // pick a seed point close to the centroid
    for (std::size_t i = 0; i < n; i++) {
        const double d = dist(cx, cy, coords[2 * i], coords[2 * i + 1]);
        if (d < min_dist) {
            i0 = i;
            min_dist = d;
        }
    }

    const double i0x = coords[2 * i0];
    const double i0y = coords[2 * i0 + 1];

    min_dist = std::numeric_limits<double>::max();

    // find the point closest to the seed
    for (std::size_t i = 0; i < n; i++) {
        if (i == i0)
            continue;
        const double d = dist(i0x, i0y, coords[2 * i], coords[2 * i + 1]);
        if (d < min_dist && d > 0.0) {
            i1 = i;
            min_dist = d;
        }
    }

    double i1x = coords[2 * i1];
    double i1y = coords[2 * i1 + 1];

    double min_radius = std::numeric_limits<double>::max();

    // find the third point which forms the smallest circumcircle with the first
    // two
    for (std::size_t i = 0; i < n; i++) {
        if (i == i0 || i == i1)
            continue;

        const double r =
            circumradius(i0x, i0y, i1x, i1y, coords[2 * i], coords[2 * i + 1]);

        if (r < min_radius) {
            i2 = i;
            min_radius = r;
        }
    }

    if (!(min_radius < std::numeric_limits<double>::max())) {
        // TODO: throw CUDA error
        // throw std::runtime_error("not triangulation");
    }

    double i2x = coords[2 * i2];
    double i2y = coords[2 * i2 + 1];

    if (orient(i0x, i0y, i1x, i1y, i2x, i2y)) {
        swap_size_t(i1, i2);
        swap_double(i1x, i2x);
        swap_double(i1y, i2y);
    }

    circumcenter(i0x, i0y, i1x, i1y, i2x, i2y, m_center_x, m_center_y);

    // bubble sort
    // points number is little
    for (std::size_t i = 0; i < n; i++) {
        for (std::size_t j = 0; j < n - i - 1; j++) {
            if (compare(i, j))
                swap_size_t(ids[i], ids[j]);
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
    double xp = std::numeric_limits<double>::quiet_NaN();
    double yp = std::numeric_limits<double>::quiet_NaN();

    for (std::size_t k = 0; k < n; k++) {
        const std::size_t i = ids[k];
        const double x = coords[2 * i];
        const double y = coords[2 * i + 1];

        // skip near-duplicate points
        if (k > 0 && check_pts_equal(x, y, xp, yp))
            continue;
        xp = x;
        yp = y;

        // skip seed triangle points
        if (check_pts_equal(x, y, i0x, i0y) ||
            check_pts_equal(x, y, i1x, i1y) || check_pts_equal(x, y, i2x, i2y))
            continue;

        // find a visible edge on the convex hull using edge hash
        std::size_t start = 0;

        size_t key = hash_key(x, y);
        for (size_t j = 0; j < m_hash_size; j++) {
            start = m_hash[fast_mod(key + j, m_hash_size)];
            if (start != INVALID_INDEX && start != hull_next[start])
                break;
        }

        start = hull_prev[start];
        size_t e = start;
        size_t q;

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
        std::size_t t = add_triangle(e, i, hull_next[e], INVALID_INDEX,
                                     INVALID_INDEX, hull_tri[e]);

        hull_tri[i] = legalize(t + 2);
        hull_tri[e] = t;
        hull_size++;

        // walk forward through the hull, adding more triangles and flipping
        // recursively
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

HOST_DEVICE_INLINE bool Delaunator::compare(std::size_t &i, std::size_t &j) {
    const double d1 = dist(coords[2 * i], coords[2 * i + 1], this->m_center_x,
                           this->m_center_y);
    const double d2 = dist(coords[2 * j], coords[2 * j + 1], this->m_center_x,
                           this->m_center_y);
    const double diff1 = d1 - d2;
    const double diff2 = coords[2 * i] - coords[2 * j];
    const double diff3 = coords[2 * i + 1] - coords[2 * j + 1];

    if (diff1 > 0.0 || diff1 < 0.0) {
        return diff1 < 0;
    } else if (diff2 > 0.0 || diff2 < 0.0) {
        return diff2 < 0;
    } else {
        return diff3 < 0;
    }
}

HOST_DEVICE_INLINE std::size_t Delaunator::hash_key(const double x,
                                                    const double y) const {
    const double dx = x - m_center_x;
    const double dy = y - m_center_y;
    return fast_mod(
        static_cast<std::size_t>(std::llround(std::floor(
            pseudo_angle(dx, dy) * static_cast<double>(m_hash_size)))),
        m_hash_size);
}

HOST_DEVICE_INLINE void Delaunator::link(const std::size_t a,
                                         const std::size_t b) {
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

HOST_DEVICE_INLINE std::size_t
Delaunator::add_triangle(std::size_t i0, std::size_t i1, std::size_t i2,
                         std::size_t a, std::size_t b, std::size_t c) {
    std::size_t t = triangles_size;
    triangles[triangles_size++] = i0;
    triangles[triangles_size++] = i1;
    triangles[triangles_size++] = i2;
    link(t, a);
    link(t + 1, b);
    link(t + 2, c);
    return t;
}

HOST_DEVICE_INLINE std::size_t Delaunator::legalize(std::size_t a) {
    std::size_t i = 0;
    std::size_t ar = 0;
    m_edge_stack_size = 0;

    // recursion eliminated with a fixed-size stack
    while (true) {
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
            triangles[a] = p1;
            triangles[b] = p0;

            auto hbl = halfedges[bl];

            // edge swapped on the other side of the hull (rare); fix the
            // halfedge reference
            if (hbl == INVALID_INDEX) {
                std::size_t e = hull_start;
                do {
                    if (hull_tri[e] == bl) {
                        hull_tri[e] = a;
                        break;
                    }
                    e = hull_next[e];
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

HOST_DEVICE_INLINE double Delaunator::get_hull_area() {
    double hull_area[POINTS_NUMBER * 2];
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

HOST_DEVICE_INLINE void
Delaunator::get_hull_points(std::size_t *hull_pts, std::size_t &hull_pts_size) {
    hull_pts_size = 0;
    std::size_t point = hull_start;
    do {
        hull_pts[hull_pts_size++] = point;
        point = hull_next[point];
    } while (point != hull_start);

    // Wrap back around
    hull_pts[hull_pts_size++] = hull_start;
}

HOST_DEVICE_INLINE void
Delaunator::get_hull_coords(double *hull_coords,
                            std::size_t &hull_coords_size) {
    std::size_t hull_pts[POINTS_NUMBER];
    std::size_t hull_pts_size = 0;
    get_hull_points(hull_pts, hull_pts_size);

    hull_coords_size = 0;
    for (std::size_t point : hull_pts) {
        double x = coords[2 * point];
        double y = coords[2 * point + 1];
        hull_coords[hull_coords_size++] = x;
        hull_coords[hull_coords_size++] = y;
    }
}

HOST_DEVICE_INLINE double Delaunator::edge_length(std::size_t e_a) {
    size_t e_b = next_halfedge(e_a);

    double x_a = coords[2 * triangles[e_a]];
    double y_a = coords[2 * triangles[e_a] + 1];

    double x_b = coords[2 * triangles[e_b]];
    double y_b = coords[2 * triangles[e_b] + 1];

    return std::sqrt(std::pow(x_a - x_b, 2) + std::pow(y_a - y_b, 2));
}

HOST_DEVICE_INLINE size_t Delaunator::get_interior_point(std::size_t e) {
    return triangles[next_halfedge(next_halfedge(e))];
}

// Delaunator

void concavehull(double *coords, std::size_t coords_size,
                 double chi_factor = 0.1) {
    if (chi_factor < 0 || chi_factor > 1) {
        // TODO: cuda throw error
        // throw std::invalid_argument(
        //     "Chi factor must be between 0 and 1 inclusive");
    }

    Delaunator d(coords, coords_size);

    // Determine initial points on outside hull
    std::size_t bpoints[POINTS_NUMBER];
    std::size_t bpoints_size = 0;

    d.get_hull_points(bpoints, bpoints_size);
}
