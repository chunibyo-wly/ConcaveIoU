#define HOST_DEVICE __host__ __device__
#define HOST_DEVICE_INLINE HOST_DEVICE __forceinline__

// Delaunator
// https://github.com/delfrrr/delaunator-cpp

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

#define POINTS_NUMBER 20
const int MAX_TRIANGLES = POINTS_NUMBER < 3 ? 1 : 2 * POINTS_NUMBER - 5;
struct Delaunator {
  public:
    double coords[POINTS_NUMBER * 2];
    std::size_t triangles[MAX_TRIANGLES * 3];
    std::size_t halfedges[MAX_TRIANGLES * 3];
    std::size_t *hull_prev;
    std::size_t *hull_next;
    std::size_t *hull_tri;
    std::size_t hull_start;

    HOST_DEVICE_INLINE Delaunator(double *in_coords, const std::size_t length);

    double get_hull_area();

  private:
    std::size_t *m_hash;
    double m_center_x;
    double m_center_y;
    std::size_t m_hash_size;
    std::size_t *m_edge_stack;

    HOST_DEVICE_INLINE std::size_t legalize(std::size_t a);
    HOST_DEVICE_INLINE std::size_t hash_key(double x, double y);
    HOST_DEVICE_INLINE std::size_t add_triangle(std::size_t i0, std::size_t i1,
                                                std::size_t i2, std::size_t a,
                                                std::size_t b, std::size_t c);
    HOST_DEVICE_INLINE void link(std::size_t a, std::size_t b);
};

HOST_DEVICE_INLINE Delaunator::Delaunator(double *in_coords,
                                          const std::size_t length) {

    std::size_t n = length / 2;

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
}

// Delaunator