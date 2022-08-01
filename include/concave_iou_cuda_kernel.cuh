#define HOST_DEVICE __host__ __device__
#define HOST_DEVICE_INLINE HOST_DEVICE __forceinline__

// Delaunator
// https://github.com/delfrrr/delaunator-cpp

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

    HOST_DEVICE_INLINE Delaunator(double *in_coords);

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

HOST_DEVICE_INLINE Delaunator::Delaunator(double *in_coords) {}

// Delaunator