// Copyright (c) OpenMMLab. All rights reserved
#ifndef CONVEX_IOU_CUDA_KERNEL_CUH
#define CONVEX_IOU_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

#define HOST_DEVICE __host__ __device__

#define MAXN 100
#define NMAX 512
__device__ const double EPS = 1E-8;

HOST_DEVICE inline int sig(double d) { return (d > EPS) - (d < -EPS); }

struct Point {
  double x, y;
  HOST_DEVICE Point() {}
  HOST_DEVICE Point(double x, double y) : x(x), y(y) {}
};

HOST_DEVICE inline bool point_same(Point& a, Point& b) {
  return sig(a.x - b.x) == 0 && sig(a.y - b.y) == 0;
}

HOST_DEVICE inline void swap1(Point* a, Point* b) {
  Point temp;
  temp.x = a->x;
  temp.y = a->y;

  a->x = b->x;
  a->y = b->y;

  b->x = temp.x;
  b->y = temp.y;
}

HOST_DEVICE inline void reverse1(Point* a, const int n) {
  for (int i = 0; i < (n - 1) / 2.0; i++) {
    Point* j = &(a[i]);
    Point* k = &(a[n - 1 - i]);
    swap1(j, k);
  }
}

HOST_DEVICE inline double cross(Point o, Point a, Point b) {
  return (a.x - o.x) * (b.y - o.y) - (b.x - o.x) * (a.y - o.y);
}

HOST_DEVICE inline double dis(Point a, Point b) {
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}
HOST_DEVICE inline double area(Point* ps, int n) {
  ps[n] = ps[0];
  double res = 0;
  for (int i = 0; i < n; i++) {
    res += ps[i].x * ps[i + 1].y - ps[i].y * ps[i + 1].x;
  }
  return res / 2.0;
}
HOST_DEVICE inline double polygon_area_grad(Point* ps, int n,
                                           int* polygon_to_pred_index,
                                           int n_pred, double* grad_C) {
  ps[n] = ps[0];
  double partion_grad[4 * 30 + 2];
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

HOST_DEVICE inline int lineCross(Point a, Point b, Point c, Point d, Point& p,
                                double* cut_grad, int m, int n, int i) {
  double s1, s2;
  double s2_s1_2;
  double ds1_dxc, ds1_dyc, ds2_dxd, ds2_dyd;
  double dxp_dxc, dxp_dyc, dxp_dxd, dxp_dyd, dyp_dxc, dyp_dyc, dyp_dxd, dyp_dyd;
  s1 = cross(a, b, c);
  s2 = cross(a, b, d);

  ds1_dxc = -(b.y - a.y);
  ds1_dyc = b.x - a.x;
  ds2_dxd = ds1_dxc;
  ds2_dyd = ds1_dyc;
  s2_s1_2 = (s2 - s1) * (s2 - s1);

  if (sig(s1) == 0 && sig(s2) == 0) return 2;
  if (sig(s2 - s1) == 0) return 0;

  dxp_dxc =
      ((s2 - d.x * ds1_dxc) * (s2 - s1) - (c.x * s2 - d.x * s1) * (-ds1_dxc)) /
      (s2_s1_2);
  dxp_dyc =
      ((0 - d.x * ds1_dyc) * (s2 - s1) - (c.x * s2 - d.x * s1) * (-ds1_dyc)) /
      (s2_s1_2);
  dxp_dxd =
      ((c.x * ds2_dxd - s1) * (s2 - s1) - (c.x * s2 - d.x * s1) * (ds2_dxd)) /
      (s2_s1_2);
  dxp_dyd =
      ((c.x * ds2_dyd - 0) * (s2 - s1) - (c.x * s2 - d.x * s1) * (ds2_dyd)) /
      (s2_s1_2);

  dyp_dxc =
      ((0 - d.y * ds1_dxc) * (s2 - s1) - (c.y * s2 - d.y * s1) * (-ds1_dxc)) /
      (s2_s1_2);
  dyp_dyc =
      ((s2 - d.y * ds1_dyc) * (s2 - s1) - (c.y * s2 - d.y * s1) * (-ds1_dyc)) /
      (s2_s1_2);
  dyp_dxd =
      ((c.y * ds2_dxd - 0) * (s2 - s1) - (c.y * s2 - d.y * s1) * (ds2_dxd)) /
      (s2_s1_2);
  dyp_dyd =
      ((c.y * ds2_dyd - s1) * (s2 - s1) - (c.y * s2 - d.y * s1) * (ds2_dyd)) /
      (s2_s1_2);

  p.x = (c.x * s2 - d.x * s1) / (s2 - s1);
  p.y = (c.y * s2 - d.y * s1) / (s2 - s1);
  if (i == n - 1) {
    cut_grad[4 * n * m + 4 * i] = dxp_dxc;  // + dyp_dxc;
    cut_grad[4 * n * m + 4 * i + 1] = dyp_dxc;
    cut_grad[4 * n * m + 4 * i + 2] = dxp_dyc;  // + dyp_dyc;
    cut_grad[4 * n * m + 4 * i + 3] = dyp_dyc;
    cut_grad[4 * n * m + 0] = dxp_dxd;  // + dyp_dxd;
    cut_grad[4 * n * m + 1] = dyp_dxd;
    cut_grad[4 * n * m + 2] = dxp_dyd;  // + dyp_dyd;
    cut_grad[4 * n * m + 3] = dyp_dyd;
  } else {
    cut_grad[4 * n * m + 4 * i] = dxp_dxc;  // + dyp_dxc;
    cut_grad[4 * n * m + 4 * i + 1] = dyp_dxc;
    cut_grad[4 * n * m + 4 * i + 2] = dxp_dyc;  // + dyp_dyc;
    cut_grad[4 * n * m + 4 * i + 3] = dyp_dyc;
    cut_grad[4 * n * m + 4 * (i + 1)] = dxp_dxd;  // + dyp_dxd;
    cut_grad[4 * n * m + 4 * (i + 1) + 1] = dyp_dxd;
    cut_grad[4 * n * m + 4 * (i + 1) + 2] = dxp_dyd;  // + dyp_dyd;
    cut_grad[4 * n * m + 4 * (i + 1) + 3] = dyp_dyd;
  }

  return 1;
}
HOST_DEVICE inline void polygon_cut(Point* p, int& n, Point a, Point b,
                                   double* cut_grad) {
  Point pp[MAXN];
  double ccur_grad[MAXN] = {};
  int m = 0;
  p[n] = p[0];
  int k = n;
  for (int i = 0; i < n; i++) {
    if (sig(cross(a, b, p[i])) > 0) {
      pp[m] = p[i];
      ccur_grad[4 * n * m + 4 * i] = 1.0;
      ccur_grad[4 * n * m + 4 * i + 3] = 1.0;
      m++;
    }
    if (sig(cross(a, b, p[i])) != sig(cross(a, b, p[i + 1]))) {
      lineCross(a, b, p[i], p[i + 1], pp[m], ccur_grad, m, n, i);
      m++;
    }
  }

  n = 0;
  for (int i = 0; i < m; i++) {
    if (!i || !(point_same(pp[i], pp[i - 1]))) {
      p[n] = pp[i];
      for (int j = 0; j < 4 * k; j++) {
        cut_grad[4 * k * n + j] = ccur_grad[4 * k * i + j];
      }
      n++;
    }
  }

  while (n > 1 && point_same(p[n - 1], p[0])) n--;
}

HOST_DEVICE inline double intersectArea(Point a, Point b, Point c, Point d,
                                       double* grad_AB, int order,
                                       int convex_n) {
  Point o(0, 0);
  int res_flag = 0;
  int s1 = sig(cross(o, a, b));
  int s2 = sig(cross(o, c, d));
  if (s1 == 0 || s2 == 0) return 0.0;
  if (s1 == -1) {
    Point* i = &a;
    Point* j = &b;
    swap1(i, j);
    res_flag = 1;
  }
  if (s2 == -1) {
    Point* i = &c;
    Point* j = &d;
    swap1(i, j);
  }
  Point p[10] = {o, a, b};
  int n = 3, n0 = 3, n1, n2, n3;
  double cut_grad1[MAXN] = {};
  double cut_grad2[MAXN] = {};
  double cut_grad3[MAXN] = {};
  double p1_p_grad[10][10] = {};
  double p2_p1_grad[10][10] = {};
  double p3_p2_grad[10][10] = {};

  double p3_p1_grad[10][10] = {};
  double p3_p_grad[10][10] = {};

  // 1
  polygon_cut(p, n, o, c, cut_grad1);
  n1 = n;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 4 * n0; j++) {
      if (!(j % 2)) {
        p1_p_grad[2 * i][j / 2] = cut_grad1[4 * n0 * i + j];
      } else {
        p1_p_grad[2 * i + 1][j / 2] = cut_grad1[4 * n0 * i + j];
      }
    }
  }

  // 2
  polygon_cut(p, n, c, d, cut_grad2);
  n2 = n;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 4 * n1; j++) {
      if (!(j % 2)) {
        p2_p1_grad[2 * i][j / 2] = cut_grad2[4 * n1 * i + j];
      } else {
        p2_p1_grad[2 * i + 1][j / 2] = cut_grad2[4 * n1 * i + j];
      }
    }
  }
  // 3
  polygon_cut(p, n, d, o, cut_grad3);
  n3 = n;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 4 * n2; j++) {
      if (!(j % 2)) {
        p3_p2_grad[2 * i][j / 2] = cut_grad3[4 * n2 * i + j];
      } else {
        p3_p2_grad[2 * i + 1][j / 2] = cut_grad3[4 * n2 * i + j];
      }
    }
  }

  // mul
  //  p3_p2(n3 * n2) * p2_p1(n2 * n1) = p3_p1 (n3 * n1)
  for (int i = 0; i < 2 * n3; i++) {
    for (int j = 0; j < 2 * n1; j++) {
      double sum = 0.0;
      for (int m = 0; m < 2 * n2; m++) {
        sum = sum + p3_p2_grad[i][m] * p2_p1_grad[m][j];
      }
      p3_p1_grad[i][j] = sum;
    }
  }

  // p3_p1 (n3 * n1) * p1_p (n1 * n0) = p3_p (n3 * n0)
  for (int i = 0; i < 2 * n3; i++) {
    for (int j = 0; j < 2 * n0; j++) {
      double sum = 0.0;
      for (int m = 0; m < 2 * n1; m++) {
        sum = sum + p3_p1_grad[i][m] * p1_p_grad[m][j];
      }
      p3_p_grad[i][j] = sum;
    }
  }

  // calculate S_grad
  int polygon_index_box_index[20];
  double grad_polygon[20];
  double S_grad[6];

  for (int i = 0; i < n3; i++) {
    polygon_index_box_index[i] = i;
    polygon_index_box_index[i + n3] = i;
  }

  double res =
      polygon_area_grad(p, n3, polygon_index_box_index, n3, grad_polygon);

  if (s1 * s2 == -1) {
    for (int j = 0; j < 2 * 3; j++) {
      double sum = 0.0;
      for (int m = 0; m < 2 * n3; m++) {
        sum = sum - grad_polygon[m] * p3_p_grad[m][j];
      }
      S_grad[j] = sum;
    }

    if (order != convex_n - 1) {
      if (res_flag) {
        grad_AB[2 * order] += S_grad[4];
        grad_AB[2 * order + 1] += S_grad[5];
        grad_AB[2 * order + 2] += S_grad[2];
        grad_AB[2 * order + 3] += S_grad[3];

      } else {
        grad_AB[2 * order] += S_grad[2];
        grad_AB[2 * order + 1] += S_grad[3];
        grad_AB[2 * order + 2] += S_grad[4];
        grad_AB[2 * order + 3] += S_grad[5];
      }
    } else {
      if (res_flag) {
        grad_AB[2 * order] += S_grad[4];
        grad_AB[2 * order + 1] += S_grad[5];
        grad_AB[0] += S_grad[2];
        grad_AB[1] += S_grad[3];

      } else {
        grad_AB[2 * order] += S_grad[2];
        grad_AB[2 * order + 1] += S_grad[3];
        grad_AB[0] += S_grad[4];
        grad_AB[1] += S_grad[5];
      }
    }
    res = -res;
  } else {
    for (int j = 0; j < 2 * 3; j++) {
      double sum = 0.0;
      for (int m = 0; m < 2 * n3; m++) {
        sum = sum + grad_polygon[m] * p3_p_grad[m][j];
      }
      S_grad[j] = sum;
    }

    if (order != convex_n - 1) {
      if (res_flag) {
        grad_AB[2 * order] += S_grad[4];
        grad_AB[2 * order + 1] += S_grad[5];
        grad_AB[2 * order + 2] += S_grad[2];
        grad_AB[2 * order + 3] += S_grad[3];
      } else {
        grad_AB[2 * order] += S_grad[2];
        grad_AB[2 * order + 1] += S_grad[3];
        grad_AB[2 * order + 2] += S_grad[4];
        grad_AB[2 * order + 3] += S_grad[5];
      }
    } else {
      if (res_flag) {
        grad_AB[2 * order] += S_grad[4];
        grad_AB[2 * order + 1] += S_grad[5];
        grad_AB[0] += S_grad[2];
        grad_AB[1] += S_grad[3];
      } else {
        grad_AB[2 * order] += S_grad[2];
        grad_AB[2 * order + 1] += S_grad[3];
        grad_AB[0] += S_grad[4];
        grad_AB[1] += S_grad[5];
      }
    }
  }
  return res;
}

HOST_DEVICE inline double intersectAreaO(Point* ps1, int n1, Point* ps2, int n2,
                                        double* grad_AB) {
  if (area(ps1, n1) < 0) reverse1(ps1, n1);
  if (area(ps2, n2) < 0) reverse1(ps2, n2);
  ps1[n1] = ps1[0];
  ps2[n2] = ps2[0];
  double res = 0;
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      res +=
          intersectArea(ps1[i], ps1[i + 1], ps2[j], ps2[j + 1], grad_AB, i, n1);
    }
  }
  return res;
}

HOST_DEVICE inline void Jarvis(Point* in_poly, int& n_poly) {
  Point p_max, p_k;
  int max_index, k_index;
  int Stack[NMAX] = {}, top1, top2;
  double sign;
  Point right_point[10], left_point[10];

  for (int i = 0; i < n_poly; i++) {
    if (in_poly[i].y < in_poly[0].y ||
        in_poly[i].y == in_poly[0].y && in_poly[i].x < in_poly[0].x) {
      Point* j = &(in_poly[0]);
      Point* k = &(in_poly[i]);
      swap1(j, k);
    }
    if (i == 0) {
      p_max = in_poly[0];
      max_index = 0;
    }
    if (in_poly[i].y > p_max.y ||
        in_poly[i].y == p_max.y && in_poly[i].x > p_max.x) {
      p_max = in_poly[i];
      max_index = i;
    }
  }

  if (max_index == 0) {
    max_index = 1;
    p_max = in_poly[max_index];
  }

  k_index = 0, Stack[0] = 0, top1 = 0;
  while (k_index != max_index) {
    p_k = p_max;
    k_index = max_index;
    for (int i = 1; i < n_poly; i++) {
      sign = cross(in_poly[Stack[top1]], in_poly[i], p_k);
      if ((sign > 0) || ((sign == 0) && (dis(in_poly[Stack[top1]], in_poly[i]) >
                                         dis(in_poly[Stack[top1]], p_k)))) {
        p_k = in_poly[i];
        k_index = i;
      }
    }
    top1++;
    Stack[top1] = k_index;
  }
  for (int i = 0; i <= top1; i++) right_point[i] = in_poly[Stack[i]];

  k_index = 0, Stack[0] = 0, top2 = 0;

  while (k_index != max_index) {
    p_k = p_max;
    k_index = max_index;
    for (int i = 1; i < n_poly; i++) {
      sign = cross(in_poly[Stack[top2]], in_poly[i], p_k);
      if ((sign < 0) || (sign == 0) && (dis(in_poly[Stack[top2]], in_poly[i]) >
                                        dis(in_poly[Stack[top2]], p_k))) {
        p_k = in_poly[i];
        k_index = i;
      }
    }
    top2++;
    Stack[top2] = k_index;
  }
  for (int i = top2 - 1; i >= 0; i--) left_point[i] = in_poly[Stack[i]];

  for (int i = 0; i < top1 + top2; i++) {
    if (i <= top1) {
      in_poly[i] = right_point[i];
    } else {
      in_poly[i] = left_point[top2 - (i - top1)];
    }
  }
  n_poly = top1 + top2;
}

HOST_DEVICE inline double intersectAreaPoly(Point* ps1, int n1, Point* ps2,
                                           int n2, double* grad_C) {
  Point polygon[MAXN];
  int n = n1 + n2, n_poly = 0;
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n - n1; j++) {
      if (point_same(ps1[i], ps2[j])) {
        for (int k = j; k < n - n1 - 1; k++) {
          ps2[k] = ps2[k + 1];
        }
        n2--;
        break;
      }
    }
  }
  n_poly = n1 + n2;
  for (int i = 0; i < n_poly; i++) {
    if (i < n1) {
      polygon[i] = ps1[i];
    } else {
      polygon[i] = ps2[i - n1];
    }
  }

  Jarvis(polygon, n_poly);

  int polygon_to_pred_index[18] = {-1, -1, -1, -1, -1, -1, -1, -1, -1,
                                   -1, -1, -1, -1, -1, -1, -1, -1, -1};
  int n_pred = 0;
  for (int i = 0; i < n_poly; i++) {
    for (int j = 0; j < n1; j++) {
      if (polygon[i].x == ps1[j].x && polygon[i].y == ps1[j].y) {
        polygon_to_pred_index[n_pred] = i;
        polygon_to_pred_index[n_pred + n1] = j;
        n_pred += 1;
        break;
      }
    }
  }
  if (n_pred == 0) {
    double polygon_area = fabs(area(polygon, n_poly));
    for (int i = 0; i < 18; i++) {
      grad_C[i] = 0.0;
    }
    return polygon_area;
  } else {
    double polygon_area =
        polygon_area_grad(polygon, n_poly, polygon_to_pred_index, n1, grad_C);
    if (polygon_area < 0) {
      for (int i = 0; i < 18; i++) {
        grad_C[i] = -grad_C[i];
      }
    }
    return fabs(polygon_area);
  }
}

// convex_find and get the polygon_index_box_index
HOST_DEVICE inline void Jarvis_and_index(Point* in_poly, int& n_poly,
                                        int* points_to_convex_ind) {
  int n_input = n_poly;
  Point input_poly[20];
  for (int i = 0; i < n_input; i++) {
    input_poly[i].x = in_poly[i].x;
    input_poly[i].y = in_poly[i].y;
  }
  Point p_max, p_k;
  int max_index, k_index;
  int Stack[20], top1, top2;
  double sign;
  Point right_point[10], left_point[10];

  for (int i = 0; i < n_poly; i++) {
    if (in_poly[i].y < in_poly[0].y ||
        in_poly[i].y == in_poly[0].y && in_poly[i].x < in_poly[0].x) {
      Point* j = &(in_poly[0]);
      Point* k = &(in_poly[i]);
      swap1(j, k);
    }
    if (i == 0) {
      p_max = in_poly[0];
      max_index = 0;
    }
    if (in_poly[i].y > p_max.y ||
        in_poly[i].y == p_max.y && in_poly[i].x > p_max.x) {
      p_max = in_poly[i];
      max_index = i;
    }
  }
  if (max_index == 0) {
    max_index = 1;
    p_max = in_poly[max_index];
  }

  k_index = 0, Stack[0] = 0, top1 = 0;
  while (k_index != max_index) {
    p_k = p_max;
    k_index = max_index;
    for (int i = 1; i < n_poly; i++) {
      sign = cross(in_poly[Stack[top1]], in_poly[i], p_k);
      if ((sign > 0) || ((sign == 0) && (dis(in_poly[Stack[top1]], in_poly[i]) >
                                         dis(in_poly[Stack[top1]], p_k)))) {
        p_k = in_poly[i];
        k_index = i;
      }
    }
    top1++;
    Stack[top1] = k_index;
  }
  for (int i = 0; i <= top1; i++) {
    right_point[i] = in_poly[Stack[i]];
  }

  k_index = 0, Stack[0] = 0, top2 = 0;

  while (k_index != max_index) {
    p_k = p_max;
    k_index = max_index;
    for (int i = 1; i < n_poly; i++) {
      sign = cross(in_poly[Stack[top2]], in_poly[i], p_k);
      if ((sign < 0) || (sign == 0) && (dis(in_poly[Stack[top2]], in_poly[i]) >
                                        dis(in_poly[Stack[top2]], p_k))) {
        p_k = in_poly[i];
        k_index = i;
      }
    }
    top2++;
    Stack[top2] = k_index;
  }

  for (int i = top2 - 1; i >= 0; i--) {
    left_point[i] = in_poly[Stack[i]];
  }

  for (int i = 0; i < top1 + top2; i++) {
    if (i <= top1) {
      in_poly[i] = right_point[i];
    } else {
      in_poly[i] = left_point[top2 - (i - top1)];
    }
  }
  n_poly = top1 + top2;
  for (int i = 0; i < n_poly; i++) {
    for (int j = 0; j < n_input; j++) {
      if (point_same(in_poly[i], input_poly[j])) {
        points_to_convex_ind[i] = j;
        break;
      }
    }
  }
}

template <typename T>
HOST_DEVICE inline float devrIoU(T const* const p, T const* const q,
                                T* point_grad, const int idx) {
  Point ps1[MAXN], ps2[MAXN];

  Point convex[MAXN];
  for (int i = 0; i < 9; i++) {
    convex[i].x = (double)p[i * 2];
    convex[i].y = (double)p[i * 2 + 1];
  }
  int n_convex = 9;
  int points_to_convex_ind[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
  Jarvis_and_index(convex, n_convex, points_to_convex_ind);

  int n1 = n_convex;
  int n2 = 4;

  for (int i = 0; i < n1; i++) {
    ps1[i].x = (double)convex[i].x;
    ps1[i].y = (double)convex[i].y;
  }

  for (int i = 0; i < n2; i++) {
    ps2[i].x = (double)q[i * 2];
    ps2[i].y = (double)q[i * 2 + 1];
  }

  int polygon_index_box_index[18];
  for (int i = 0; i < n1; i++) {
    polygon_index_box_index[i] = i;
    polygon_index_box_index[i + n1] = i;
  }

  double grad_A[18] = {};
  double grad_AB[18] = {};
  double grad_C[18] = {};

  double inter_area = intersectAreaO(ps1, n1, ps2, n2, grad_AB);
  double S_pred =
      polygon_area_grad(ps1, n1, polygon_index_box_index, n1, grad_A);
  if (S_pred < 0) {
    for (int i = 0; i < n_convex * 2; i++) {
      grad_A[i] = -grad_A[i];
    }
  }
  double union_area = fabs(S_pred) + fabs(area(ps2, n2)) - inter_area;

  double iou = inter_area / union_area;
  double polygon_area = intersectAreaPoly(ps1, n1, ps2, n2, grad_C);

  //    printf("%d:live\n", idx);
  double rot_giou = iou - (polygon_area - union_area) / polygon_area;

  float grad_point_temp[18] = {};

  for (int i = 0; i < n_convex; i++) {
    int grad_point = points_to_convex_ind[i];
    grad_point_temp[2 * grad_point] =
        (float)((union_area + inter_area) / (union_area * union_area) *
                    grad_AB[2 * i] -
                iou / union_area * grad_A[2 * i] -
                1 / polygon_area * (grad_AB[2 * i] - grad_A[2 * i]) -
                (union_area) / polygon_area / polygon_area * grad_C[2 * i]);
    grad_point_temp[2 * grad_point + 1] =
        (float)((union_area + inter_area) / (union_area * union_area) *
                    grad_AB[2 * i + 1] -
                iou / union_area * grad_A[2 * i + 1] -
                1 / polygon_area * (grad_AB[2 * i + 1] - grad_A[2 * i + 1]) -
                (union_area) / polygon_area / polygon_area * grad_C[2 * i + 1]);
  }

  for (int i = 0; i < 9; i++) {
    point_grad[2 * i] = grad_point_temp[2 * i];
    point_grad[2 * i + 1] = grad_point_temp[2 * i + 1];
  }
  return (float)rot_giou;
}

HOST_DEVICE inline bool isApproxEqual(double &a, double &b) {
    return sig(a - b) == 0;
}

HOST_DEVICE inline bool isRectangleOrPolygon(Point *polygon, int n) {
    double a = dis(polygon[0], polygon[1]), b = dis(polygon[2], polygon[3]);
    if (min(a, b) / max(a, b) > 0.97)
        return true;
    else
        return false;
}

HOST_DEVICE inline void
findIntersectionBetweenLineRectangle(Point p1, Point p2, double minx,
                                     double miny, double maxx, double maxy,
                                     Point *sy) {
    double x = -1, y = -1;
    if (isApproxEqual(p1.x, p2.x)) {
        x = p1.x;
        if (minx <= x && x <= maxx)
            sy[0] = Point(x, miny), sy[1] = Point(x, maxy);
    } else if (isApproxEqual(p1.y, p2.y)) {
        y = p1.y;
        if (miny <= y && y <= maxy)
            sy[0] = Point(minx, y), sy[1] = Point(maxx, y);
    } else {
        double m = (p2.y - p1.y) / (p2.x - p1.x);
        double c = p1.y - m * p1.x;
        int cnt = 0;

        // Left side
        y = m * minx + c;
        if (miny <= y && y <= maxy)
            sy[cnt++] = Point(minx, y);

        // Right side
        y = m * maxx + c;
        if (miny <= y && y <= maxy)
            sy[cnt++] = Point(maxx, y);

        // Bottom side
        x = (miny - c) / m;
        if (cnt < 1 && minx <= x && x <= maxx)
            sy[cnt++] = Point(x, miny);

        // Top side
        x = (maxy - c) / m;
        if (cnt < 1 && minx <= x && x <= maxx)
            sy[cnt++] = Point(x, maxy);
    }
}

HOST_DEVICE inline void findSymmetryPolygon(Point *polygon, int n,
                                            Point *output, int &outputN) {
    Point p1, p2;
    for (int i = 0; i < n; ++i) {
        if (isRectangleOrPolygon(polygon, n)) {
            // 矩形
            if (dis(polygon[0], polygon[1]) > dis(polygon[1], polygon[2]))
                p1 = Point((polygon[1].x + polygon[2].x) / 2,
                           (polygon[1].y + polygon[2].y) / 2),
                p2 = Point((polygon[0].x + polygon[3].x) / 2,
                           (polygon[0].y + polygon[3].y) / 2);
            else
                p1 = Point((polygon[0].x + polygon[1].x) / 2,
                           (polygon[0].y + polygon[1].y) / 2),
                p2 = Point((polygon[2].x + polygon[3].x) / 2,
                           (polygon[2].y + polygon[3].y) / 2);
        } else {
            // 多边形
            int index = -1;
            double max_S = -1;
            for (int i = 0; i < n; ++i) {
                Point &p1 = polygon[i], &p2 = polygon[(i + 1) % n],
                      &p3 = polygon[(i + 2) % n];
                double S = fabs(cross(p2, p1, p3));
                if (S > max_S) {
                    max_S = S;
                    index = (i + 1) % n;
                }
            }
            p1 = Point(polygon[index].x, polygon[index].y);
            p2 = Point(polygon[(index + 2) % n].x, polygon[(index + 2) % n].y);
        }
    }

    // 获取逆时针方向所有点
    Point sy[5] = {};
    double minx = 0, miny = 0, maxx = 10000, maxy = 10000;
    findIntersectionBetweenLineRectangle(p1, p2, minx, miny, maxx, maxy, sy);

    Point O(0, 0);
    if (sig(cross(O, sy[0], sy[1])) < 0)
        output[0] = sy[1], output[1] = sy[0];
    else
        output[0] = sy[0], output[1] = sy[1];

    outputN = 2;
    if (sig(cross(output[0], output[1], Point(minx, miny))) > 0)
        output[outputN++] = Point(minx, miny);
    if (sig(cross(output[0], output[1], Point(maxx, miny))) > 0)
        output[outputN++] = Point(maxx, miny);
    if (sig(cross(output[0], output[1], Point(maxx, maxy))) > 0)
        output[outputN++] = Point(maxx, maxy);
    if (sig(cross(output[0], output[1], Point(minx, maxy))) > 0)
        output[outputN++] = Point(minx, maxy);
    Jarvis(output, outputN);
}

template <typename T>
HOST_DEVICE inline float devrSymmetryIOU(T const *const p, T const *const q,
                                         T const *const symmetry, T *point_grad,
                                         const int idx) {
    // GIoU 计算
    Point ps1[MAXN], ps2[MAXN];

    Point convex[MAXN];
    for (int i = 0; i < 9; i++) {
        convex[i].x = (double)p[i * 2];
        convex[i].y = (double)p[i * 2 + 1];
    }
    int n_convex = 9;
    int points_to_convex_ind[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
    Jarvis_and_index(convex, n_convex, points_to_convex_ind);

    int n1 = n_convex;
    int n2 = 4;

    for (int i = 0; i < n1; i++) {
        ps1[i].x = (double)convex[i].x;
        ps1[i].y = (double)convex[i].y;
    }

    for (int i = 0; i < n2; i++) {
        ps2[i].x = (double)q[i * 2];
        ps2[i].y = (double)q[i * 2 + 1];
    }

    int polygon_index_box_index[18];
    for (int i = 0; i < n1; i++) {
        polygon_index_box_index[i] = i;
        polygon_index_box_index[i + n1] = i;
    }

    // 预测值 A 的面积对 A 的梯度
    double grad_A[18] = {};
    // A B 多边形交集的梯度
    double grad_AB[18] = {};
    double grad_C[18] = {};

    double inter_area = intersectAreaO(ps1, n1, ps2, n2, grad_AB);
    double S_pred =
        polygon_area_grad(ps1, n1, polygon_index_box_index, n1, grad_A);
    if (S_pred < 0) {
        for (int i = 0; i < n_convex * 2; i++) {
            grad_A[i] = -grad_A[i];
        }
    }
    // 并集的面积 = S_A + S_B - S_AB
    // 因为 S_B 都是零，所以可以根据 S_A 和 S_AB 的梯度推导出来
    double union_area = fabs(S_pred) + fabs(area(ps2, n2)) - inter_area;

    double iou = inter_area / union_area;
    double polygon_area = intersectAreaPoly(ps1, n1, ps2, n2, grad_C);

    //    printf("%d:live\n", idx);
    double rot_giou = iou - (polygon_area - union_area) / polygon_area;

    float grad_point_temp[18] = {};

    for (int i = 0; i < n_convex; i++) {
        int grad_point = points_to_convex_ind[i];
        grad_point_temp[2 * grad_point] =
            (float)((union_area + inter_area) / (union_area * union_area) *
                        grad_AB[2 * i] -
                    iou / union_area * grad_A[2 * i] -
                    1 / polygon_area * (grad_AB[2 * i] - grad_A[2 * i]) -
                    (union_area) / polygon_area / polygon_area * grad_C[2 * i]);
        grad_point_temp[2 * grad_point + 1] =
            (float)((union_area + inter_area) / (union_area * union_area) *
                        grad_AB[2 * i + 1] -
                    iou / union_area * grad_A[2 * i + 1] -
                    1 / polygon_area *
                        (grad_AB[2 * i + 1] - grad_A[2 * i + 1]) -
                    (union_area) / polygon_area / polygon_area *
                        grad_C[2 * i + 1]);
    }
    // GIoU 计算

    // SymmetryIoU
    Point syPs[6];
    int syN = 0;
    findSymmetryPolygon(ps1, n1, syPs, syN);

    double grad_S1[18] = {};
    double S1 = intersectAreaO(ps1, n1, syPs, syN, grad_S1);
    double S = S_pred;
    for (int i = 0; i < n_convex; i++) {
        int grad_point = points_to_convex_ind[i];
        grad_point_temp[2 * grad_point] +=
            (float) ((2 * S1 - S) / S * (grad_S1[2 * i] * S - grad_A[2 * i] * S1) / S / S);
        grad_point_temp[2 * grad_point + 1] +=
            (float) ((2 * S1 - S) / S * (grad_S1[2 * i + 1] * S - grad_A[2 * i + 1] * S1) / S / S);
    }
    // SymmetryIoU

    for (int i = 0; i < 9; i++) {
        point_grad[2 * i] = grad_point_temp[2 * i];
        point_grad[2 * i + 1] = grad_point_temp[2 * i + 1];
    }
    return (float)(rot_giou - (2 * S1 - S) / S * (2 * S1 - S) / S / 4);
}

template <typename T>
__global__ void convex_giou_cuda_kernel(const int ex_n_boxes,
                                        const int gt_n_boxes, const T* ex_boxes,
                                        const T* gt_boxes, T* point_grad) {
  CUDA_1D_KERNEL_LOOP(index, ex_n_boxes) {
    const T* cur_box = ex_boxes + index * 18;
    const T* cur_gt_box = gt_boxes + index * 8;
    T* cur_grad = point_grad + index * 19;
    T giou = devrIoU(cur_box, cur_gt_box, cur_grad, threadIdx.x);
    cur_grad[18] = giou;
  }
}

HOST_DEVICE inline int lineCross(Point a, Point b, Point c, Point d, Point& p) {
  double s1, s2;
  s1 = cross(a, b, c);
  s2 = cross(a, b, d);
  if (sig(s1) == 0 && sig(s2) == 0) return 2;
  if (sig(s2 - s1) == 0) return 0;
  p.x = (c.x * s2 - d.x * s1) / (s2 - s1);
  p.y = (c.y * s2 - d.y * s1) / (s2 - s1);
  return 1;
}

HOST_DEVICE inline void polygon_cut(Point* p, int& n, Point a, Point b) {
  Point pp[MAXN];
  int m = 0;
  p[n] = p[0];
  for (int i = 0; i < n; i++) {
    if (sig(cross(a, b, p[i])) > 0) {
      pp[m] = p[i];
      m++;
    }
    if (sig(cross(a, b, p[i])) != sig(cross(a, b, p[i + 1]))) {
      lineCross(a, b, p[i], p[i + 1], pp[m]);
      m++;
    }
  }
  n = 0;
  for (int i = 0; i < m; i++) {
    if (!i || !(point_same(pp[i], pp[i - 1]))) {
      p[n] = pp[i];
      n++;
    }
  }

  while (n > 1 && point_same(p[n - 1], p[0])) n--;
}

HOST_DEVICE inline double intersectArea(Point a, Point b, Point c, Point d) {
  Point o(0, 0);
  int s1 = sig(cross(o, a, b));
  int s2 = sig(cross(o, c, d));
  if (s1 == 0 || s2 == 0) return 0.0;
  if (s1 == -1) {
    Point* i = &a;
    Point* j = &b;
    swap1(i, j);
  }
  if (s2 == -1) {
    Point* i = &c;
    Point* j = &d;
    swap1(i, j);
  }
  Point p[10] = {o, a, b};
  int n = 3;

  polygon_cut(p, n, o, c);
  polygon_cut(p, n, c, d);
  polygon_cut(p, n, d, o);
  double res = area(p, n);
  if (s1 * s2 == -1) res = -res;
  return res;
}
HOST_DEVICE inline double intersectAreaO(Point* ps1, int n1, Point* ps2,
                                        int n2) {
  if (area(ps1, n1) < 0) reverse1(ps1, n1);
  if (area(ps2, n2) < 0) reverse1(ps2, n2);
  ps1[n1] = ps1[0];
  ps2[n2] = ps2[0];
  double res = 0;
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      res += intersectArea(ps1[i], ps1[i + 1], ps2[j], ps2[j + 1]);
    }
  }
  return res;
}

template <typename T>
HOST_DEVICE inline float devrIoU(T const* const p, T const* const q) {
  Point ps1[MAXN], ps2[MAXN];
  Point convex[MAXN];
  for (int i = 0; i < 9; i++) {
    convex[i].x = (double)p[i * 2];
    convex[i].y = (double)p[i * 2 + 1];
  }
  int n_convex = 9;
  int points_to_convex_ind[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
  Jarvis_and_index(convex, n_convex, points_to_convex_ind);
  int n1 = n_convex;
  for (int i = 0; i < n1; i++) {
    ps1[i].x = (double)convex[i].x;
    ps1[i].y = (double)convex[i].y;
  }
  int n2 = 4;
  for (int i = 0; i < n2; i++) {
    ps2[i].x = (double)q[i * 2];
    ps2[i].y = (double)q[i * 2 + 1];
  }
  double inter_area = intersectAreaO(ps1, n1, ps2, n2);
  double S_pred = area(ps1, n1);
  double union_area = fabs(S_pred) + fabs(area(ps2, n2)) - inter_area;
  double iou = inter_area / union_area;
  return (float)iou;
}

template <typename T>
__global__ void convex_iou_cuda_kernel(const int ex_n_boxes,
                                       const int gt_n_boxes, const T* ex_boxes,
                                       const T* gt_boxes, T* iou) {
  CUDA_1D_KERNEL_LOOP(index, ex_n_boxes) {
    const T* cur_box = ex_boxes + index * 18;
    for (int i = 0; i < gt_n_boxes; i++) {
      iou[index * gt_n_boxes + i] = devrIoU(cur_box, gt_boxes + i * 8);
    }
  }
}
#endif  // CONVEX_IOU_CUDA_KERNEL_CUH
