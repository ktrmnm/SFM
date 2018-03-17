#ifndef LINALG_H
#define LINALG_H

#include <vector>
//#include <list>
#include <utility>
#include <limits>
//#include <iostream>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>

#include "utils.h"

namespace linalg {

//enum LinearSolverStrategy struct { LUStable, QRStable, QRFast };

template <typename T>
Eigen::VectorXd stdvec2vxd(std::size_t n, const std::vector<T>& vec) {
  Eigen::VectorXd vxd = Eigen::VectorXd::Zero(n);
  for (std::size_t i = 0; i < n; ++i) {
    vxd(i) = static_cast<double>(vec[i]);
  }
  return vxd;
}

template <typename T>
Eigen::MatrixXd stdvecvec2mxd(std::size_t n, std::size_t m, const std::vector<T>& mat) {
  Eigen::MatrixXd mxd = Eigen::MatrixXd::Zero(n, m);
  for (std::size_t ic = 0; ic < m; ++ic) {
    for (std::size_t ir = 0; ir < n; ++ ir) {
      mxd(ir, ic) = static_cast<double>(mat[ic][ir]);
    }
  }
  return mxd;
}

// Reduce number of bases
// Preconditions:
// - bases.size() == coef.size() == m
// - bases[0], ..., bases[m-1] are vectors with equal dimension
// - coef[0], ..., coef[m-1] are non-negative, and the sum of coefs equals to 1
std::size_t reduce_bases(std::size_t n,
                  std::vector<std::vector<double>>& bases, std::vector<double>& coef)
{
  auto m = bases.size();
  Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(n + 1, m);

  for (std::size_t ic = 0; ic < m; ++ic) {
    Y(0, ic) = 1;
    for (std::size_t ir = 0; ir < n; ++ir) {
      Y(ir + 1, ic) = bases[ic][ir];
    }
  }

  auto decomp = Y.fullPivLu();
  bool is_injective = decomp.isInjective();

  double tol = decomp.threshold();

  auto _delete_zero = [&]() {
    for (std::size_t i = m; i != 0; --i) {
      if (utils::is_abs_close(coef[i - 1], 0.0, tol)) {
        //std::cout << "delete " << i - 1 << std::endl;
        if (i - 1 != m - 1) {
          coef[i - 1] = coef[m - 1];
          bases[i - 1] = std::move(bases[m - 1]);
          Y.col(i - 1) = Y.col(m - 1);
        }
        coef.pop_back();
        bases.pop_back();
        Y.conservativeResize(Eigen::NoChange, m - 1);
        m--;
      }
    }
  };

  _delete_zero();

  while (!is_injective) {
    auto mu = decomp.kernel().col(0);
    //std::cout << "mu = " << mu << std::endl;

    double update = std::numeric_limits<double>::max();
    std::size_t i_erase = m;
    for (std::size_t i = 0; i < m; ++i) {
      if (mu(i) > 0 && !utils::is_abs_close(mu(i), 0.0, tol)) {
        double theta = coef[i] / mu(i);
        if (theta < update) {
          update = theta;
          i_erase = i;
        }
      }
    }
    for (std::size_t i = 0; i < m; ++i) {
      coef[i] = coef[i] - update * mu(i);
    }

    _delete_zero();

    decomp = Y.fullPivLu();
    is_injective = decomp.isInjective();
  }
  //std::cout << "Y = \n" << Y << std::endl;
  return m; // new number of bases
}



}

#endif
