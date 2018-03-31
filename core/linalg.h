// Copyright 2018 Kentaro Minami. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

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
#include <Eigen/Cholesky>

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
// - bases.size() == coef.size() == indices.size() == m
// - bases[0], ..., bases[m-1] are vectors with equal dimension
// - coef[0], ..., coef[m-1] are non-negative, and the sum of coefs equals to 1
std::size_t reduce_bases(std::size_t n, std::vector<std::vector<double>>& bases,
                        std::vector<double>& coef)
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

std::size_t reduce_bases_with_order_swap(std::size_t n, std::vector<std::vector<double>>& bases,
                        std::vector<double>& coef, std::vector<std::vector<std::size_t>>& orders)
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
          orders[i - 1] = std::move(orders[m - 1]);
          Y.col(i - 1) = Y.col(m - 1);
        }
        coef.pop_back();
        bases.pop_back();
        orders.pop_back();
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

void calc_affine_minimizer(std::size_t n, const std::vector<std::vector<double>>& bases, std::vector<double>& coef,
                          std::vector<double>& solution) {
  std::size_t m = bases.size();
  Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(n, m);
  for (std::size_t ic = 0; ic < m; ++ic) {
    for (std::size_t ir = 0; ir < n; ++ir) {
      Y(ir, ic) = bases[ic][ir];
    }
  }
  Eigen::MatrixXd YTY = Y.transpose() * Y;
  Eigen::VectorXd ones = Eigen::VectorXd::Ones(m);
  Eigen::VectorXd alpha = YTY.ldlt().solve(ones);
  double normalize = alpha.sum();
  alpha /= normalize;
  coef.resize(m);
  for (std::size_t i = 0; i < m; ++i) {
    coef[i] = alpha(i);
  }
  Eigen::VectorXd solvxd = Y * alpha;
  solution.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    solution[i] = solvxd(i);
  }
}

}

#endif
