#ifndef STAN_MATH_PRIM_MAT_FUN_MRRR_HPP
#define STAN_MATH_PRIM_MAT_FUN_MRRR_HPP

#include <Eigen/Dense>

#include <queue>
#include <vector>
#include <algorithm>
#include <limits>

#include <cmath>

namespace stan {
namespace math {
namespace internal {

template <typename T>
struct impls {
  static T sign(T a) { return std::copysign(T(1), a); }

  static bool isinfinite(T a) { return std::isinf(a); }

  static bool isNaN(T a) { return std::isnan(a); }
};

template <typename T>
struct impls<std::complex<T>> {
  static std::complex<T> sign(std::complex<T> a) {
    return {std::copysign(std::sqrt(T(2.0)), std::real(a)),
            std::copysign(std::sqrt(T(2.0)), std::imag(a))};
  }

  static bool isinfinite(std::complex<T> a) {
    return std::isinf(std::real(a)) || std::isinf(std::imag(a));
  }

  static bool isNaN(std::complex<T> a) {
    return std::isinf(std::real(a)) || std::isinf(std::imag(a));
  }
};

template <typename T>
T sign(T a) {
  return impls<T>::sign(a);
}

template <typename T>
bool isinfinite(T a) {
  return impls<T>::isinfinite(a);
}

template <typename T>
bool isNaN(T a) {
  return impls<T>::isNaN(a);
}

template <typename Real>
struct constants {
  static constexpr Real perturbation_range = std::numeric_limits<Real>::epsilon() * 10;
  static constexpr int bisect_k = 64 / sizeof(Real);
};

/**
 * Generates a random number for perturbing a relatively robust representation
 * @tparam Real real type of return
 * @return A uniformly distributed random number between `1 - perturbation_range
 * / 2` and `1 + perturbation_range / 2`.
 */
template <typename Real>
inline Real get_random_perturbation_multiplier() {
  static const Real rand_norm = constants<Real>::perturbation_range / RAND_MAX;
  static const Real almost_one = Real(1) - constants<Real>::perturbation_range * Real(0.5);
  return almost_one + std::rand() * rand_norm;
}

/**
 * Calculates LDL decomposition of a shifted triagonal matrix T. D is diagonal,
 * L is lower unit triangular (diagonal elements are 1, all elements except
 * diagonal and subdiagonal are 0),T - shift * I = L * D * L^T. Also calculates
 * element growth of D: sum(abs(D)) / abs(sum(D)).
 * @tparam Scalar scalar used
 * @tparam Real real valued scalar
 * @param diagonal Diagonal of T
 * @param subdiagonal Subdiagonal of T.
 * @param shift Shift.
 * @param[out] l Subdiagonal of L.
 * @param[out] d_plus Diagonal of D.
 * @return Element growth.
 */
template <typename Scalar, typename Real = typename Eigen::NumTraits<Scalar>::Real>
Real get_ldl(
    const Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic, 1>> diagonal,
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> subdiagonal,
    const Real shift,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& l,
    Eigen::Matrix<Real, Eigen::Dynamic, 1>& d_plus) {
  using std::conj;
  using std::fabs;
  using std::real;
  d_plus[0] = diagonal[0] - shift;
  Real element_growth = fabs(d_plus[0]);
  Real element_growth_denominator = d_plus[0];
  for (int i = 0; i < subdiagonal.size(); i++) {
    l[i] = subdiagonal[i] / d_plus[i];
    d_plus[i] *= get_random_perturbation_multiplier<Real>();
    d_plus[i + 1] = diagonal[i + 1] - shift - real(conj(l[i]) * subdiagonal[i]);
    l[i] *= get_random_perturbation_multiplier<Real>();
    element_growth += fabs(d_plus[i + 1]);
    element_growth_denominator += d_plus[i + 1];
  }
  d_plus[subdiagonal.size()] *= get_random_perturbation_multiplier<Real>();
  return element_growth / fabs(element_growth_denominator);
}

/**
 * Shifts a LDL decomposition. The algorithm is sometimes called stationary
 * quotients-differences with shifts (stqds). D and D+ are diagonal, L and L+
 * are lower unit triangular (diagonal elements are 1, all elements except
 * diagonal and subdiagonal are 0). L * D * L^T - shift * I = L+ * D * L+^T.
 * Also calculates element growth of D+: sum(abs(D+)) / abs(sum(D+)).
 * @tparam Scalar scalar used
 * @tparam Real real valued scalar
 * @param l Subdiagonal of L.
 * @param d Diagonal of D.
 * @param shift Shift.
 * @param[out] l_plus Subdiagonal of L+.
 * @param[out] d_plus Diagonal of D+.
 * @return Element growth.
 */
template <typename Scalar, typename Real = typename Eigen::NumTraits<Scalar>::Real>
Real get_shifted_ldl(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& l,
                     const Eigen::Matrix<Real, Eigen::Dynamic, 1>& d,
                     const Real shift,
                     Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& l_plus,
                     Eigen::Matrix<Real, Eigen::Dynamic, 1>& d_plus) {
  using std::conj;
  using std::fabs;
  using std::isinf;
  using std::norm;
  using std::real;
  const int n = l.size();
  Real s = -shift;
  Real element_growth = 0;
  Real element_growth_denominator = 0;
  for (int i = 0; i < n; i++) {
    d_plus[i] = s + d[i];
    element_growth += fabs(d_plus[i]);
    element_growth_denominator += d_plus[i];
    l_plus[i] = l[i] * (d[i] / d_plus[i]);
    if (isinf(d_plus[i]) && isinf(s)) {  // this happens if d_plus[i]==0 -> in next iteration d_plus==inf and s==inf
      s = norm(l[i]) * d[i] - shift;
    } else {
      s = real(l_plus[i] * conj(l[i])) * s - shift;
    }
  }
  d_plus[n] = s + d[n];
  element_growth += fabs(d_plus[n]);
  return element_growth / fabs(element_growth_denominator);
}

/**
 * Calculates shifted LDL and UDU factorizations. Combined with twist index they
 * form twisted factorization for calculation of an eigenvector corresponding to
 * eigenvalue that is equal to the shift. Tha algorithm is sometimes called
 * diferential twisted quotient-differences with shifts (dtwqds). L * D * L^T -
 * shift * I = L+ * D+ * L+^T = U- * D- * U-^T D, D+ and D- are diagonal, L and
 * L+ are lower unit triangular (diagonal elements are 1, all elements except
 * diagonal and subdiagonal are 0), U- is upper unit triangular (diagonal
 * elements are 1, all elements except diagonal and superdiagonal are 0)
 * @tparam Scalar scalar used
 * @tparam Real real valued scalar
 * @param l Subdiagonal of L.
 * @param d Diagonal of D.
 * @param shift Shift.
 * @param[out] l_plus Subdiagonal of L+.
 * @param[out] u_minus Superdiagonal of U-.
 * @return Twist index.
 */
template <typename Scalar, typename Real = typename Eigen::NumTraits<Scalar>::Real>
int get_twisted_factorization(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& l,
    const Eigen::Matrix<Real, Eigen::Dynamic, 1>& d,
    const Real shift,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& l_plus,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& u_minus) {
  using std::conj;
  using std::copysign;
  using std::fabs;
  using std::norm;
  using std::real;
  const int n = l.size();
  // calculate shifted ldl
  Eigen::Matrix<Real, Eigen::Dynamic, 1> s(n + 1);
  s[0] = -shift;
  for (int i = 0; i < n; i++) {
    Real d_plus = s[i] + d[i];
    l_plus[i] = l[i] * (d[i] / d_plus);
    if (isNaN(l_plus[i])) {  // d_plus==0: one (or both) of d[i], l[i] is very close to 0
      if (norm(l[i]) < norm(d[i])) {
        l_plus[i] = d[i] * sign(l[i] * d_plus);
      } else {
        l_plus[i] = l[i] * sign(d[i] * d_plus);
      }
    }
    s[i + 1] = real(l_plus[i] * conj(l[i])) * s[i] - shift;
    if (isNaN(s[i + 1])) {
      if (norm(l_plus[i]) > norm(s[i])) {  // l_plus[i]==inf
        if (norm(s[i]) > norm(l[i])) { // l[i]==0
          s[i + 1] = s[i] * sign(d[i] * d_plus) - shift;
        } else {  // s[i]==0
          s[i + 1] = Eigen::numext::abs(l[i]) * sign(s[i + 1]) - shift;
        }
      } else { // s[i]==inf
        if (norm(l_plus[i]) > norm(l[i])) { // l[i]==0
          s[i + 1] = Eigen::numext::abs(l_plus[i]) * sign(s[i + 1]) - shift;
        } else { // l_plus[i]==0
          s[i + 1] = norm(l[i]) * sign(s[i] * d[i] * d_plus) - shift;
        }
      }
    }
  }
  // calculate shifted udu and twist index
  Real p = d[n] - shift;
  Real min_gamma = fabs(s[n] + d[n]);
  int twist_index = n;

  for (int i = n - 1; i >= 0; i--) {
    Real d_minus = d[i] * norm(l[i]) + p;
    Real t = d[i] / d_minus;
    u_minus[i] = l[i] * t;
    if (isNaN(u_minus[i])) {
      if (isNaN(t)) {
        t = copysign(1., d[i] * d_minus);
        u_minus[i] = l[i] * t;
      } else {  // t==inf, l[i]==0
        u_minus[i] = d[i] * sign(l[i] * t);
      }
    }
    Real gamma = fabs(s[i] + t * p);
    if (isNaN(gamma)) {  // t==inf, p==0 OR t==0, p==inf
      Real d_sign = d[i] * copysign(1., d_minus * t);
      gamma = fabs(s[i] + d_sign);
      p = d_sign - shift;
    } else {  // general case
      p = p * t - shift;
    }
    if (gamma < min_gamma) {
      min_gamma = gamma;
      twist_index = i;
    }
  }
  return twist_index;
}

/**
 * Calculates Sturm count of a LDL decomposition of a tridiagonal matrix -
 * number of eigenvalues larger or equal to shift. Uses stqds - calculation of
 * shifted LDL decomposition algorithm and counts number of positive elements in
 * D.
 * @tparam Scalar scalar used
 * @tparam Real real valued scalar
 * @param l Subdiagonal of L.
 * @param d Diagonal of D.
 * @param shift Shift.
 * @return Sturm count.
 */
template <typename Scalar, typename Real = typename Eigen::NumTraits<Scalar>::Real>
int get_sturm_count_ldl(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& l,
                        const Eigen::Matrix<Real, Eigen::Dynamic, 1>& d,
                        const Real shift) {
  using std::isinf;
  using std::norm;
  const int n = l.size();
  Real s = -shift;
  Real d_plus;
  int count = 0;
  for (int i = 0; i < n; i++) {
    d_plus = s + d[i];
    count += d_plus >= 0;
    if (isinf(d_plus) && isinf(s)) {  // this happens if d_plus==0 -> in next iteration d_plus==inf and s==inf
      s = norm(l[i]) * d[i] - shift;
    } else {
      s = norm(l[i]) * s * (d[i] / d_plus) - shift;
    }
  }
  d_plus = s + d[n];
  count += d_plus >= 0;
  return count;
}

/**
 * Refines bounds on the i-th largest eigenvalue of LDL decomposition of a
 * matrix using bisection.
 * @tparam Scalar scalar used
 * @tparam Real real valued scalar
 * @param l Subdiagonal of L.
 * @param d Diagonal of D.
 * @param[in,out] low Low bound on the eigenvalue.
 * @param[in,out] high High bound on the eigenvalue.
 * @param i i-th eigenvalue
 */
template <typename Scalar, typename Real = typename Eigen::NumTraits<Scalar>::Real>
void eigenval_bisect_refine(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& l,
                            const Eigen::Matrix<Real, Eigen::Dynamic, 1>& d,
                            Real& low, Real& high, const int i) {
  using std::fabs;
  const Real eps = std::numeric_limits<Real>::epsilon() * 3;
  while (!(fabs((high - low) / (high + low)) < eps)) {  // if the condition was flipped it would be wrong for the case where division yields NaN
    Real mid = (high + low) * Real(0.5);
    if (get_sturm_count_ldl(l, d, mid) > i) {
      low = mid;
    } else {
      high = mid;
    }
  }
}

/**
 * Calculates bounds on eigenvalues of a symmetric tridiagonal matrix T using
 * Gresgorin discs.
 * @tparam Scalar scalar used
 * @tparam Real real valued scalar
 * @param diagonal Diagonal of T
 * @param subdiagonal Subdiagonal of T
 * @param[out] min_eigval Lower bound on eigenvalues.
 * @param[out] max_eigval Upper bound on eigenvalues.
 */
template <typename Scalar, typename Real = typename Eigen::NumTraits<Scalar>::Real>
void get_gresgorin(
    const Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic, 1>> diagonal,
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> subdiagonal,
    Real& min_eigval, Real& max_eigval) {
  using std::fabs;
  const int n = diagonal.size();
  min_eigval = diagonal[0] - Eigen::numext::abs(subdiagonal[0]);
  max_eigval = diagonal[0] + Eigen::numext::abs(subdiagonal[0]);
  for (int i = 1; i < n - 1; i++) {
    min_eigval = std::min(min_eigval, diagonal[i] - Eigen::numext::abs(subdiagonal[i]) - Eigen::numext::abs(subdiagonal[i - 1]));
    max_eigval = std::max(max_eigval, diagonal[i] + Eigen::numext::abs(subdiagonal[i]) + Eigen::numext::abs(subdiagonal[i - 1]));
  }
  min_eigval = std::min(min_eigval, diagonal[n - 1] - Eigen::numext::abs(subdiagonal[n - 2]));
  max_eigval = std::max(max_eigval, diagonal[n - 1] + Eigen::numext::abs(subdiagonal[n - 2]));
}

/**
 * Calculates lower Sturm count of a tridiagonal matrix T - number of
 * eigenvalues lower than shift for up to Bisect_K different shifts.
 * @tparam Real real valued scalar
 * @tparam Bisect_K maximal number of sturm counts to calculate.
 * @param diagonal Diagonal of T.
 * @param subdiagonal_norm Squared norm of subdiagonal elements of T.
 * @param shifts Up to Bisect_K different shifts. First `n_valid` are used.
 * @param n_valid How many Sturm counts to actually compute.
 * @return Array of Sturm counts of size Bisect_K. First `n_valid` are actual
 * results.
 */
template <typename Real, int Bisect_K = constants<Real>::bisect_k>
Eigen::Array<int, Bisect_K, 1> get_sturm_count_T_vec(
    const Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic, 1>> diagonal,
    const Eigen::Matrix<Real, Eigen::Dynamic, 1>& subdiagonal_norm,
    const Eigen::Array<Real, constants<Real>::bisect_k, 1>& shifts,
    const int n_valid) {
  Eigen::Array<Real, Bisect_K, 1> d;
  d.head(n_valid) = diagonal[0] - shifts.head(n_valid);
  Eigen::Array<int, Bisect_K, 1> counts;
  counts.head(n_valid) = (d.head(n_valid) < 0).template cast<int>();
  for (int j = 1; j < diagonal.size(); j++) {
    d.head(n_valid) = diagonal[j] - shifts.head(n_valid) - subdiagonal_norm[j - 1] / d.head(n_valid);
    counts.head(n_valid) += (d.head(n_valid) < 0).template cast<int>();
  }
  return counts;
}

template <typename Real>
struct bisection_task {
  int start, end;
  Real low, high;
};

/**
 * Calculates eigenvalues of tridiagonal matrix T using bisection.
 * @tparam Scalar scalar used
 * @tparam Real real valued scalar
 * @tparam Bisect_K maximal number of sturm counts to calculate at once.
 * @param diagonal Diagonal of T.
 * @param subdiagonal_squared Squared elements of the subdiagonal.
 * @param min_eigval Lower bound on all eigenvalues.
 * @param max_eigval Upper bound on all eigenvalues.
 * @param[out] low Lower bounds on eigenvalues.
 * @param[out] high Upper bounds on eigenvalues.
 */
template <typename Scalar, typename Real = typename Eigen::NumTraits<Scalar>::Real, int Bisect_K = constants<Real>::bisect_k>
void eigenvals_bisect(
    const Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic, 1>> diagonal,
    const Eigen::Matrix<Real, Eigen::Dynamic, 1>& subdiagonal_squared,
    const Real min_eigval, const Real max_eigval,
    Eigen::Matrix<Real, Eigen::Dynamic, 1>& low,
    Eigen::Matrix<Real, Eigen::Dynamic, 1>& high) {
  using std::fabs;
  using task = bisection_task<Real>;
  const int n = diagonal.size();
  const Real eps = std::numeric_limits<Real>::epsilon() * 3;

  std::queue<task> task_queue;
  task_queue.push(task{0, n, min_eigval, max_eigval});
  while (!task_queue.empty()) {
    const int n_valid = std::min(Bisect_K, static_cast<int>(task_queue.size()));
    Eigen::Array<Real, Bisect_K, 1> shifts;
    task t[Bisect_K];
    for (int i = 0; i < n_valid; i++) {
      t[i] = task_queue.front();
      task_queue.pop();
    }
    for (int i = 0; i < Bisect_K; i++) {
      const int task_idx = i % n_valid;
      const int idx_in_task = i / n_valid;
      const int task_total = Bisect_K / n_valid + (Bisect_K % n_valid > task_idx);
      shifts[i] = t[task_idx].low + (t[task_idx].high - t[task_idx].low) * (idx_in_task + 1.) / (task_total + 1);
    }
    const Eigen::Array<int, Bisect_K, 1> counts = get_sturm_count_T_vec(diagonal, subdiagonal_squared, shifts, Bisect_K);
    for (int i = 0; i < n_valid; i++) {
      if (counts[i] >= t[i].start + 1) {
        if ((t[i].high - shifts[i]) / fabs(shifts[i]) > eps && shifts[i] - t[i].low > std::numeric_limits<Real>::min()) {
          task_queue.push({t[i].start, counts[i], t[i].low, shifts[i]});
        } else {
          const int n_eq = counts[i] - t[i].start;
          low.segment(t[i].start, n_eq) = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Constant(n_eq, t[i].low);
          high.segment(t[i].start, n_eq) = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Constant(n_eq, shifts[i]);
        }
      }
    }
    for (int i = 0; i < Bisect_K; i++) {
      const int task_idx = i % n_valid;
      const int idx_in_task = i / n_valid;
      const int task_total = Bisect_K / n_valid + (Bisect_K % n_valid > task_idx);
      int my_end = t[task_idx].end;
      Real my_high = t[task_idx].high;
      if (i + n_valid < Bisect_K) {
        my_end = counts[i + n_valid];
        my_high = shifts[i + n_valid];
      }
      if (counts[i] <= my_end - 1) {
        if ((my_high - shifts[i]) / fabs(shifts[i]) > eps && my_high - shifts[i] > std::numeric_limits<Real>::min()) {
          task_queue.push({counts[i], my_end, shifts[i], my_high});
        } else {
          int my_start = t[task_idx].start;
          if (i - n_valid >= 0) {
            my_start = counts[i - n_valid];
          }
          const int n_eq = my_end - counts[i];
          low.segment(counts[i], n_eq) = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Constant(n_eq, shifts[i]);
          high.segment(counts[i], n_eq) = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Constant(n_eq, my_high);
        }
      }
    }
  }
  low = low.reverse().eval();
  high = high.reverse().eval();
}

/**
 * Calculates an eigenvector from twisted factorization T - shift * I = L+ * D+
 * * L+^T = U- * D- * U-^T.
 * @tparam Scalar scalar used
 * @tparam Real real valued scalar
 * @param l_plus Subdiagonal of the L+.
 * @param u_minus Superdiagonal of the U-.
 * @param subdiagonal Subdiagonal of T
 * @param i At which column of `eigenvecs` to store resulting vector.
 * @param twist_idx Twist index.
 * @param[out] eigenvectors Matrix in which to store resulting vector.
 */
template <typename Scalar, typename Real = typename Eigen::NumTraits<Scalar>::Real>
void calculate_eigenvector(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& l_plus,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& u_minus,
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& subdiagonal,
    const int i, const int twist_idx,
    Eigen::Ref<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>& eigenvectors) {
  auto vec = eigenvectors.col(i);
  const int n = vec.size();
  vec[twist_idx] = 1;
  for (int j = twist_idx + 1; j < n; j++) {
    if (vec[j - 1] != 0.) {
      vec[j] = -u_minus[j - 1] * vec[j - 1];
    } else {
      vec[j] = -subdiagonal[j - 2] * vec[j - 2] / subdiagonal[j - 1];
      if (isNaN(vec[j]) || isinfinite(vec[j])) {  // subdiagonal[j - 1]==0
        vec[j] = 0;
      }
    }
  }
  for (int j = twist_idx - 1; j >= 0; j--) {
    if (vec[j + 1] != 0.) {
      vec[j] = -std::conj(l_plus[j]) * vec[j + 1];
    } else {
      vec[j] = -subdiagonal[j + 1] * vec[j + 2] / subdiagonal[j];
      if (isNaN(vec[j]) || isinfinite(vec[j])) {  // subdiagonal[j]==0
        vec[j] = 0;
      }
    }
  }
  vec *= 1. / vec.norm();
}

/**
 * Finds good shift and shifts a LDL decomposition so as to keep element growth
 * low. L * D * L^T - shift * I = L2 * D2 * L2^T.
 * @tparam Scalar scalar used
 * @tparam Real real valued scalar
 * @param l Subdiagonal of L.
 * @param d Diagonal of D.
 * @param low Low bound on wanted shift.
 * @param high High bound on wanted shift.
 * @param max_ele_growth Maximum desired element growth. If no better options
 * are found it might be exceeded.
 * @param max_shift Maximal difference of shift from wanted bounds.
 * @param[out] l2 Subdiagonal of L2.
 * @param[out] d2 Diagonal of D2.
 * @param[out] shift Shift.
 * @param[out] min_element_growth Element growth achieved with resulting shift.
 */
template <typename Scalar, typename Real = typename Eigen::NumTraits<Scalar>::Real>
void find_shift(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& l,
                const Eigen::Matrix<Real, Eigen::Dynamic, 1>& d,
                const Real low, const Real high, const Real max_ele_growth, const Real max_shift,
                Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& l2,
                Eigen::Matrix<Real, Eigen::Dynamic, 1>& d2,
                Real& shift, Real& min_element_growth) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> l3(l2.size());
  Eigen::Matrix<Real, Eigen::Dynamic, 1> d3(d2.size());
  const std::vector<Real> shifts = {
      low,
      high - max_shift * Real(0.1),
      low + max_shift * Real(0.1),
      high - max_shift * Real(0.25),
      low + max_shift * Real(0.25),
      high - max_shift * Real(0.5),
      low + max_shift * Real(0.5),
      high - max_shift * Real(0.75),
      low + max_shift * Real(0.75),
      high - max_shift,
      low + max_shift,
  };
  min_element_growth = std::numeric_limits<Real>::infinity();
  for (Real sh : shifts) {
    const Real element_growth = get_shifted_ldl(l, d, sh, l3, d3);
    if (element_growth < min_element_growth) {
      l2.swap(l3);
      d2.swap(d3);
      shift = sh;
      min_element_growth = element_growth;
      if (element_growth <= max_ele_growth) {
        break;
      }
    }
  }
}

/**
 * Finds a good value for shift of the initial LDL factorization T - shift * I =
 * L * D * L^T.
 * @tparam Scalar scalar used
 * @tparam Real real valued scalar
 * @param diagonal Diagonal of T.
 * @param subdiagonal Subdiagonal of T.
 * @param l0 Subdiagonal of L.
 * @param d0 Diagonal of D.
 * @param min_eigval Lower bound on eigenvalues of T.
 * @param max_eigval High bound on eigenvalues of T
 * @param max_ele_growth Maximum desired element growth.
 * @return shift
 */
template <typename Scalar, typename Real = typename Eigen::NumTraits<Scalar>::Real>
Real find_initial_shift(
    const Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic, 1>> diagonal,
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> subdiagonal,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& l0,
    Eigen::Matrix<Real, Eigen::Dynamic, 1>& d0,
    const Real min_eigval, const Real max_eigval, const Real max_ele_growth) {
  Real shift = 0;
  if (min_eigval > 0) {
    shift = Real(0.9) * min_eigval;
  } else if (max_eigval <= 0) {
    shift = Real(0.9) * max_eigval;
  }
  Real element_growth = get_ldl(diagonal, subdiagonal, shift, l0, d0);
  if (element_growth < max_ele_growth) {
    return shift;
  }
  Real plus = (max_eigval - min_eigval) * 10 * std::numeric_limits<Real>::epsilon();
  while (!(element_growth < max_ele_growth)) {  // if condition is flipped it would be wrong for the case where element_growth is nan
    plus *= -2;
    element_growth = get_ldl(diagonal, subdiagonal, shift + plus, l0, d0);
  }
  return shift + plus;
}

template <typename Scalar, typename Real>
struct mrrr_task {
  int start, end;
  Real shift;  // total shift, not just the last one
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> l;
  Eigen::Matrix<Real, Eigen::Dynamic, 1> d;
  int level;
};

/**
 * Calculates eigenvalues and eigenvectors of a irreducible tridiagonal matrix T
 * using MRRR algorithm. Use `tridiagonal_eigensolver` if any subdiagonal
 * element might be (very close to) zero.
 * @tparam Scalar scalar used
 * @tparam Real real valued scalar
 * @param diagonal Diagonal of of T.
 * @param subdiagonal Subdiagonal of T.
 * @param[out] eigenvalues Eigenvlues.
 * @param[out] eigenvectors Eigenvectors.
 * @param min_rel_sep Minimal relative separation of eigenvalues before
 * computing eigenvectors.
 * @param max_ele_growth Maximal desired element growth of LDL decompositions.
 */
template <typename Scalar, typename Real>
void mrrr(
    const Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic, 1>> diagonal,
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> subdiagonal,
    Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, 1>> eigenvalues,
    Eigen::Ref<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> eigenvectors,
    const Real min_rel_sep = Real(1e-1), const Real max_ele_growth = Real(2)) {
  using std::copysign;
  using std::fabs;
  using task = mrrr_task<Scalar, Real>;
  const Real shift_error = std::numeric_limits<Real>::epsilon() * 100;
  const int n = diagonal.size();
  Eigen::Matrix<Real, Eigen::Dynamic, 1> high(n), low(n);
  Real min_eigval;
  Real max_eigval;
  get_gresgorin(diagonal, subdiagonal, min_eigval, max_eigval);
  Eigen::Matrix<Real, Eigen::Dynamic, 1> d(n), d0(n);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> l(n - 1), l0(n - 1);
  const Real shift0 = find_initial_shift(diagonal, subdiagonal, l0, d0, min_eigval, max_eigval, max_ele_growth);
  for (int i = 0; i < n; i++) {
    if (i != n - 1) {
      l[i] = l0[i] * get_random_perturbation_multiplier<Real>();
    }
    d[i] = d0[i] * get_random_perturbation_multiplier<Real>();
  }
  const Eigen::Matrix<Real, Eigen::Dynamic, 1> subdiagonal_norm = (subdiagonal.array().conjugate() * subdiagonal.array()).real();

  eigenvals_bisect<Scalar, Real>(diagonal, subdiagonal_norm, min_eigval, max_eigval, low, high);
  eigenvalues = (high + low) * Real(0.5);
  low.array() -= shift0;
  high.array() -= shift0;
  for (int i = 0; i < n; i++) {
    low[i] = low[i] * (1 - copysign(constants<Real>::perturbation_range * n, low[i]));
    high[i] = high[i] * (1 + copysign(constants<Real>::perturbation_range * n, high[i]));
    eigenval_bisect_refine(l, d, low[i], high[i], i);
  }
  std::queue<task> block_queue;
  block_queue.push(task{0, n, shift0, std::move(l), std::move(d), 0});
  l.resize(n - 1);  // after move out
  d.resize(n);
  while (!block_queue.empty()) {
    const task block = block_queue.front();
    block_queue.pop();
    Real shift = std::numeric_limits<Real>::infinity();
    Real min_element_growth = std::numeric_limits<Real>::infinity();
    Eigen::Matrix<Real, Eigen::Dynamic, 1> d2(n);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> l2(n - 1), l_plus(n - 1), u_minus(n - 1);
    for (int i = block.start; i < block.end; i++) {
      // find eigenvalue cluster size
      int cluster_end;
      for (cluster_end = i + 1; cluster_end < block.end; cluster_end++) {
        const int prev = cluster_end - 1;
        const Real end_threshold = low[prev] * (1 - copysign(shift_error, low[prev]));
        if (high[cluster_end] < end_threshold) {
          break;
        }
      }
      cluster_end--;  // now this is the index of the last element of the cluster
      if (cluster_end - i > 0) {  // cluster
        const Real max_shift = (high[i] - low[cluster_end]) * 10;
        Real next_shift, min_ele_growth;
        find_shift(block.l, block.d, low[cluster_end], high[i], max_ele_growth, max_shift, l, d, next_shift, min_ele_growth);
        for (int j = i; j <= cluster_end; j++) {
          low[j] = low[j] * (1 - copysign(shift_error, low[j])) - next_shift;
          high[j] = high[j] * (1 + copysign(shift_error, high[j])) - next_shift;
          eigenval_bisect_refine(l, d, low[j], high[j], j);
        }
        block_queue.push(task{i, cluster_end + 1, block.shift + next_shift, std::move(l), std::move(d), block.level + 1});
        l.resize(n - 1);  // after move out
        d.resize(n);

        i = cluster_end;
      } else {  // isolated eigenvalue
        int twist_idx;
        const Real low_gap = i == block.start ? std::numeric_limits<Real>::infinity() : low[i - 1] - high[i];
        const Real high_gap = i == block.end - 1 ? std::numeric_limits<Real>::infinity() : low[i] - high[i + 1];
        const Real min_gap = std::min(low_gap, high_gap);
        const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* l_ptr;
        const Eigen::Matrix<Real, Eigen::Dynamic, 1>* d_ptr;
        if (!(fabs(min_gap / ((high[i] + low[i]) * Real(0.5))) > min_rel_sep)) {
          if (!(fabs(min_gap / ((high[i] + low[i]) * Real(0.5) - shift)) > min_rel_sep && min_element_growth < max_ele_growth)) {
            const Real max_shift = min_gap / min_rel_sep;
            find_shift(block.l, block.d, low[i], high[i], max_ele_growth, max_shift, l2, d2, shift, min_element_growth);
          }
          low[i] = low[i] * (1 - copysign(shift_error, low[i])) - shift;
          high[i] = high[i] * (1 + copysign(shift_error, high[i])) - shift;
          eigenval_bisect_refine(l2, d2, low[i], high[i], i);
          l_ptr = &l2;
          d_ptr = &d2;
        } else {
          l_ptr = &block.l;
          d_ptr = &block.d;
        }
        twist_idx = get_twisted_factorization(*l_ptr, *d_ptr, (low[i] + high[i]) * Real(0.5), l_plus, u_minus);
        calculate_eigenvector(l_plus, u_minus, subdiagonal, i, twist_idx, eigenvectors);
      }
    }
  }
}

/**
 * Sorts eigenpairs in place in ascending order of eigenvalues.
 * @tparam Scalar scalar used
 * @tparam Real real valued scalar
 * @param eigenvalues eigenvalues
 * @param eigenvectors eigenvectors
 */
template <typename Scalar, typename Real = typename Eigen::NumTraits<Scalar>::Real,
    typename = typename std::enable_if<std::is_arithmetic<Real>::value>::type>
void sort_eigenpairs(Eigen::Matrix<Real, Eigen::Dynamic, 1>& eigenvalues,
                     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& eigenvectors){
  int n = eigenvalues.size();
  Eigen::VectorXi indices(n);
  for(int i=0;i<n;i++){
    indices[i]=i;
  }
  std::sort(indices.data(), indices.data()+n,
      [&eigenvalues](int a, int b){return eigenvalues[a] < eigenvalues[b];});
  Eigen::Matrix<Real, Eigen::Dynamic, 1> eigenvalues2(n);
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigenvectors2(n,n);
  for(int i=0;i<n;i++){
    eigenvalues2[i] = eigenvalues[indices[i]];
    eigenvectors2.col(i) = eigenvectors.col(indices[i]);
  }
  eigenvalues.swap(eigenvalues2);
  eigenvectors.swap(eigenvectors2);
}

/**
 * Calculates eigenvalues and eigenvectors of a tridiagonal matrix T using MRRR
 * algorithm. If a subdiagonal element is close to zero compared to neighbors on
 * diagonal the problem can be split into smaller ones.
 * @tparam Scalar scalar used
 * @tparam Real real valued scalar
 * @param diagonal Diagonal of of T.
 * @param subdiagonal Subdiagonal of T.
 * @param[out] sorted eigenvalues Eigenvlues.
 * @param[out] eigenvectors Eigenvectors.
 * @param split_threshold Threshold for splitting the problem
 */
template <typename Scalar, typename Real = typename Eigen::NumTraits<Scalar>::Real,
    typename = typename std::enable_if<std::is_arithmetic<Real>::value>::type>
void tridiagonal_eigensolver(
    const Eigen::Matrix<Real, Eigen::Dynamic, 1>& diagonal,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& subdiagonal,
    Eigen::Matrix<Real, Eigen::Dynamic, 1>& eigenvalues,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& eigenvectors,
    const Real split_threshold = 1e-12) {
  const int n = diagonal.size();
  eigenvectors.resize(n, n);
  eigenvalues.resize(n);
  int last = 0;
  for (int i = 0; i < subdiagonal.size(); i++) {
    if (Eigen::numext::abs(subdiagonal[i] / diagonal[i]) < split_threshold && Eigen::numext::abs(subdiagonal[i] / diagonal[i + 1]) < split_threshold) {
      eigenvectors.block(last, i + 1, i + 1 - last, n - i - 1) = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Constant(i + 1 - last, n - i - 1, 0);
      eigenvectors.block(i + 1, last, n - i - 1, i + 1 - last) = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Constant(n - i - 1, i + 1 - last, 0);
      if (last == i) {
        eigenvectors(last, last) = 1;
        eigenvalues[last] = diagonal[last];
      } else {
        mrrr<Scalar, Real>(
            diagonal.segment(last, i + 1 - last),
            subdiagonal.segment(last, i - last),
            eigenvalues.segment(last, i + 1 - last),
            eigenvectors.block(last, last, i + 1 - last, i + 1 - last));
      }
      last = i + 1;
    }
  }
  if (last == n - 1) {
    eigenvectors(last, last) = 1;
    eigenvalues[last] = diagonal[last];
  } else {
    mrrr<Scalar, Real>(diagonal.segment(last, n - last),
                       subdiagonal.segment(last, subdiagonal.size() - last),
                       eigenvalues.segment(last, n - last),
                       eigenvectors.block(last, last, n - last, n - last));
  }
  sort_eigenpairs(eigenvalues, eigenvectors);
}

}  // namespace internal
}  // namespace math
}  // namespace stan

#endif
