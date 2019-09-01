#include <stan/math/prim/scal.hpp>
#include <Eigen/Core>
#include <gtest/gtest.h>
#include <vector>
#include <type_traits>

// for testing that stan with just prim/scal does not know what fvar is.
template <typename T>
class fvar {};

// dummy var class
class var {};

// Note: Failure to find specialization is defined as "false"
template <typename T, typename = void>
struct require_not_tester : std::false_type {};

template <typename T>
struct require_not_tester<T, stan::require_not<T>> : std::true_type {};

TEST(template_enablers, require_not) {
  auto val = require_not_tester<std::false_type>::value;
  EXPECT_TRUE(val);
  val = require_not_tester<std::true_type>::value;
  EXPECT_FALSE(val);
}

template <typename T1, typename T2, typename = void>
struct require_all_not_tester : std::false_type {};

template <typename T1, typename T2>
struct require_all_not_tester<T1, T2, stan::require_all_not<T1, T2>>
    : std::true_type {};

TEST(template_enablers, require_all_not) {
  auto val = require_all_not_tester<std::false_type, std::false_type>::value;
  EXPECT_TRUE(val);
  val = require_all_not_tester<std::true_type, std::true_type>::value;
  EXPECT_FALSE(val);
  val = require_all_not_tester<std::true_type, std::false_type>::value;
  EXPECT_TRUE(val);
}

template <typename T1, typename T2, typename = void>
struct require_any_not_tester : std::false_type {};

template <typename T1, typename T2>
struct require_any_not_tester<T1, T2, stan::require_any_not<T1, T2>>
    : std::true_type {};

TEST(template_enablers, require_any_not) {
  auto val = require_any_not_tester<std::false_type, std::false_type>::value;
  EXPECT_TRUE(val);
  val = require_any_not_tester<std::true_type, std::true_type>::value;
  EXPECT_FALSE(val);
  val = require_any_not_tester<std::true_type, std::false_type>::value;
  EXPECT_FALSE(val);
}

template <typename T1, typename T2, typename = void>
struct require_all_tester : std::false_type {};

template <typename T1, typename T2>
struct require_all_tester<T1, T2, stan::require_all<T1, T2>>
    : std::true_type {};

TEST(template_enablers, require_all) {
  auto val = require_all_tester<std::false_type, std::false_type>::value;
  EXPECT_FALSE(val);
  val = require_all_tester<std::true_type, std::true_type>::value;
  EXPECT_TRUE(val);
  val = require_all_tester<std::true_type, std::false_type>::value;
  EXPECT_FALSE(val);
}

template <typename T1, typename T2, typename = void>
struct require_any_tester : std::false_type {};

template <typename T1, typename T2>
struct require_any_tester<T1, T2, stan::require_any<T1, T2>>
    : std::true_type {};

TEST(template_enablers, require_any) {
  auto val = require_any_tester<std::false_type, std::false_type>::value;
  EXPECT_FALSE(val);
  val = require_any_tester<std::true_type, std::true_type>::value;
  EXPECT_TRUE(val);
  val = require_any_tester<std::true_type, std::false_type>::value;
  EXPECT_TRUE(val);
}
