#ifdef STAN_OPENCL
#include <stan/math/prim/mat.hpp>
#include <stan/math/gpu/householder_qr.hpp>
#include <gtest/gtest.h>
#include <algorithm>



TEST(MathMatrixGPU, householder_m_exception_pass) {
    stan::math::matrix_d m(1, 1), q(1, 1), r(1, 1);

    EXPECT_NO_THROW(q = stan::math::qr_Q_gpu(m));
    EXPECT_NO_THROW(r = stan::math::qr_R_gpu(m));
}

TEST(MathMatrixGPU, householder_m_value_check) {
    double eps = 1e-10;
    stan::math::matrix_d m0(3, 4);
    m0 << 2, 2, 2, 2,
        3, 4, 5, 6,
        -1, -2, 0,3;

    stan::math::matrix_d q0, r0, q1, r1;

    EXPECT_NO_THROW(q0 = stan::math::qr_Q_gpu(m0));
    EXPECT_NO_THROW(r0 = stan::math::qr_R_gpu(m0));

    q1=stan::math::qr_Q(m0);
    r1=stan::math::qr_R(m0);

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 4; j++){
            EXPECT_NEAR(r0(i, j), r1(i, j), eps);
        }
        for(int j = 0; j < 3; j++){
            EXPECT_NEAR(q0(i, j), q1(i, j), eps);
        }
    }
}

#endif
