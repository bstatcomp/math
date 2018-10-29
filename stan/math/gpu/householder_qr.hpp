//
// Created by tadej on 14. 09. 2018.
//

#ifndef CLQR_HOUSEHOLDER_QR_HPP
#define CLQR_HOUSEHOLDER_QR_HPP

#include <stan/math/gpu/matrix_gpu.hpp>
#include <stan/math/gpu/identity.hpp>
#include <CL/cl.hpp>
#include <algorithm>
#include <vector>


namespace stan {
namespace math {

/**
 * Calculates QR decomposition of A using the block Householder algorithm on a CPU.
 * @param A matrix to factorize
 * @param Q out the orthonormal matrix
 * @param R out the lower triangular matrix
 * @param r Block size. Optimal value depends on the hardware.
 */
void block_householder_qr(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R, int r = 60) {
    R = A;
    Q = Eigen::MatrixXd::Identity(R.rows(), R.rows());
    //Eigen::ArrayXd mags = A.triangularView<Eigen::Lower>().colwise().norm(); //householder norms could be calculated ahead of time (only below diadonal elements)
    for (size_t k = 0; k < std::min(R.rows() - 1, R.cols()); k += r) {
        int actual_r = std::min({r, static_cast<int>(R.cols() - k), static_cast<int>(R.rows() - k)});
        Eigen::MatrixXd V(R.rows() - k, actual_r);
        V.triangularView<Eigen::StrictlyUpper>() = Eigen::MatrixXd::Constant(V.rows(), V.cols(), 0);

        for (size_t j = 0; j < actual_r; j++) {
            Eigen::VectorXd householder = R.block(k + j, k + j, R.rows() - k - j, 1);
            householder[0] -= copysign(householder.norm(), householder[0]);
            if (householder.rows() != 1) {
                householder *= SQRT_2 / householder.norm();
            }

            V.col(j).tail(V.rows() - j) = householder;
            R.block(k + j, k + j, A.rows() - k - j, actual_r - j) -=
                    householder *
                    (R.block(k + j, k + j, A.rows() - k - j, actual_r - j).transpose() * householder).transpose();
        }

        Eigen::MatrixXd& Y = V;
        Eigen::MatrixXd W = V;
        for (size_t j = 1; j < actual_r; j++) {
            W.col(j) = V.col(j) - W.leftCols(j) * (Y.leftCols(j).transpose() * V.col(j));
        }
        R.block(k, k + actual_r, R.rows() - k, R.cols() - k - actual_r) -=
                Y * (W.transpose() * R.block(k, k + actual_r, R.rows() - k, R.cols() - k - actual_r));

        Q.rightCols(Q.cols() - k) -= (Q.rightCols(Q.cols() - k) * W) * Y.transpose();
    }
}

/**
 * Calculates QR decomposition of A using the block Householder algorithm, which divides work between CPU and GPU.
 * This is slower than full GPU implementation (`block_householder_qr_gpu`) for large matrices (> ~8000*8000) and
 * faster for smaller matrices.
 * @param A matrix to factorize
 * @param Q out the orthonormal matrix
 * @param R out the lower triangular matrix
 * @param r Block size. Optimal value depends on the hardware.
 */
void block_householder_qr_gpu_hybrid(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R, int r = 120) {
    R = A;
    matrix_gpu Q_gpu(static_cast<const Eigen::MatrixXd>(Eigen::MatrixXd::Identity(A.rows(), A.rows())));

    cl::Kernel kernel_1 = opencl_context.get_kernel("householder_QR_1");
    cl::CommandQueue& cmdQueue = opencl_context.queue();

    for (size_t k = 0; k < std::min(R.rows() - 1, R.cols()); k += r) {
        int actual_r = std::min({r, static_cast<int>(R.cols() - k), static_cast<int>(R.rows() - k)});
        Eigen::MatrixXd V(R.rows() - k, actual_r);
        V.triangularView<Eigen::StrictlyUpper>() = Eigen::MatrixXd::Constant(V.rows(), V.cols(), 0);

        for (size_t j = 0; j < actual_r; j++) {
            Eigen::VectorXd householder = R.block(k + j, k + j, R.rows() - k - j, 1);
            householder[0] -= copysign(householder.norm(), householder[0]);
            if (householder.rows() != 1) {
                householder *= SQRT_2 / householder.norm();
            }

            V.col(j).tail(V.rows() - j) = householder;
            R.block(k + j, k + j, A.rows() - k - j, actual_r - j) -=
                    householder *
                    (R.block(k + j, k + j, A.rows() - k - j, actual_r - j).transpose() * householder).transpose();
        }

        Eigen::MatrixXd& Y = V;
        Eigen::MatrixXd W = V;
        for (size_t j = 1; j < actual_r; j++) {
            W.col(j) = V.col(j) - W.leftCols(j) * (Y.leftCols(j).transpose() * V.col(j));
        }
        matrix_gpu Y_gpu(Y);
        matrix_gpu W_gpu(W);

        //R.block(k, k + actual_r, R.rows() - k, R.cols() - k - actual_r) -=
        //        Y * (W.transpose() * R.block(k, k + actual_r, R.rows() - k, R.cols() - k - actual_r));
        matrix_gpu R_block_gpu(static_cast<const Eigen::MatrixXd>(R.block(k, k + actual_r, R.rows() - k, R.cols() - k - actual_r)));
        R_block_gpu=subtract(R_block_gpu, multiply(Y_gpu, multiply(transpose(W_gpu), R_block_gpu)));
        Eigen::MatrixXd R_block(R_block_gpu.rows(),R_block_gpu.cols());
        copy(R_block,R_block_gpu);
        R.block(k, k + actual_r, R.rows() - k, R.cols() - k - actual_r) = R_block;


        //Q.rightCols(Q.cols() - k) -= (Q.rightCols(Q.cols() - k) * W) * Y.transpose();
        matrix_gpu Q_block(Q_gpu.rows(), Q_gpu.cols() - k);
        Q_block.sub_block(Q_gpu, 0, k, 0, 0, Q_block.rows(), Q_block.cols());
        Q_gpu.sub_block(subtract(Q_block, multiply(multiply(Q_block, W_gpu), transpose(Y_gpu))),
                        0, 0, 0, k, Q_block.rows(), Q_block.cols());
    }
    Q = Eigen::MatrixXd(A.rows(), A.rows());
    copy(Q, Q_gpu);
}

/**
 * Calculates QR decomposition of A using the block Householder algorithm on a GPU.
 * This is faster than hybrid CPU/GPU (`block_householder_qr_gpu_hybrid`) implementation
 * for large matrices (> ~8000*8000) and slower for smaller matrices.
 * @param A matrix to factorize
 * @param Q out the orthonormal matrix
 * @param R out the lower triangular matrix
 * @param r Block size. Optimal value depends on the hardware.
 */
void block_householder_qr_gpu(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R, int r = 160) {
    matrix_gpu R_gpu(A);
    matrix_gpu Q_gpu=identity(A.rows());

    cl::Kernel kernel_1 = opencl_context.get_kernel("householder_QR_1");
    cl::Kernel kernel_2 = opencl_context.get_kernel("householder_QR_2");
    cl::Kernel kernel_3 = opencl_context.get_kernel("householder_QR_3");
    cl::Kernel kernel_4 = opencl_context.get_kernel("householder_QR_4");
    cl::CommandQueue& cmdQueue = opencl_context.queue();

    for (size_t k = 0; k < std::min(R_gpu.rows() - 1, R_gpu.cols()); k += r) {
        int actual_r = std::min({r, static_cast<int>(R_gpu.cols() - k), static_cast<int>(R_gpu.rows() - k)});
        matrix_gpu V_gpu(R_gpu.rows() - k, actual_r);
        V_gpu.zeros<gpu::Upper>();

        for (size_t j = 0; j < actual_r; j++) {
            matrix_gpu tmp_gpu(A.rows() - k - j, actual_r - j);
            try{
                opencl_context.set_kernel_args(kernel_2, R_gpu.rows(), R_gpu.cols(), (int)(k + j),(int)j,
                                               R_gpu.buffer(), V_gpu.buffer());
                cmdQueue.enqueueNDRangeKernel(kernel_2, cl::NullRange,
                                              cl::NDRange(128),
                                              cl::NDRange(128), NULL, NULL);
                opencl_context.set_kernel_args(kernel_1, R_gpu.rows(), R_gpu.cols(),
                                               (int)(k+j), (int)(k+j),
                                               tmp_gpu.rows(), tmp_gpu.cols(),V_gpu.rows(),
                                               R_gpu.buffer(), V_gpu.buffer(), tmp_gpu.buffer());
                cmdQueue.enqueueNDRangeKernel(kernel_1, cl::NullRange,
                                              cl::NDRange(((actual_r+63)/64)*64),
                                              cl::NDRange(64), NULL, NULL);
            }
            catch (const cl::Error& e) {
                std::cout << "err1";
                check_opencl_error("QR", e);
            }

            R_gpu.sub_block(tmp_gpu,0,0,k+j,k+j,tmp_gpu.rows(),tmp_gpu.cols());
        }
        matrix_gpu& Y_gpu = V_gpu;
        matrix_gpu W_gpu = V_gpu;
        for (size_t j = 1; j < actual_r; j++) {
            matrix_gpu tmp_gpu(j,1);
            try{
                opencl_context.set_kernel_args(kernel_3, Y_gpu.rows(), Y_gpu.cols(), (int)j,
                                               Y_gpu.buffer(), V_gpu.buffer(), tmp_gpu.buffer());
                cmdQueue.enqueueNDRangeKernel(kernel_3, cl::NullRange,
                                              cl::NDRange(((j+63)/64)*64),
                                              cl::NDRange(64), NULL, NULL);
                opencl_context.set_kernel_args(kernel_4, W_gpu.rows(), W_gpu.cols(),(int)j,
                                                       W_gpu.buffer(), tmp_gpu.buffer(), V_gpu.buffer());
                cmdQueue.enqueueNDRangeKernel(kernel_4, cl::NullRange,
                                              cl::NDRange(((W_gpu.rows()+63)/64)*64),
                                              cl::NDRange(64), NULL, NULL);
            }
            catch (const cl::Error& e) {
                std::cout << "err2";
                check_opencl_error("QR", e);
            }
        }
        matrix_gpu R_block(R_gpu.rows() - k, R_gpu.cols() - k - actual_r);
        R_block.sub_block(R_gpu, k, k + actual_r, 0, 0, R_block.rows(), R_block.cols());
        R_gpu.sub_block(subtract(R_block, multiply(Y_gpu, multiply(transpose(W_gpu), R_block))),
                        0, 0, k, k + actual_r, R_gpu.rows() - k, R_gpu.cols() - k - actual_r);

        matrix_gpu Q_block(Q_gpu.rows(), Q_gpu.cols() - k);
        Q_block.sub_block(Q_gpu, 0, k, 0, 0, Q_block.rows(), Q_block.cols());
        Q_gpu.sub_block(subtract(Q_block, multiply(multiply(Q_block, W_gpu), transpose(Y_gpu))),
                        0, 0, 0, k, Q_block.rows(), Q_block.cols());
    }
    R = Eigen::MatrixXd(A.rows(), A.cols());
    Q = Eigen::MatrixXd(A.rows(), A.rows());
    copy(Q, Q_gpu);
    copy(R, R_gpu);
}

/**
 * Returns the upper triangular factor of the fat QR decomposition. Does not use GPU implementation, as it is not autodiff-able.
 * @param m Matrix.
 * @tparam T scalar type
 * @return Upper triangular matrix with maximal rows
 */
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> qr_R_gpu(
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m) {
    return qr_R(m);
}

/**
 * Returns the upper triangular factor of the fat QR decomposition. GPU implementation is used only for double types.
 * @param m Matrix.
 * @tparam T scalar type
 * @return Upper triangular matrix with maximal rows
 */
template <>
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> qr_R_gpu(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& m) {
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
    check_nonzero_size("qr_R", "m", m);

    matrix_t Q, R;
    block_householder_qr_gpu(m, Q, R);

    if (m.rows() > m.cols())
        R.bottomRows(m.rows() - m.cols()).setZero();
    const int min_size = std::min(m.rows(), m.cols());
    for (int i = 0; i < min_size; i++) {
        for (int j = 0; j < i; j++)
            R.coeffRef(i, j) = 0.0;
        if (R(i, i) < 0)
            R.row(i) *= -1.0;
    }
    return R;
}

/**
 * Returns the orthogonal factor of the fat QR decomposition. Does not use GPU implementation, as it is not autodiff-able.
 * @param m Matrix.
 * @tparam T scalar type
 * @return Orthogonal matrix with maximal columns
 */
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> qr_Q_gpu(
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m) {
    return qr_Q(m);
}

/**
 * Returns the orthogonal factor of the fat QR decomposition. GPU implementation is used only for double types.
 * @param m Matrix.
 * @tparam T scalar type
 * @return Orthogonal matrix with maximal columns
 */
template <>
Eigen::Matrix<double , Eigen::Dynamic, Eigen::Dynamic> qr_Q_gpu(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& m) {
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
    check_nonzero_size("qr_Q", "m", m);

    matrix_t Q, R;
    block_householder_qr_gpu(m, Q, R);

    const int min_size = std::min(m.rows(), m.cols());
    for (int i = 0; i < min_size; i++)
        if (R(i, i) < 0)
            Q.col(i) *= -1.0;
    return Q;
}

}
}

#endif //CLQR_HOUSEHOLDER_QR_HPP
