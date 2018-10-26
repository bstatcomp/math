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

void block_householder_qr(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R, int r) {
    R = A;
    Q = Eigen::MatrixXd::Identity(R.rows(), R.rows());
    //Eigen::ArrayXd mags = A.triangularView<Eigen::Lower>().colwise().norm(); //lahko ahead of time za cel R - ampak le od diagonale dol
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

void block_householder_qr_gpu5(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R, int r) {
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

void block_householder_qr_gpu6(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R, int r) {
    matrix_gpu R_gpu(A);
    matrix_gpu Q_gpu=identity(A.rows());
    //matrix_gpu Q_gpu(static_cast<const Eigen::MatrixXd>(Eigen::MatrixXd::Identity(A.rows(), A.rows())));

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
                cmdQueue.finish();
                opencl_context.set_kernel_args(kernel_1, R_gpu.rows(), R_gpu.cols(),
                                               (int)(k+j), (int)(k+j),
                                               tmp_gpu.rows(), tmp_gpu.cols(),V_gpu.rows(),
                                               R_gpu.buffer(), V_gpu.buffer(), tmp_gpu.buffer());
                cmdQueue.enqueueNDRangeKernel(kernel_1, cl::NullRange,
                                              cl::NDRange(((actual_r+63)/64)*64),
                                              cl::NDRange(64), NULL, NULL);
                cmdQueue.finish();
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
                cmdQueue.finish();
                opencl_context.set_kernel_args(kernel_4, W_gpu.rows(), W_gpu.cols(),(int)j,
                                                       W_gpu.buffer(), tmp_gpu.buffer(), V_gpu.buffer());
                cmdQueue.enqueueNDRangeKernel(kernel_4, cl::NullRange,
                                              cl::NDRange(((W_gpu.rows()+63)/64)*64),
                                              cl::NDRange(64), NULL, NULL);
                cmdQueue.finish();
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

}
}

#endif //CLQR_HOUSEHOLDER_QR_HPP
