//
// Created by tadej on 14. 09. 2018.
//

#ifndef CLQR_HOUSEHOLDER_QR_HPP
#define CLQR_HOUSEHOLDER_QR_HPP

#include <stan/math/gpu/matrix_gpu.hpp>
#include <stan/math/gpu/identity.hpp>
#include <CL/cl.hpp>
#include <algorithm>
#include <iostream>
#include <vector>


namespace stan {
namespace math {

void p(const Eigen::MatrixXd a) {
    std::cout << a << std::endl;
}

void p(const Eigen::VectorXd a) {
    std::cout << a << std::endl;
}

void p(const matrix_gpu a) {
    Eigen::MatrixXd b(a.rows(), a.cols());
    copy(b, a);
    std::cout << b << std::endl;
}

void householder_qr(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R) {
    //matrix_gpu R_gpu(A);
    //matrix_gpu Q_gpu(Eigen::MatrixXd::Identity(A.rows()));
    R = A;
    Q = Eigen::MatrixXd::Identity(R.rows(), R.rows());
    for (size_t k = 0; k < std::min(R.rows() - 1, R.cols()); k++) {
        Eigen::VectorXd householder = R.block(k, k, R.rows() - k, 1);
        householder[0] -= copysign(householder.norm(), householder[0]);
        householder *= SQRT_2 / householder.norm();
        R.bottomRightCorner(R.rows() - k, R.cols() - k) -=
                householder * (R.bottomRightCorner(R.rows() - k, R.cols() - k).transpose() * householder).transpose();
        Q.rightCols(Q.cols() - k) -= (Q.rightCols(Q.cols() - k) * householder) * householder.transpose();
    }
}

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

void block_householder_qr_gpu(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R, int r) {
    matrix_gpu R_gpu(A);
    matrix_gpu Q_gpu(static_cast<const Eigen::MatrixXd>(Eigen::MatrixXd::Identity(A.rows(), A.rows())));
    //Eigen::ArrayXd mags = A.triangularView<Eigen::Lower>().colwise().norm(); //lahko ahead of time za cel R - ampak le od diagonale dol
    for (size_t k = 0; k < std::min(R_gpu.rows() - 1, R_gpu.cols()); k += r) {
        int actual_r = std::min({r, static_cast<int>(R_gpu.cols() - k), static_cast<int>(R_gpu.rows() - k)});
        matrix_gpu V(R_gpu.rows() - k, actual_r);
        //V.triangularView<Eigen::StrictlyUpper>() = Eigen::MatrixXd::Constant(V.rows(), V.cols(), 0);
        V.zeros<gpu::Upper>();

        //V.diagonal() -= (mags * V.diagonal().array().sign()).matrix();
        //now V contains first actual_r housholder vectors

        //V.array().colwise() *= SQRT_2 / V.colwise().norm().array().transpose();
        for (size_t j = 0; j < actual_r; j++) {
            //TODO do on GPU?
            matrix_gpu householder_gpu(R_gpu.rows() - k - j, 1);
            householder_gpu.sub_block(R_gpu, k + j, k + j, 0, 0, householder_gpu.rows(), householder_gpu.cols());
            Eigen::VectorXd householder(R_gpu.rows() - k - j, 1);
            copy(householder, householder_gpu);
            householder[0] -= copysign(householder.norm(), householder[0]);
            if (householder.rows() != 1) {
                householder *= SQRT_2 / householder.norm();
            }

            //V.col(j).tail(V.rows() - j) = householder;
            copy(householder_gpu, householder);
            V.sub_block(householder_gpu, 0, 0, j, j, V.rows() - j, 1);
            //R.block(k + j, k + j, A.rows() - k - j, actual_r - j) -= householder * (R.block(k + j, k + j, A.rows() - k - j, actual_r - j).transpose() * householder).transpose();
            matrix_gpu R_block(A.rows() - k - j, actual_r - j);
            R_block.sub_block(R_gpu, k + j, k + j, 0, 0, R_block.rows(), R_block.cols());
            R_gpu.sub_block(subtract(R_block, multiply(householder_gpu,
                                                       transpose(multiply(transpose(R_block), householder_gpu)))),
                            0, 0, k + j, k + j, R_block.rows(), R_block.cols());
        }

        matrix_gpu& Y = V;
        matrix_gpu W = V;
        for (size_t j = 1; j < actual_r; j++) {
            //W.col(j) = V.col(j) - W.leftCols(j) * (Y.leftCols(j).transpose() * V.col(j));
            matrix_gpu Y_left(Y.rows(), j);
            Y_left.sub_block(Y, 0, 0, 0, 0, Y.rows(), j);
            matrix_gpu W_left(W.rows(), j);
            W_left.sub_block(W, 0, 0, 0, 0, W.rows(), j);
            matrix_gpu householder_ex(V.rows(), 1);
            householder_ex.sub_block(V, 0, j, 0, 0, householder_ex.rows(), householder_ex.cols());
            W.sub_block(subtract(householder_ex, multiply(W_left, multiply(transpose(Y_left), householder_ex))),
                        0, 0, 0, j, W.rows(), 1);
        }
        //R.block(k, k + actual_r, R.rows() - k, R.cols() - k - actual_r) -=
        //        Y * (W.transpose() * R.block(k, k + actual_r, R.rows() - k, R.cols() - k - actual_r));
        matrix_gpu R_block(R_gpu.rows() - k, R_gpu.cols() - k - actual_r);
        R_block.sub_block(R_gpu, k, k + actual_r, 0, 0, R_block.rows(), R_block.cols());
        R_gpu.sub_block(subtract(R_block, multiply(Y, multiply(transpose(W), R_block))),
                        0, 0, k, k + actual_r, R_gpu.rows() - k, R_gpu.cols() - k - actual_r);


        //Q.rightCols(Q.cols() - k) -= (Q.rightCols(Q.cols() - k) * W) * Y.transpose();
        matrix_gpu Q_block(Q_gpu.rows(), Q_gpu.cols() - k);
        Q_block.sub_block(Q_gpu, 0, k, 0, 0, Q_block.rows(), Q_block.cols());
        Q_gpu.sub_block(subtract(Q_block, multiply(multiply(Q_block, W), transpose(Y))),
                        0, 0, 0, k, Q_block.rows(), Q_block.cols());
    }
    R = Eigen::MatrixXd(A.rows(), A.cols());
    Q = Eigen::MatrixXd(A.rows(), A.rows());
    copy(Q, Q_gpu);
    copy(R, R_gpu);
}

void block_householder_qr_gpu2(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R, int r) {
    matrix_gpu R_gpu(A);
    matrix_gpu Q_gpu(static_cast<const Eigen::MatrixXd>(Eigen::MatrixXd::Identity(A.rows(), A.rows())));
    //Eigen::ArrayXd mags = A.triangularView<Eigen::Lower>().colwise().norm(); //lahko ahead of time za cel R - ampak le od diagonale dol
    for (size_t k = 0; k < std::min(R_gpu.rows() - 1, R_gpu.cols()); k += r) {
        int actual_r = std::min({r, static_cast<int>(R_gpu.cols() - k), static_cast<int>(R_gpu.rows() - k)});
        Eigen::MatrixXd V(R_gpu.rows() - k, actual_r);
        V.triangularView<Eigen::StrictlyUpper>() = Eigen::MatrixXd::Constant(V.rows(), V.cols(), 0);

        //V.diagonal() -= (mags * V.diagonal().array().sign()).matrix();
        //now V contains first actual_r housholder vectors

        //V.array().colwise() *= SQRT_2 / V.colwise().norm().array().transpose();
        for (size_t j = 0; j < actual_r; j++) {
            matrix_gpu householder_gpu(R_gpu.rows() - k - j, 1);
            householder_gpu.sub_block(R_gpu, k + j, k + j, 0, 0, householder_gpu.rows(), householder_gpu.cols());
            Eigen::VectorXd householder(R_gpu.rows() - k - j, 1);
            copy(householder, householder_gpu);

            householder[0] -= copysign(householder.norm(), householder[0]);
            if (householder.rows() != 1) {
                householder *= SQRT_2 / householder.norm();
            }

            V.col(j).tail(V.rows() - j) = householder;

            matrix_gpu R_block_gpu(A.rows() - k - j, actual_r - j);
            R_block_gpu.sub_block(R_gpu, k + j, k + j, 0, 0, R_block_gpu.rows(), R_block_gpu.cols());
            Eigen::MatrixXd R_block(R_block_gpu.rows(), R_block_gpu.cols());
            copy(R_block_gpu, R_block);
            R_block -= householder * (R_block.transpose() * householder).transpose();
            copy(R_block, R_block_gpu);
            R_gpu.sub_block(R_block_gpu, 0, 0, k + j, k + j, R_block_gpu.rows(), R_block_gpu.cols());
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
        matrix_gpu R_block(R_gpu.rows() - k, R_gpu.cols() - k - actual_r);
        R_block.sub_block(R_gpu, k, k + actual_r, 0, 0, R_block.rows(), R_block.cols());
        R_gpu.sub_block(subtract(R_block, multiply(Y_gpu, multiply(transpose(W_gpu), R_block))),
                        0, 0, k, k + actual_r, R_gpu.rows() - k, R_gpu.cols() - k - actual_r);


        //Q.rightCols(Q.cols() - k) -= (Q.rightCols(Q.cols() - k) * W) * Y.transpose();
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

void block_householder_qr_gpu3(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R, int r) {
    matrix_gpu R_gpu(A);
    matrix_gpu Q_gpu(static_cast<const Eigen::MatrixXd>(Eigen::MatrixXd::Identity(A.rows(), A.rows())));
    //Eigen::ArrayXd mags = A.triangularView<Eigen::Lower>().colwise().norm(); //lahko ahead of time za cel R - ampak le od diagonale dol

    cl::Kernel kernel_1 = opencl_context.get_kernel("householder_QR_1");
    cl::CommandQueue& cmdQueue = opencl_context.queue();

    for (size_t k = 0; k < std::min(R_gpu.rows() - 1, R_gpu.cols()); k += r) {
        int actual_r = std::min({r, static_cast<int>(R_gpu.cols() - k), static_cast<int>(R_gpu.rows() - k)});
        Eigen::MatrixXd V(R_gpu.rows() - k, actual_r);
        V.triangularView<Eigen::StrictlyUpper>() = Eigen::MatrixXd::Constant(V.rows(), V.cols(), 0);

        //V.diagonal() -= (mags * V.diagonal().array().sign()).matrix();
        //now V contains first actual_r housholder vectors

        //V.array().colwise() *= SQRT_2 / V.colwise().norm().array().transpose();
        for (size_t j = 0; j < actual_r; j++) {
            matrix_gpu householder_gpu(R_gpu.rows() - k - j, 1);
            householder_gpu.sub_block(R_gpu, k + j, k + j, 0, 0, householder_gpu.rows(), householder_gpu.cols());
            Eigen::VectorXd householder(householder_gpu.rows());
            copy(householder, householder_gpu);
            householder[0] -= copysign(householder.norm(), householder[0]);
            if (householder.rows() != 1) {
                householder *= SQRT_2 / householder.norm();
            }

            V.col(j).tail(V.rows() - j) = householder;
            //R.block(k + j, k + j, A.rows() - k - j, actual_r - j) -=
            //        householder *
            //        (R.block(k + j, k + j, A.rows() - k - j, actual_r - j).transpose() * householder).transpose();
            copy(householder_gpu, householder);
            matrix_gpu tmp_gpu(A.rows() - k - j, actual_r - j);
            try {
                opencl_context.set_kernel_args(kernel_1, R_gpu.rows(), R_gpu.cols(),
                                               static_cast<int>(k + j), static_cast<int>(k + j),
                                               static_cast<int>(A.rows() - k - j), static_cast<int>(actual_r - j),
                                               R_gpu.buffer(), householder_gpu.buffer(), tmp_gpu.buffer());
                cmdQueue.enqueueNDRangeKernel(kernel_1, cl::NullRange,
                                              cl::NDRange(((actual_r+63)/64)*64),
                                              cl::NDRange(64), NULL, NULL);
            }
            catch (const cl::Error& e) {
                check_opencl_error("QR", e);
            }
            R_gpu.sub_block(tmp_gpu,0,0,k + j, k + j,tmp_gpu.rows(), tmp_gpu.cols());
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
        /*matrix_gpu R_block_gpu(static_cast<const Eigen::MatrixXd>(R.block(k, k + actual_r, R.rows() - k, R.cols() - k - actual_r)));
        R_block_gpu=subtract(R_block_gpu, multiply(Y_gpu, multiply(transpose(W_gpu), R_block_gpu)));
        Eigen::MatrixXd R_block(R_block_gpu.rows(),R_block_gpu.cols());
        copy(R_block,R_block_gpu);
        R.block(k, k + actual_r, R.rows() - k, R.cols() - k - actual_r) = R_block;*/

        matrix_gpu R_block(R_gpu.rows() - k, R_gpu.cols() - k - actual_r);
        R_block.sub_block(R_gpu, k, k + actual_r, 0, 0, R_block.rows(), R_block.cols());
        R_gpu.sub_block(subtract(R_block, multiply(Y_gpu, multiply(transpose(W_gpu), R_block))),
                        0, 0, k, k + actual_r, R_gpu.rows() - k, R_gpu.cols() - k - actual_r);


        //Q.rightCols(Q.cols() - k) -= (Q.rightCols(Q.cols() - k) * W) * Y.transpose();
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

void block_householder_qr_gpu4(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R, int r) {
    R = A;
    matrix_gpu Q_gpu(static_cast<const Eigen::MatrixXd>(Eigen::MatrixXd::Identity(A.rows(), A.rows())));
    //Eigen::ArrayXd mags = A.triangularView<Eigen::Lower>().colwise().norm(); //lahko ahead of time za cel R - ampak le od diagonale dol

    cl::Kernel kernel_1 = opencl_context.get_kernel("householder_QR_1");
    cl::CommandQueue& cmdQueue = opencl_context.queue();

    for (size_t k = 0; k < std::min(R.rows() - 1, R.cols()); k += r) {
        int actual_r = std::min({r, static_cast<int>(R.cols() - k), static_cast<int>(R.rows() - k)});
        Eigen::MatrixXd V(R.rows() - k, actual_r);
        V.triangularView<Eigen::StrictlyUpper>() = Eigen::MatrixXd::Constant(V.rows(), V.cols(), 0);

        //V.diagonal() -= (mags * V.diagonal().array().sign()).matrix();
        //now V contains first actual_r housholder vectors

        //V.array().colwise() *= SQRT_2 / V.colwise().norm().array().transpose();
        for (size_t j = 0; j < actual_r; j++) {
            Eigen::VectorXd householder = R.block(k + j, k + j, R.rows() - k - j, 1);
            householder[0] -= copysign(householder.norm(), householder[0]);
            if (householder.rows() != 1) {
                householder *= SQRT_2 / householder.norm();
            }

            V.col(j).tail(V.rows() - j) = householder;
            //R.block(k + j, k + j, A.rows() - k - j, actual_r - j) -=
            //        householder *
            //        (R.block(k + j, k + j, A.rows() - k - j, actual_r - j).transpose() * householder).transpose();
            //matrix_gpu R_block_gpu2(static_cast<const Eigen::MatrixXd>(R.block(k + j, k + j, A.rows() - k - j, actual_r - j)));
            matrix_gpu householder_gpu(householder);
            matrix_gpu R_block_gpu(A.rows() - k - j, actual_r - j);
            cl::size_t<3> buffer_offset;
            buffer_offset[0]=0;
            buffer_offset[1]=0;
            buffer_offset[2]=0;
            cl::size_t<3> host_offset;
            host_offset[0]=(k+j)* sizeof(double);
            host_offset[1]=(k+j);
            host_offset[2]=0;
            cl::size_t<3> region;
            region[0]=(A.rows() - k - j)* sizeof(double);
            region[1]=(actual_r - j)/** sizeof(double)*/;
            region[2]=1;

            region[0]=R_block_gpu.rows()* sizeof(double);
            region[1]=R_block_gpu.cols()/** sizeof(double)*/;


            /*host_offset[0]=0;
            host_offset[1]=0;
            host_offset[2]=0;
            region[0]=sizeof(double);
            region[1]=1;
            region[2]=1;*/

            matrix_gpu tmp_gpu(R_block_gpu.rows(), R_block_gpu.cols());
            try {
                cmdQueue.enqueueWriteBufferRect(R_block_gpu.buffer(),true, buffer_offset, host_offset, region,
                                                R_block_gpu.rows()*sizeof(double),0, //buffer pitch
                                                R.rows()*sizeof(double), 0, //host pitch
                                                R.data());
                opencl_context.set_kernel_args(kernel_1, R_block_gpu.rows(), R_block_gpu.cols(),
                                               0, 0,
                                               R_block_gpu.rows(), R_block_gpu.cols(),householder_gpu.rows(),
                                               R_block_gpu.buffer(), householder_gpu.buffer(), tmp_gpu.buffer());
                cmdQueue.enqueueNDRangeKernel(kernel_1, cl::NullRange,
                                              cl::NDRange(((actual_r+63)/64)*64),
                                              cl::NDRange(64), NULL, NULL);

                cmdQueue.enqueueReadBufferRect(tmp_gpu.buffer(),true, buffer_offset, host_offset, region,
                                                R_block_gpu.rows()*sizeof(double),0, //buffer pitch
                                                R.rows()*sizeof(double), 0, //host pitch
                                                R.data());
                /*matrix_gpu R_gpu(R);
                R_gpu.sub_block(tmp_gpu,0,0,k+j,k+j,tmp_gpu.rows(),tmp_gpu.cols());
                copy(R,R_gpu);*/
            }
            catch (const cl::Error& e) {
                check_opencl_error("QR", e);
            }
            /*Eigen::MatrixXd R_block(R_block_gpu.rows(),R_block_gpu.cols());
            copy(R_block,tmp_gpu);
            R.block(k + j, k + j, A.rows() - k - j, actual_r - j) = R_block;*/
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
