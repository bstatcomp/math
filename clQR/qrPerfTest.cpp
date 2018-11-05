#include <iostream>

#define OPENCL_PLATFORM_ID 0
#define OPENCL_DEVICE_ID 0

#include <stan/math.hpp>
#include <stan/math/gpu/add.hpp>
#include <chrono>
#include <Eigen/QR>
#include <stan/math/gpu/householder_qr.hpp>


using namespace Eigen;
using namespace std;
using namespace stan::math;

using Mat=Matrix<double, Dynamic, Dynamic>;
using Vec=VectorXd;

void p(const Eigen::MatrixXd& a) {
    std::cout << a << std::endl;
}

void p(const Eigen::VectorXd& a) {
    std::cout << a << std::endl;
}

void p(const matrix_gpu& a) {
    Eigen::MatrixXd b(a.rows(), a.cols());
    copy(b, a);
    std::cout << b << std::endl;
}

void chk(const Mat& a, const Mat& Q, const Mat& R){
    Mat R2 = R;
    R2.triangularView<Eigen::StrictlyLower>() = Eigen::MatrixXd::Constant(R.rows(), R.cols(), 0);
    cout << "R triang: " << (R - R2).array().abs().sum() << endl;
    cout << "reconstruct: " << (Q * R - a).array().abs().sum() << endl;
    cout << "ID: " << (Q.transpose() * Q).array().abs().sum() - Q.rows() << endl;
}

int test() {
    /*Mat a(3,3);
    a << 12,-51,4,
    6,167,-68,
    -4,24,-41;*/
    Mat a(6, 4);
    a << 12, -51, 4, 4, 5, 6,
            6, 167, -68, 7, 8, 9,
            -4, 24, -41, 7, 9, 6,
            1, 2, 3, 4, 5, 6;
    cout << "a:" << endl << a << endl;

    cout << "Eigen impl:" << endl;
    HouseholderQR<Mat> qr(a);
    Mat R = qr.matrixQR().triangularView<Eigen::Upper>();
    Mat Q = qr.householderQ();
    Mat R2 = R;
    chk(a,Q,R);

    for (int r = 1; r <= 3; r++) {
        cout << "##################################################################################  block impl, r=" << r << endl;
        block_householder_qr_gpu(a, Q, R, r);
        chk(a,Q,R);
    }
}

void test_my_mul(){
    Mat a(7,6);
    a << 12, -51,   4,   1,  2,  3,
         6,  167, -68,   5,  5,  5,
        -4,   24, -41,  -6, -9,  1,
         1,    3,   7,   2,  4,  6,
        -3,   -4,  -3,  -3, -3,  1,
         6,  167, -68,   5,  5,  5,
        -4,   24, -41,  -6, -9,  1;
    Vec h(4);
    h << 1,2,-3,3;
    Mat V(7,6);
    V << 0,0,0,0,2,0,
         0,1,0,0,3,0,
         0,2,0,0,4,0,
         0,-3,0,0,5,0,
         0,-3,0,0,6,0,
         0,-3,0,0,9,0,
         0,3,0,0,-9,0;
    int cols=4;
    matrix_gpu a_gpu(a);
    matrix_gpu h_gpu(h);
    matrix_gpu V_gpu(V);
    matrix_gpu res_gpu(cols,1);
    //cout << "block" <<endl;
    //cout << a.block(1,2,4,4) << endl;
    Mat cpu = a.leftCols(cols).transpose() * a.col(cols);
    a.col(cols) = V.col(cols) - a.leftCols(cols) * cpu;
    cl::Kernel kernel = opencl_context.get_kernel("householder_QR_3");
    cl::Kernel kernel4 = opencl_context.get_kernel("householder_QR_4");
    cl::CommandQueue& cmdQueue = opencl_context.queue();
    try {
        opencl_context.set_kernel_args(kernel, (int) a.rows(), (int) a.cols(), cols,
                                       a_gpu.buffer(), a_gpu.buffer(), res_gpu.buffer());
        cmdQueue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange(64),
                                      cl::NDRange(64), NULL, NULL);

        cout << "a:" <<endl;
        p(a_gpu);
        opencl_context.set_kernel_args(kernel4, (int) a.rows(), (int) a.cols(), cols,
                                       a_gpu.buffer(), res_gpu.buffer(), V_gpu.buffer());
        cmdQueue.enqueueNDRangeKernel(kernel4, cl::NullRange,
                                      cl::NDRange(64),
                                      cl::NDRange(64), NULL, NULL);
    }
    catch (const cl::Error& e) {
        check_opencl_error("QR", e);
    }

    cout << "cpu res" <<endl;
    cout << cpu << endl;
    cout << "GPU res" <<endl;
    p(res_gpu);


    cout << "cpu res2" <<endl;
    cout << a << endl;
    cout << "GPU res2" <<endl;
    p(a_gpu);

}


int main() {
    //test();
    //test_my_mul();
    //return 0;

    int A = 1001;
    int B = 1001;
    const int MAX_BLOCK=400;
    Mat a = Mat::Random(A, B);
    Mat R,Q,R2;
    //block_householder_qr(a, Q, R, 80);
    //cout << "reconstruct: " << (Q * R - a).array().abs().sum() << endl;
    //return 0;

    auto start = std::chrono::steady_clock::now();
    /*HouseholderQR<Mat> qr(a);
    cout << "CPU: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()
         << "ms" << endl;
    R = qr.matrixQR().triangularView<Upper>();
    Q = qr.householderQ();
    cout << "CPU total: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()
         << "ms" << endl;
    cout << "(" << R.rows() << " " << R.cols() << ") (" << Q.rows() << " " << Q.cols() << ")" << endl;
    R2 = R;
    R2.triangularView<Eigen::StrictlyLower>() = Eigen::MatrixXd::Constant(R.rows(), R.cols(), 0);
    cout << "R triang: " << (R - R2).array().abs().sum() << endl;
    cout << "reconstruct: " << (Q * R - a).array().abs().sum() << endl;
    cout << "ID: " << (Q.transpose() * Q).array().abs().sum() - Q.rows() << endl;
*/
    /*start = std::chrono::steady_clock::now();
    householder_qr(a, Q, R);
    cout << "CPU my: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()
         << "ms" << endl;
    R2 = R;
    R2.triangularView<Eigen::StrictlyLower>() = Eigen::MatrixXd::Constant(R.rows(), R.cols(), 0);
    cout << "R triang: " << (R - R2).array().abs().sum() << endl;
    cout << "reconstruct: " << (Q * R - a).array().abs().sum() << endl;
    cout << "ID: " << (Q.transpose() * Q).array().abs().sum() - Q.rows() << endl;
    */

    //force kernel compilation
    cl::Kernel kernel_1 = opencl_context.get_kernel("householder_QR_1");
    cl::Kernel kernel_2 = opencl_context.get_kernel("matrix_multiply");

    start = std::chrono::steady_clock::now();
    block_householder_qr_gpu_hybrid(a, Q, R, 120);
    cout << "GPU - hybrid my block 120: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()
         << "ms" << endl;
    chk(a,Q,R);

    block_householder_qr_gpu(a, Q, R, 160);
    start = std::chrono::steady_clock::now();
    block_householder_qr_gpu(a, Q, R, 160);
    cout << "GPU my block 160: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()
         << "ms" << endl;
    chk(a,Q,R);
/*
    start = std::chrono::steady_clock::now();
    block_householder_qr(a, Q, R, 60);
    cout << "CPU my block 60: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()
         << "ms" << endl;
    chk(a,Q,R);
*/
    /*cout << "##################### CPU" << endl;
    for (int r = 20; r < std::min({A,B,MAX_BLOCK}); r += 5) {
        start = std::chrono::steady_clock::now();
        block_householder_qr(a, Q, R, r);
        cout << "block " << r << ": "
             << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()
             << "ms" << endl;
        chk(a,Q,R);
    }
    cout << "##################### GPU 1" << endl;
    for (int r = 20; r < std::min({A,B,MAX_BLOCK}); r += 5) {
        start = std::chrono::steady_clock::now();
        block_householder_qr_gpu(a, Q, R, r);
        cout << "block " << r << ": "
             << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()
             << "ms" << endl;
        chk(a,Q,R);
    }
    cout << "##################### GPU 2" << endl;
    for (int r = 20; r < std::min({A,B,MAX_BLOCK}); r += 5) {
        start = std::chrono::steady_clock::now();
        block_householder_qr_gpu2(a, Q, R, r);
        cout << "block " << r << ": "
             << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()
             << "ms" << endl;
        chk(a,Q,R);
    }
    cout << "##################### GPU 3" << endl;
    for (int r = 20; r < std::min({A,B,MAX_BLOCK}); r += 5) {
        start = std::chrono::steady_clock::now();
        block_householder_qr_gpu3(a, Q, R, r);
        cout << "block " << r << ": "
             << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()
             << "ms" << endl;
        chk(a,Q,R);
    }
    cout << "##################### GPU 4" << endl;
    for (int r = 20; r < std::min({A,B,MAX_BLOCK}); r += 5) {
        start = std::chrono::steady_clock::now();
        block_householder_qr_gpu4(a, Q, R, r);
        cout << "block " << r << ": "
             << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()
             << "ms" << endl;
        chk(a,Q,R);
    }*/

    /*cout << "##################### GPU 5" << endl;
    for (int r = 20; r < std::min({A,B,MAX_BLOCK}); r += 5) {
        start = std::chrono::steady_clock::now();
        block_householder_qr_gpu5(a, Q, R, r);
        cout << "block " << r << ": "
             << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()
             << "ms" << endl;
        chk(a,Q,R);
    }


    cout << "##################### GPU 6" << endl;
    for (int r = 20; r < std::min({A,B,MAX_BLOCK}); r += 5) {
        start = std::chrono::steady_clock::now();
        block_householder_qr_gpu6(a, Q, R, r);
        cout << "block " << r << ": "
             << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()
             << "ms" << endl;
        chk(a,Q,R);
    }*/

    return 0;
}