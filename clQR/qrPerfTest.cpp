#include <iostream>

#define OPENCL_PLATFORM_ID 1
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

void chk(Mat a, Mat Q, Mat R){
    /*Mat R2 = R;
    R2.triangularView<Eigen::StrictlyLower>() = Eigen::MatrixXd::Constant(R.rows(), R.cols(), 0);
    cout << "R triang: " << (R - R2).array().abs().sum() << endl;
    cout << "reconstruct: " << (Q * R - a).array().abs().sum() << endl;
    cout << "ID: " << (Q.transpose() * Q).array().abs().sum() - Q.rows() << endl;*/
}

int test() {
    /*Mat a(3,3);
    a << 12,-51,4,
    6,167,-68,
    -4,24,-41;*/
    Mat a(4, 6);
    a << 12, -51, 4, 4, 5, 6,
            6, 167, -68, 7, 8, 9,
            -4, 24, -41, 7, 9, 6,
            1, 2, 3, 4, 5, 6;
    cout << "a:" << endl << a << endl;

    cout << "Eigen impl:" << endl;
    HouseholderQR<Mat> qr(a);
    Mat R = qr.matrixQR();
    Mat Q = qr.householderQ();
    Mat R2 = R;
    R2.triangularView<Eigen::StrictlyLower>() = Eigen::MatrixXd::Constant(R.rows(), R.cols(), 0);
    cout << "R triang: " << (R - R2).array().abs().sum() << endl;
    cout << "reconstruct: " << (Q * R - a).array().abs().sum() << endl;
    cout << "ID: " << (Q.transpose() * Q).array().abs().sum() - Q.rows() << endl;
    /*cout << "Q1:" << endl << Q << endl;
    cout << "R1:" << endl << R << endl;
    cout << "reconstruct1:" << endl << Q * R << endl;
    cout << "ID1:" << endl << Q.transpose() * Q << endl;*/


    cout << "my naive impl:" << endl;
    householder_qr(a, Q, R);
    /*cout << "Q2:" << endl << Q << endl;
    cout << "R2:" << endl << R << endl;
    cout << "reconstruct2:" << endl << Q * R << endl;
    cout << "ID2:" << endl << Q.transpose() * Q << endl;*/
    R2 = R;
    R2.triangularView<Eigen::StrictlyLower>() = Eigen::MatrixXd::Constant(R.rows(), R.cols(), 0);
    cout << "R triang: " << (R - R2).array().abs().sum() << endl;
    cout << "reconstruct: " << (Q * R - a).array().abs().sum() << endl;
    cout << "ID: " << (Q.transpose() * Q).array().abs().sum() - Q.rows() << endl;

    for (int r = 1; r <= 3; r++) {
        cout << "##################################################################################  block impl, r=" << r << endl;
        block_householder_qr_gpu3(a, Q, R, r);
        /*cout << "Q3:" << endl << Q << endl;
        cout << "R3:" << endl << R << endl;
        cout << "reconstruct3:" << endl << Q * R << endl;
        cout << "ID3:" << endl << Q.transpose() * Q << endl;*/
        R2 = R;
        R2.triangularView<Eigen::StrictlyLower>() = Eigen::MatrixXd::Constant(R.rows(), R.cols(), 0);
        cout << "R triang: " << (R - R2).array().abs().sum() << endl;
        cout << "reconstruct: " << (Q * R - a).array().abs().sum() << endl;
        cout << "ID: " << (Q.transpose() * Q).array().abs().sum() - Q.rows() << endl;
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
    Mat V(5,5);
    V << 0,0,0,0,0,
        0,1,0,0,0,
        0,2,0,0,0,
        0,-3,0,0,0,
        0,3,0,0,0;
    cout << "block" <<endl;
    cout << a.block(1,2,4,4) << endl;
    cout << "tmp vec" <<endl;
    Mat tmp =(a.block(1,2,4,4).transpose()*h);
    cout << tmp << endl;
    Mat cpu = a.block(1,2,4,4)-h*tmp.transpose();
    cout << "cpu res" <<endl;
    cout << cpu << endl;
    cl::Kernel kernel = opencl_context.get_kernel("householder_QR_1");
    cl::CommandQueue& cmdQueue = opencl_context.queue();
    matrix_gpu a_gpu(a);
    matrix_gpu h_gpu(h);
    matrix_gpu V_gpu(V);
    matrix_gpu res_gpu(4,4);
    try {
        opencl_context.set_kernel_args(kernel, (int) a.rows(), (int) a.cols(), 1, 2, 4, 4, (int)V.rows(),
                a_gpu.buffer(), V_gpu.buffer(), res_gpu.buffer());
        cmdQueue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange(6),
                                      cl::NDRange(3), NULL, NULL);
    }
    catch (const cl::Error& e) {
        check_opencl_error("QR", e);
    }
    Mat res(res_gpu.rows(),res_gpu.cols());
    copy(res,res_gpu);
    cout << "cpu res" <<endl;
    cout << cpu << endl;
    cout << "GPU res" <<endl;
    cout << res << endl;
    cout << (res-cpu).array().abs().sum() << endl;

}


int main() {
    //test();
    //test_my_mul();
    //return 0;

    int A = 1100;
    int B = 1100;
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

    /*start = std::chrono::steady_clock::now();
    block_householder_qr(a, Q, R, 60);
    cout << "CPU my block 60: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()
         << "ms" << endl;
    R2 = R;
    R2.triangularView<Eigen::StrictlyLower>() = Eigen::MatrixXd::Constant(R.rows(), R.cols(), 0);
    cout << "R triang: " << (R - R2).array().abs().sum() << endl;
    cout << "reconstruct: " << (Q * R - a).array().abs().sum() << endl;
    cout << "ID: " << (Q.transpose() * Q).array().abs().sum() - Q.rows() << endl;


    start = std::chrono::steady_clock::now();
    block_householder_qr_gpu3(a, Q, R, 120);
    cout << "GPU my block 120 (+compile): "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()
         << "ms" << endl;
    */
    start = std::chrono::steady_clock::now();
    block_householder_qr_gpu4(a, Q, R, 90);
    cout << "GPU my block 90: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()
         << "ms" << endl;
    chk(a,Q,R);


    cout << "##################### CPU" << endl;
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
    }

    cout << "##################### GPU 5" << endl;
    for (int r = 20; r < std::min({A,B,MAX_BLOCK}); r += 5) {
        start = std::chrono::steady_clock::now();
        block_householder_qr_gpu5(a, Q, R, r);
        cout << "block " << r << ": "
             << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()
             << "ms" << endl;
        chk(a,Q,R);
    }

    return 0;
}