#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <map>
#include <algorithm>
using namespace cv;
using namespace std;

void harris_self(const Mat& img, int num_of_corner, vector<KeyPoint>& kp_vec) {

    // ******************* convert cv to Eigen *******************
    Mat img_gray; 
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    Eigen::MatrixXd img_e(img_gray.rows, img_gray.cols);
    cv2eigen(img_gray, img_e);
    
    // ******************* sobel operator ******************* 
    Eigen::MatrixXd sobel_x_e(3, 3);
    Eigen::MatrixXd sobel_y_e(3, 3);
    sobel_x_e << -1,  0, 1,
                 -2,  0, 2,
                 -1,  0, 1;
    sobel_y_e << -1, -2, -1,
                  0,  0,  0,
                  1,  2,  1;

    Eigen::MatrixXd drt_x_e = Eigen::MatrixXd::Zero(img_e.rows(), img_e.cols());
    Eigen::MatrixXd drt_y_e = Eigen::MatrixXd::Zero(img_e.rows(), img_e.cols());
    for(int i=1; i<img_e.rows()-1; ++i) {
        for(int j=1; j<img_e.cols()-1; ++j) {
            int temp_x, temp_y;
            Eigen::MatrixXd temp = img_e.block(i-1, j-1, 3, 3);
            
            temp_x = temp.cwiseProduct(sobel_x_e).sum();
            temp_y = temp.cwiseProduct(sobel_y_e).sum();
            
            drt_x_e(i, j) = temp_x;
            drt_y_e(i, j) = temp_y;
        }
    }
    Eigen::MatrixXd drt_x2_e = drt_x_e.cwiseProduct(drt_x_e);
    Eigen::MatrixXd drt_y2_e = drt_y_e.cwiseProduct(drt_y_e);
    Eigen::MatrixXd drt_xy_e = drt_x_e.cwiseProduct(drt_y_e);

    // ******************* structure tensor *******************
    // gaussion window
    double sigma = 2;
    int blocksize = 3*sigma+1;
    int offset = (blocksize-1)/2;
    Eigen::MatrixXd guassion_filter(blocksize, blocksize);
    for(int i=0; i<blocksize; ++i) {
        for(int j=0; j<blocksize; ++j) {
            int x2 = (i-offset)*(i-offset);
            int y2 = (j-offset)*(j-offset);
            guassion_filter(i, j) = exp(-(x2+y2)/(2*M_PI*sigma*sigma));
        }
    }
    // x padding
    Eigen::MatrixXd padded_drt_x_e = Eigen::MatrixXd::Zero(img_e.rows()+offset*2, img_e.cols()+offset*2);
    padded_drt_x_e.block(offset, offset, img_e.rows(), img_e.cols()) = drt_x_e;
    Eigen::MatrixXd padded_drt_y_e = Eigen::MatrixXd::Zero(img_e.rows()+offset*2, img_e.cols()+offset*2);
    padded_drt_y_e.block(offset, offset, img_e.rows(), img_e.cols()) = drt_y_e;

    // R matrix
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(img_e.rows(), img_e.cols());
    double k = 0.04;
    for(int i=offset+1; i<img_e.rows()-offset-1; ++i) {
        for(int j=offset+1; j<img_e.cols()-offset-1; ++j) {
            Eigen::MatrixXd temp_x2 = drt_x2_e.block(i-offset, j-offset, blocksize, blocksize);
            Eigen::MatrixXd temp_y2 = drt_y2_e.block(i-offset, j-offset, blocksize, blocksize);
            Eigen::MatrixXd temp_xy = drt_xy_e.block(i-offset, j-offset, blocksize, blocksize);
            Eigen::MatrixXd temp_x2_g = temp_x2.cwiseProduct(guassion_filter);
            Eigen::MatrixXd temp_y2_g = temp_y2.cwiseProduct(guassion_filter);
            Eigen::MatrixXd temp_xy_g = temp_xy.cwiseProduct(guassion_filter);
            double x2_sum = temp_x2_g.sum();
            double y2_sum = temp_y2_g.sum();
            double xy_sum = temp_xy_g.sum();
            Eigen::Matrix2f temp;
            temp << x2_sum, xy_sum, xy_sum, y2_sum;
            double R_value = x2_sum*y2_sum - xy_sum*xy_sum - k*(x2_sum+y2_sum)*(x2_sum+y2_sum);
            R(i, j) = R_value;
        }
    }

    // ******************* non-maximum suppression *******************
    Eigen::MatrixXd R_nms = Eigen::MatrixXd::Zero(img_e.rows(), img_e.cols());
    
    // !! map不能自定义按值排序!!所以用
    vector<pair<pair<int, int>, double>> R_vec;
    for(int i=offset+1; i<img_e.rows()-offset-1; ++i) {
        for(int j=offset+1; j<img_e.cols()-offset-1; ++j) {
            double r = R(i, j);
            if(r < R(i-1, j-1) || r < R(i-1, j) || r < R(i-1, j+1) ||
               r < R(i, j-1) || r < R(i, j+1) ||
               r < R(i+1, j-1) || r < R(i+1, j) || r < R(i+1, j+1)) {
                R_nms(i, j) = 0;
            }
            else {
                R_vec.push_back(make_pair(make_pair(i, j), r));
            }
        }
    }
    sort(R_vec.begin(), R_vec.end(), [](const pair<pair<int, int>, double>& a, const pair<pair<int, int>, double>& b) {
        return a.second > b.second;
    });
    // float max_r = R_nms.maxCoeff();
    // float min_r = R_nms.minCoeff();
    // R_nms = 0+ (R_nms.array()-min_r)/(max_r - min_r)*255;
    // int count = 0;
    // for(int i=0; i<R_nms.rows(); ++i) {
    //     for(int j=0; j<R_nms.cols(); ++j) {
    //         if(R_nms(i, j) > 20) {
    //             KeyPoint k_p;
    //             k_p.pt.y = i;
    //             k_p.pt.x = j;
    //             kp_vec.push_back(k_p);
    //             count++;
    //         }
    //     }
    // }

    for(int i=0; i<num_of_corner; ++i) {
        double k_row = R_vec[i].first.first;
        double k_col = R_vec[i].first.second;
        KeyPoint    k_p;
        k_p.pt.y = k_row;
        k_p.pt.x = k_col;
        kp_vec.push_back(k_p);
    }
    // cout << kp_vec << endl;
    // drawKeypoints(img, kp_vec, img);
    // imshow(name, img);
    // waitKey(0);
}