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
    Eigen::MatrixXi img_e(img_gray.rows, img_gray.cols);
    cv2eigen(img_gray, img_e);
    
    // ******************* sobel operator ******************* 
    Eigen::MatrixXi padded_img_e = Eigen::MatrixXi::Zero(img_e.rows()+2, img_e.cols()+2);
    padded_img_e.block(1,1, img_e.rows(), img_e.cols()) = img_e;

    Eigen::MatrixXi sobel_x_e(3, 3);
    Eigen::MatrixXi sobel_y_e(3, 3);
    sobel_x_e << 1,  0, -1,
                 2,  0, -2,
                 1,  0, -1;
    sobel_y_e << 1,  2,  1,
                 0,  0,  0,
                -1, -2, -1;

    Eigen::MatrixXi drt_x_e(img_e.rows(), img_e.cols());
    Eigen::MatrixXi drt_y_e(img_e.rows(), img_e.cols());
    for(int i=1; i<img_e.rows()+1; ++i) {
        for(int j=1; j<img_e.cols()+1; ++j) {
            int temp_x, temp_y;
            Eigen::MatrixXi temp = padded_img_e.block(i-1, j-1, 3, 3);
            
            temp_x = temp.cwiseProduct(sobel_x_e).sum();
            temp_y = temp.cwiseProduct(sobel_y_e).sum();
            
            drt_x_e(i-1, j-1) = temp_x;
            drt_y_e(i-1, j-1) = temp_y;
        }
    }

    // ******************* structure tensor *******************
    // repeat the edge pixels
    // x
    Eigen::MatrixXi padded_drt_x_e(img_e.rows()+2, img_e.cols()+2);
    padded_drt_x_e.block(1, 1, img_e.rows(), img_e.cols()) = drt_x_e;
    padded_drt_x_e(0, 0) = 0;
    padded_drt_x_e(img_e.rows()+1, 0) = 0;
    padded_drt_x_e(0, img_e.cols()+1) = 0;
    padded_drt_x_e(img_e.rows()+1, img_e.cols()+1) = 0;
    padded_drt_x_e.block(1, 0, img_e.rows(), 1) = drt_x_e.col(0);
    padded_drt_x_e.block(0, 1, 1, img_e.cols()) = drt_x_e.row(0);
    padded_drt_x_e.block(1, img_e.cols()+1, img_e.rows(), 1) = drt_x_e.col(img_e.cols()-1);
    padded_drt_x_e.block(img_e.rows()+1, 1, 1, img_e.cols()) = drt_x_e.row(img_e.rows()-1);
    
    // y
    Eigen::MatrixXi padded_drt_y_e(img_e.rows()+2, img_e.cols()+2);
    padded_drt_y_e.block(1, 1, img_e.rows(), img_e.cols()) = drt_y_e;
    padded_drt_y_e(0, 0) = 0;
    padded_drt_y_e(img_e.rows()+1, 0) = 0;
    padded_drt_y_e(0, img_e.cols()+1) = 0;
    padded_drt_y_e(img_e.rows()+1, img_e.cols()+1) = 0;
    padded_drt_y_e.block(1, 0, img_e.rows(), 1) = drt_y_e.col(0);
    padded_drt_y_e.block(0, 1, 1, img_e.cols()) = drt_y_e.row(0);
    padded_drt_y_e.block(1, img_e.cols()+1, img_e.rows(), 1) = drt_y_e.col(img_e.cols()-1);
    padded_drt_y_e.block(img_e.rows()+1, 1, 1, img_e.cols()) = drt_y_e.row(img_e.rows()-1);
    
    // calculate R without store the structure tensor
    int blocksize = 3;
    float k = 0.04;
    Eigen::MatrixXd R(img_e.rows(), img_e.cols());

    for(int i=1; i<img_e.rows()+1; ++i) {
        for(int j=1; j<img_e.cols()+1; ++j) {
            int squr_sum_x = 0,
                squr_sum_y = 0,
                sum_xy = 0;
            for(int k=0; k<blocksize; ++k) {
                for(int l=0; l<blocksize; ++l) {
                    squr_sum_x += padded_drt_x_e(i-1+k, j-1+l) * padded_drt_x_e(i-1+k, j-1+l);
                    squr_sum_y += padded_drt_y_e(i-1+k, j-1+l) * padded_drt_y_e(i-1+k, j-1+l);
                    sum_xy += padded_drt_x_e(i-1+k, j-1+l) * padded_drt_y_e(i-1+k, j-1+l);
                }
            }

            Eigen::Matrix2d temp;
            double r;

            temp << squr_sum_x, sum_xy,
                    sum_xy, squr_sum_y;
            r = temp.determinant()-k*(temp.trace()*temp.trace());
            R(i-1, j-1) = r;
        }
    }
    cout << R.row(0) << endl;

    // ******************* non-maximum suppression *******************
    Eigen::MatrixXd R_nms(img_e.rows(), img_e.cols());
    R_nms = R;
    R_nms.block(0, 0, 1, img_e.cols()) = Eigen::MatrixXd::Zero(1, img_e.cols());
    R_nms.block(img_e.rows()-1, 0, 1, img_e.cols()) = Eigen::MatrixXd::Zero(1, img_e.cols());
    R_nms.block(0, 0, img_e.rows(), 1) = Eigen::MatrixXd::Zero(img_e.rows(), 1);
    R_nms.block(0, img_e.cols()-1, img_e.rows(), 1) = Eigen::MatrixXd::Zero(img_e.rows(), 1);
    
    // !! map不能自定义按值排序!!所以用vector<pair<pair<int, int>, double>>
    vector<pair<pair<int, int>, double>> sort_map;
    
    for(int i=1; i<img_e.rows()-1; ++i) {
        for(int j=1; j<img_e.cols()-1; ++j) {
            double r = R(i, j);
            if(r < R(i-1, j-1) || r < R(i-1, j) || r < R(i-1, j+1) ||
               r < R(i, j-1) || r < R(i, j+1) ||
               r < R(i+1, j-1) || r < R(i+1, j) || r < R(i+1, j+1)) {
                R_nms(i, j) = 0;
            }
            else {
                sort_map.push_back(make_pair(make_pair(i-1, j-1), r));
            }
        }
    }
    sort(sort_map.begin(), sort_map.end(), 
         [](const pair<pair<int, int>, double>& a, const pair<pair<int, int>, double>& b) {
                return a.second > b.second;
        });
    // select keypoints

    for(int i=0; i<num_of_corner; ++i) {
        double k_row = sort_map[i].first.first;
        double k_col = sort_map[i].first.second;
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