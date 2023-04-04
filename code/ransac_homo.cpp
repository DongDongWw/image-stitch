#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

using namespace cv;
using namespace std;

void calc_homography(const vector<KeyPoint>& kp_vec1, const vector<KeyPoint>& kp_vec2, Eigen::MatrixXf& H) {
    // ******************* least square *******************
    int num_of_kp = kp_vec1.size();
    Eigen::MatrixXf A(2*num_of_kp, 8);
    Eigen::MatrixXf b(2*num_of_kp, 1);
    for(int i=0; i<num_of_kp; ++i) {
        A(2*i, 0) = kp_vec1[i].pt.x;
        A(2*i, 1) = kp_vec1[i].pt.y;
        A(2*i, 2) = 1;
        A(2*i, 3) = 0;
        A(2*i, 4) = 0;
        A(2*i, 5) = 0;
        A(2*i, 6) = -kp_vec1[i].pt.x*kp_vec2[i].pt.x;
        A(2*i, 7) = -kp_vec1[i].pt.y*kp_vec2[i].pt.x;
        A(2*i+1, 0) = 0;
        A(2*i+1, 1) = 0;
        A(2*i+1, 2) = 0;
        A(2*i+1, 3) = kp_vec1[i].pt.x;
        A(2*i+1, 4) = kp_vec1[i].pt.y;
        A(2*i+1, 5) = 1;
        A(2*i+1, 6) = -kp_vec1[i].pt.x*kp_vec2[i].pt.y;
        A(2*i+1, 7) = -kp_vec1[i].pt.y*kp_vec2[i].pt.y;
        b(2*i, 0) = kp_vec2[i].pt.x;
        b(2*i+1, 0) = kp_vec2[i].pt.y;
    }
    Eigen::MatrixXf x = (A.transpose()*A).inverse()*A.transpose()*b;
    H << x(0, 0), x(1, 0), x(2, 0),
         x(3, 0), x(4, 0), x(5, 0),
         x(6, 0), x(7, 0), 1;
}

float reproject_and_ssd(const vector<KeyPoint>& kp_vec1, const vector<KeyPoint>& kp_vec2, const vector<DMatch> &matches, const Eigen::MatrixXf& H) {
    float ssd = 0;
    for(auto& m : matches) {
        float x1 = kp_vec1[m.queryIdx].pt.x;
        float y1 = kp_vec1[m.queryIdx].pt.y;
        float x2 = kp_vec1[m.trainIdx].pt.x;
        float y2 = kp_vec1[m.trainIdx].pt.y; 
        float r_x2, r_y2;
        r_x2 = (H(0,0)*x1+H(0,1)*y1+H(0,2))/(H(2,0)*x1+H(2,1)*y1+1); 
        r_y2 = (H(1,0)*x1+H(1,1)*y1+H(1,2))/(H(2,0)*x1+H(2,1)*y1+1);

        ssd += (x2-r_x2)*(x2-r_x2)+(y2-r_y2)*(y2-r_y2);
    }
    return ssd;
}
void ransac_homo(const vector<KeyPoint>& kp_vec1, const vector<KeyPoint>& kp_vec2, const vector<DMatch> &matches, Eigen::MatrixXf& H) {
    // RANSAC
    // 1. randomly sample 4 points to calculate homography matrix
    // 2. use the above h matrix to **reproject** points and count the number of good matches
    // 3. choose the best h matrix by comparing all h matrices

    int sample_num = 10;
    int m_size = matches.size();

    Eigen::MatrixXf H_best;
    float min_ssd = 1.0e20;
    for(int i=0; i<sample_num; ++i) {
        // 4 random indexes
        vector<int> r_nums;
        for(int j=0; j<4; ++j)
            r_nums.push_back(rand()%m_size);

        // 4 random point pairs
        Eigen::MatrixXf H_temp;
        vector<KeyPoint> pts_1, pts_2;
        for(auto& idx : r_nums) {
            KeyPoint k1, k2;
            int idx_1 ,idx_2;
            idx_1 = matches[idx].queryIdx;
            idx_2 = matches[idx].trainIdx;

            k1.pt = kp_vec1[idx_1].pt;
            k2.pt = kp_vec2[idx_1].pt;
            pts_1.push_back(k1);
            pts_2.push_back(k2);
        }
        // calculate one h matrix
        calc_homography(pts_1, pts_2, H_temp);
        float ssd = reproject_and_ssd(kp_vec1, kp_vec2, matches, H_temp);
        if(ssd < min_ssd) {
            min_ssd = ssd;
            H_best = H_temp;
        }

    }
}