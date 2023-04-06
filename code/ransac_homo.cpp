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
    // Eigen::MatrixXf x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    H << x(0, 0), x(1, 0), x(2, 0),
         x(3, 0), x(4, 0), x(5, 0),
         x(6, 0), x(7, 0), 1;
}

int reproject_and_loss(const vector<KeyPoint>& kp_vec1, const vector<KeyPoint>& kp_vec2, const vector<DMatch> &matches, const Eigen::MatrixXf& H) {
    // float ssd = 0;
    int good_match_num = 0;
    for(auto& m : matches) {
        float x1 = kp_vec1[m.queryIdx].pt.x;
        float y1 = kp_vec1[m.queryIdx].pt.y;
        float x2 = kp_vec2[m.trainIdx].pt.x;
        float y2 = kp_vec2[m.trainIdx].pt.y; 
        float r_x2, r_y2;
        r_x2 = (H(0,0)*x1+H(0,1)*y1+H(0,2))/(H(2,0)*x1+H(2,1)*y1+1); 
        r_y2 = (H(1,0)*x1+H(1,1)*y1+H(1,2))/(H(2,0)*x1+H(2,1)*y1+1);

        float sd = (x2-r_x2)*(x2-r_x2)+(y2-r_y2)*(y2-r_y2);
        if(sd < 5)
            ++good_match_num;
    }
    return good_match_num;
}
void ransac_homo(const vector<KeyPoint>& kp_vec1, const vector<KeyPoint>& kp_vec2, const vector<DMatch> &matches, Eigen::MatrixXf& H) {
    // RANSAC
    // 1. randomly sample 4 points to calculate homography matrix
    // 2. use the above h matrix to **reproject** points and count the number of good matches
    // 3. choose the best h matrix by comparing all h matrices

    int sample_num = 200;
    int m_size = matches.size();

    Eigen::MatrixXf H_best(3, 3);
    int max_match_num = 0;
    for(int i=0; i<sample_num; ++i) {

        // 4 different random indexes
        vector<int> r_nums;
        for(int j=0; j<4; ++j) {

            int r_num = rand()%m_size;
            bool f = true;
            for(int k=0; k<r_nums.size(); ++k) {
                if(r_nums[k] == r_num){
                    f = false;
                    break;
                }
            }
            if(f==true) r_nums.push_back(r_num);
            else        --j;
        }
            

        // 4 random point pairs
        Eigen::MatrixXf H_temp(3, 3);
        vector<KeyPoint> pts_1, pts_2;
        for(auto& idx : r_nums) {
            KeyPoint k1, k2;
            int idx_1 ,idx_2;
            idx_1 = matches[idx].queryIdx;
            idx_2 = matches[idx].trainIdx;

            k1.pt.x = kp_vec1[idx_1].pt.x;
            k1.pt.y = kp_vec1[idx_1].pt.y;
            k2.pt.x = kp_vec2[idx_2].pt.x;
            k2.pt.y = kp_vec2[idx_2].pt.y;
            pts_1.push_back(k1);
            pts_2.push_back(k2);
        }
        // calculate one h matrix
        calc_homography(pts_1, pts_2, H_temp);
        // reproject and calculate ssd
        int match_num = reproject_and_loss(kp_vec1, kp_vec2, matches, H_temp);
        if(match_num > max_match_num) {
            max_match_num = match_num;
            H_best = H_temp;
        }
    }
    H = H_best;
}