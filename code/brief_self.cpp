#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <bitset>
using namespace cv;
using namespace std;
extern int ORB_pattern[256 * 4];
typedef vector<uint32_t>  Desctype;
void brief_self(const Mat& img, const vector<KeyPoint>& kp_vec, vector<KeyPoint>& nice_kp_vec, vector<Desctype>& descriptors) {
    // ******************* convert cv to Eigen *******************
    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    Eigen::MatrixXi img_e(img_gray.rows, img_gray.cols);
    cv2eigen(img_gray, img_e);
    
    // ******************* caculate descriptor *******************
    int half_patch_size = 16;
    
    // nice_kp_vec
    // vector<KeyPoint> nice_kp_vec;
    for(int i=0; i<kp_vec.size(); ++i) {
        KeyPoint kp_t = kp_vec[i];
        int x = kp_t.pt.x;
        int y = kp_t.pt.y;
        if(x-half_patch_size<0 || x+half_patch_size>img_e.cols()-1 || 
           y-half_patch_size<0 || y+half_patch_size>img_e.rows()-1) {
            continue;
        }
        else
            nice_kp_vec.push_back(kp_t);
    }

    // random 256 pairs of points: follows a guassion distribution
    // vector<pair<int, int>> random_pairs;

    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::normal_distribution<double> dist(0, 4);
    // for(int i=0; i<256; ++i) {
    //     // guassion distribution
    //     int x1 = static_cast<int>(dist(gen));
    //     x1 = std::min(half_patch_size, std::max(-half_patch_size, x1));
    //     int y1 = static_cast<int>(dist(gen));
    //     y1 = std::min(half_patch_size, std::max(-half_patch_size, y1));
    //     int x2 = static_cast<int>(dist(gen));
    //     x2 = std::min(half_patch_size, std::max(-half_patch_size, x2));
    //     int y2 = static_cast<int>(dist(gen));
    //     y2 = std::min(half_patch_size, std::max(-half_patch_size, y2));
    //     // uniform distribution
    //     // int x1 = rand()%(half_patch_size*2+1) - half_patch_size;
    //     // int y1 = rand()%(half_patch_size*2+1) - half_patch_size;
    //     // int x2 = rand()%(half_patch_size*2+1) - half_patch_size;
    //     // int y2 = rand()%(half_patch_size*2+1) - half_patch_size;
    //     // cout << x1 << y1 << x2 << y2 << endl;
    //     random_pairs.push_back(make_pair(x1, y1));
    //     random_pairs.push_back(make_pair(x2, y2));
    // }
    // theta -> rotate -> descriptor

    for(int i=0; i<nice_kp_vec.size(); ++i) {
        KeyPoint kp_t = nice_kp_vec[i];
        int x = kp_t.pt.x;
        int y = kp_t.pt.y;    

        // caculate sin and cos
        float sum_dx=0, sum_dy=0;
        for(int dx=-half_patch_size/2; dx < half_patch_size/2; ++dx) {
            for(int dy=-half_patch_size/2; dy < half_patch_size/2; ++dy) {
                uchar pixel = img_gray.at<uchar>(y + dy, x + dx);
                sum_dx += dx*pixel;
                sum_dy += dy*pixel;
            }
        }
        float m_sqrt = sqrt(sum_dx*sum_dx+sum_dy*sum_dy) + 1e-10;  
        float sin_theta = sum_dy / m_sqrt;
        float cos_theta = sum_dx / m_sqrt;
        Desctype desc_256;
        // cout << string(10, '*') << "point: " << i << string(10, '*') << endl;
        for(int j=0; j<8; ++j) {

            uint32_t desc_32 = 0;
            for(int k=0; k<32; ++k) {
                int x1 = ORB_pattern[(j*32+k)];
                int y1 = ORB_pattern[(j*32+k)+1];
                int x2 = ORB_pattern[(j*32+k)+2];
                int y2 = ORB_pattern[(j*32+k)+3];
                // rotate
                float x1_r = x1*cos_theta - y1*sin_theta + x;
                float y1_r = y1*cos_theta + x1*sin_theta + y;
                float x2_r = x2*cos_theta - y2*sin_theta + x;
                float y2_r = y2*cos_theta + x2*sin_theta + y;
                // cout << x1_r << ", " << y1_r << endl;
                if(img_gray.at<uchar>(y1_r, x1_r) > img_gray.at<uchar>(y2_r, x2_r))
                    desc_32 |= 1<<k;
            }
            // cout << desc_32 << " ";
            desc_256.push_back(desc_32);
        }
        // cout << endl;
        descriptors.push_back(desc_256);
        
    }
    
    
}
