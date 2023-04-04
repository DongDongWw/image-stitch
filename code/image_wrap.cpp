#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

using namespace cv;
using namespace std;

void image_stitch(const Mat& img, const Mat& img_wrap, const Eigen::MatrixXf& H, Mat& img_stitch) {
    img_stitch = Mat::zeros(img.rows, img.cols + img_wrap.cols, CV_8UC3);
    img_stitch(Rect(0, 0, img.cols, img.rows)) = img;
    for(int x=0; x<img_wrap.cols; ++x){
        for(int y=0; y<img_wrap.rows; ++y){
            int r_x1 = (int)((H(0,0)*x+H(0,1)*y+H(0,2))/(H(2,0)*x+H(2,1)*y+1));
            int r_y1 = (int)((H(1,0)*x+H(1,1)*y+H(1,2))/(H(2,0)*x+H(2,1)*y+1));
            if(r_x1 >= 0 && r_x1 < img_stitch.cols && r_y1 >= 0 && r_y1 < img_stitch.rows) {
                img_stitch.at<Vec3b>(y, x) = img_wrap.at<Vec3b>(r_y1, r_x1);
            }
        }
    }

}