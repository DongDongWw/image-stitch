#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void rm_crack(Mat& img) {
    // median filter
    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    img_gray.convertTo(img_gray, CV_8U);
    // cout << img_gray << endl;
    int w_size = 2;
    int half_w_size = w_size/2;
    for(int i=half_w_size; i<img.rows-half_w_size; ++i) {
        for(int j=half_w_size; j<img.cols-half_w_size; ++j) {

            if(img_gray.at<uchar>(i, j) != 0) continue;
            
            else {
                vector<pair<pair<int, int>, uchar>> window;
                for(int x=0; x<w_size; ++x) {
                    for(int y=0; y<w_size; ++y) {
                        int temp_x = i+x-half_w_size;
                        int temp_y = j+y-half_w_size;
                        if(img_gray.at<uchar>(temp_x, temp_y) == 0) continue;
                        else{
                            uchar pix = img_gray.at<uchar>(temp_x, temp_y);
                            window.push_back(make_pair(make_pair(x,y), pix));
                        }
                    }
                }
                if(window.size()!=0){
                    sort(window.begin(), window.end(), [](pair<pair<int, int>, uchar> a, pair<pair<int, int>, uchar> b) {
                        return a.second < b.second; });
                    pair<int,int> pos = window[window.size()/2].first;
                    img.at<Vec3b>(i, j)[0] = img.at<Vec3b>(i+pos.first-half_w_size, j+pos.second-half_w_size)[0];
                    img.at<Vec3b>(i, j)[1] = img.at<Vec3b>(i+pos.first-half_w_size, j+pos.second-half_w_size)[1];
                    img.at<Vec3b>(i, j)[2] = img.at<Vec3b>(i+pos.first-half_w_size, j+pos.second-half_w_size)[2];
                }
            }

        }
    }
}