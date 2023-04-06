#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void bright_consistency(Mat& img) {
    cv::Mat img_yuv;
    cv::cvtColor(img, img_yuv, cv::COLOR_BGR2YUV);

    // Split the image into channels
    std::vector<cv::Mat> channels;
    cv::split(img_yuv, channels);

    // Equalize the Y channel only
    cv::equalizeHist(channels[0], channels[0]);

    // Merge the result back into the image
    cv::merge(channels, img_yuv);

    // Convert back to BGR
    cv::cvtColor(img_yuv, img, cv::COLOR_YUV2BGR);
}