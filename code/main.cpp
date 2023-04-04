#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <bitset>

using namespace std;
using namespace cv;


void harris_opencv(const Mat& img, vector<KeyPoint>& kp_vec);
void harris_self(const Mat& img, int num_of_corner, vector<KeyPoint>& kp_vec);
void brief_self(const Mat& img, const vector<KeyPoint>& kp_vec, vector<KeyPoint>& nice_kp_vec, vector<vector<uint32_t>>& descriptors);
void brute_force_match(const vector<vector<uint32_t>> &desc1, const vector<vector<uint32_t>> &desc2, vector<DMatch> &matches);
void brief_gao(const cv::Mat &img, vector<cv::KeyPoint> &keypoints, vector<KeyPoint>& nice_kp_vec,vector<vector<uint32_t>> &descriptors);
void ransac_homo(const vector<KeyPoint>& kp_vec1, const vector<KeyPoint>& kp_vec2, const vector<DMatch> &matches, Eigen::MatrixXf& H);
void image_stitch(const Mat& img, const Mat& img_wrap, const Eigen::MatrixXf& H, Mat& img_stitch);
int main()
{	
	int num_of_corner = 200;
	string path_0 = "../../campus/campus_001.jpg";
	string path_1 = "../../campus/campus_000.jpg";
	Mat img_000;
	Mat img_001;
	Mat img_stitch;
	vector<KeyPoint> kp_000;
	vector<KeyPoint> kp_001;
	vector<vector<uint32_t>> descriptors_000;
	vector<vector<uint32_t>> descriptors_001;
	vector<KeyPoint> nice_kp_vec_000;
	vector<KeyPoint> nice_kp_vec_001;
	vector<DMatch> matches;
	Eigen::MatrixXf H;
	img_000 = imread(path_0, IMREAD_COLOR);
	img_001 = imread(path_1, IMREAD_COLOR);
	harris_self(img_000, num_of_corner, kp_000);
	harris_self(img_001, num_of_corner, kp_001);

	brief_self(img_000, kp_000, nice_kp_vec_000, descriptors_000);
	brief_self(img_001, kp_001, nice_kp_vec_001, descriptors_001);

	brute_force_match(descriptors_000, descriptors_001, matches);
	ransac_homo(nice_kp_vec_000, nice_kp_vec_001, matches, H);
	image_stitch(img_000, img_001, H, img_stitch);
	imshow("stitch", img_stitch);
	waitKey(0);
	// Mat image_show;
	// drawMatches(img_000, nice_kp_vec_000, img_001, nice_kp_vec_001, matches, image_show);
	// imshow("matches", image_show);
	// imwrite("matches.png", image_show);
	// waitKey(0);


}
