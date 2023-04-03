#include <string>
#include <opencv2/opencv.hpp>
#include <bitset>

using namespace std;
using namespace cv;


void harris_opencv(const string& path);
void harris_self(const Mat& img, int num_of_corner, vector<KeyPoint>& kp_vec);
void brief_self(const Mat& img, const vector<KeyPoint>& kp_vec, vector<KeyPoint>& nice_kp_vec, vector<vector<uint32_t>>& descriptors);
void brute_force_match(const vector<vector<uint32_t>> &desc1, const vector<vector<uint32_t>> &desc2, vector<DMatch> &matches);
void brief_gao(const cv::Mat &img, vector<cv::KeyPoint> &keypoints, vector<KeyPoint>& nice_kp_vec,vector<vector<uint32_t>> &descriptors);
int main()
{	
	int num_of_corner = 1000;
	string path_0 = "../../campus/campus_001.jpg";
	/************************************************/
	harris_opencv(path_0);
	vector<KeyPoint> kp_vec;
	Mat img = imread(path_0);
	harris_self(img, 100, kp_vec);
	/************************************************/

	/*
	string path_1 = "../../campus/campus_002.jpg";
	Mat img_000;
	Mat img_001;
	vector<KeyPoint> kp_000;
	vector<KeyPoint> kp_001;
	vector<vector<uint32_t>> descriptors_000;
	vector<vector<uint32_t>> descriptors_001;
	vector<KeyPoint> nice_kp_vec_000;
	vector<KeyPoint> nice_kp_vec_001;
	vector<DMatch> matches;

	img_000 = imread(path_0, IMREAD_COLOR);
	img_001 = imread(path_1, IMREAD_COLOR);
	harris_self(img_000, num_of_corner, kp_000);
	harris_self(img_001, num_of_corner, kp_001);

	brief_self(img_000, kp_000, nice_kp_vec_000, descriptors_000);
	brief_self(img_001, kp_001, nice_kp_vec_001, descriptors_001);
	// brief_gao(img_000, kp_000, nice_kp_vec_000, descriptors_000);
	// brief_gao(img_001, kp_001, nice_kp_vec_001, descriptors_001);

	brute_force_match(descriptors_000, descriptors_001, matches);

	Mat image_show;
	drawMatches(img_000, nice_kp_vec_000, img_001, nice_kp_vec_001, matches, image_show);
	imshow("matches", image_show);
	imwrite("matches.png", image_show);
	waitKey(0);
	*/

	// vector<uint32_t> desc_1 = descriptors_001[0]; 
	// cout << desc_1[0] << endl;
	// for(int i=0; i<descriptors_000.size(); ++i) {
	// 	vector<uint32_t> desc_0 = descriptors_000[i];

	// 	int max_match_dist = 256;
	// 	cout << string(10, '*') << "point: " << i << string(10, '*') << endl;

	// 	for(int j=0; j<descriptors_001.size(); ++j) {
	// 		int match_num = 0;
	// 		vector<uint32_t> desc_1 = descriptors_001[j];
	// 		for(int k=0; k<desc_0.size(); ++k) {
	// 			uint32_t xor_desc = (desc_0[k] ^ desc_1[k]);
	// 			bitset<32> xor_bits(xor_desc);  // convert integer to bitset
	// 			// cout << xor_bits << endl;
	// 			match_num += xor_bits.count();  // count the number of bit 1's
	// 		}
	// 		if(match_num < max_match_dist) {
	// 			max_match_dist = match_num;
	// 		}
	// 	}
	// 	cout << "max_match_dist: " << max_match_dist << endl;
	// }


	// waitKey(0);
}
