#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <bitset>
#include <filesystem>
using namespace std;
using namespace cv;

typedef vector<uint32_t> Desctype;
// void harris_opencv(const Mat& img, vector<KeyPoint>& kp_vec);
void harris_self(const Mat& img, int num_of_corner, vector<KeyPoint>& kp_vec);
void brief_self(const Mat& img, const vector<KeyPoint>& kp_vec, vector<KeyPoint>& nice_kp_vec, vector<Desctype>& descriptors);
void brute_force_match(const vector<Desctype> &desc1, const vector<Desctype> &desc2, vector<DMatch> &matches);
// void brief_gao(const cv::Mat &img, vector<cv::KeyPoint> &keypoints, vector<KeyPoint>& nice_kp_vec,vector<vector<uint32_t>> &descriptors);
void ransac_homo(const vector<KeyPoint>& kp_vec1, const vector<KeyPoint>& kp_vec2, const vector<DMatch> &matches, Eigen::MatrixXf& H);
void image_stitch(const Mat& img, const Mat& img_wrap, const Eigen::MatrixXf& H, Mat& img_stitch);
void rm_crack(Mat& img);
void bright_consistency(Mat& img);
int main()
{	
	std::string path = "../../hw";
	vector<string> img_paths; 
	for (const auto & entry : filesystem::directory_iterator(path)){
		img_paths.push_back(entry.path());
	}
	sort(img_paths.begin(), img_paths.end());
	for(auto& s : img_paths)
	 	cout << s << endl;

	// calculate homograpy matrices
	int img_num = img_paths.size();
	vector<Eigen::MatrixXf> homo_vec;
	for(int i=0; i<img_num-1; ++i) {
		string path_wrap = img_paths[i],
			   path_fix  = img_paths[i+1];
		Mat img_wrap, img_fix;
		vector<KeyPoint> kps_wrap;
		vector<KeyPoint> kps_fix;
		vector<Desctype> des_wrap;
		vector<Desctype> des_fix;
		vector<KeyPoint> kps_wrap_f;
		vector<KeyPoint> kps_fix_f;
		vector<DMatch> matches;
		Eigen::MatrixXf homograpy;
		img_wrap = imread(path_wrap, IMREAD_COLOR);
		img_fix  = imread(path_fix, IMREAD_COLOR);

		harris_self(img_wrap, 1500, kps_wrap);
		harris_self(img_fix, 1500, kps_fix);
		brief_self(img_wrap, kps_wrap, kps_wrap_f, des_wrap);
		brief_self(img_fix, kps_fix, kps_fix_f, des_fix);

		brute_force_match(des_wrap, des_fix, matches);
		ransac_homo(kps_wrap_f, kps_fix_f, matches, homograpy);
		homo_vec.push_back(homograpy);

		cout << to_string(i)+"->"+to_string(i+1)+" matches number: " << matches.size()<<endl;
		Mat image_show;
		drawMatches(img_wrap, kps_wrap_f, img_fix, kps_fix_f, matches, image_show);
		imwrite("matches of "+to_string(i)+"->"+to_string(i+1)+".png", image_show);

		Mat img_temp;
		image_stitch(img_fix, img_wrap, homograpy, img_temp);
		imwrite("project of "+to_string(i)+"->"+to_string(i+1)+".png", img_temp);
	}

	Mat img_stitch;
	img_stitch = imread(img_paths[0], IMREAD_COLOR);

	for(int i=1; i<img_num; ++i){
		string path_fix  = img_paths[i];
		Mat img_fix;
		Mat img_temp;
		Eigen::MatrixXf homograpy = homo_vec[i-1];
		img_fix  = imread(path_fix, IMREAD_COLOR);

		image_stitch(img_fix, img_stitch, homograpy, img_temp);
		img_stitch = img_temp.clone();
		imwrite("stitch_"+to_string(i)+".png", img_stitch);
		rm_crack(img_stitch);
		imwrite("stitch_"+to_string(i)+"_filter"+".png", img_stitch);
		bright_consistency(img_stitch);
		imwrite("stitch_"+to_string(i)+"_filter"+"_bright_consistence"+".png", img_stitch);
	}


}
