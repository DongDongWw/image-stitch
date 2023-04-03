#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

void harris_opencv(const string& path) {
	Mat img = imread(path, IMREAD_COLOR);
	if (!img.data)
	{
		cout << "读取图像错误，请确认图像文件是否正确" << endl;
		return;
	}

	//转成灰度图像
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	cout << gray.size() << endl;
	//计算Harris系数
	Mat harris;
	int blockSize = 2;     // 邻域半径
	int apertureSize = 3;  // 邻域大小
	cornerHarris(gray, harris, blockSize, apertureSize, 0.04);
	//归一化便于进行数值比较和结果显示
	Mat harrisn;
	cout << harris<< endl;
	normalize(harris, harrisn, 0, 255, NORM_MINMAX);
	//将图像的数据类型变成CV_8U
	cout << harrisn.size() << endl;
	convertScaleAbs(harrisn, harrisn);
	//寻找Harris角点
	vector<KeyPoint> keyPoints;
	for (int row = 0; row < harrisn.rows; row++)
	{
		for (int col = 0; col < harrisn.cols; col++)
		{
			int R = harrisn.at<uchar>(row, col);
			if (R > 125)
			{
				//向角点存入KeyPoint中
				KeyPoint keyPoint;
				keyPoint.pt.y = row;
				keyPoint.pt.x = col;
				keyPoints.push_back(keyPoint);
			}
		}
	}
	
}