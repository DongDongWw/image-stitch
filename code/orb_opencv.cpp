#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Function to find keypoints and descriptors using ORB
void findKeypointsAndDescriptors(Mat image, vector<KeyPoint>& keypoints, Mat& descriptors) {
    Ptr<ORB> orb = ORB::create();
    orb->detectAndCompute(image, noArray(), keypoints, descriptors);
}

// Function to match keypoints between two images
void matchKeypoints(Mat descriptors1, Mat descriptors2, vector<DMatch>& matches) {
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    matcher->match(descriptors1, descriptors2, matches);
}

// Function to find homography matrix using RANSAC algorithm
Mat findHomography(vector<Point2f> points1, vector<Point2f> points2) {
    Mat homography;
    vector<uchar> inliers(points1.size(), 0);
    homography = findHomography(points1, points2, RANSAC, 3, inliers);
    return homography;
}

int main() {
    // Read images
    Mat image1 = imread("../../campus/campus_001.jpg");
    Mat image2 = imread("../../campus/campus_002.jpg");
    
    // Find keypoints and descriptors for both images
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    findKeypointsAndDescriptors(image1, keypoints1, descriptors1);
    findKeypointsAndDescriptors(image2, keypoints2, descriptors2);
    
    // Match keypoints between images
    vector<DMatch> matches;
    matchKeypoints(descriptors1, descriptors2, matches);
    
    // Filter matches using Lowe's ratio test
    vector<DMatch> filteredMatches;
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i].distance < 0.7 * matches[i+1].distance) {
            filteredMatches.push_back(matches[i]);
        }
    }
    cout << "Number of matches: " << matches.size() << endl;
    Mat image_show;
    drawMatches(image1, keypoints1, image2, keypoints2, filteredMatches, image_show);
	imshow("matches", image_show);
	imwrite("matches.png", image_show);
	waitKey(0);

    // Find corresponding points in both images
    // vector<Point2f> points1, points2;
    // for (int i = 0; i < filteredMatches.size(); i++) {
    //     points1.push_back(keypoints1[filteredMatches[i].queryIdx].pt);
    //     points2.push_back(keypoints2[filteredMatches[i].trainIdx].pt);
    // }
    
    // // Find homography matrix using RANSAC algorithm
    // Mat homography = findHomography(points1, points2);
    
    // // Warp second image to align with the first image
    // Mat warpedImage;
    // warpPerspective(image2, warpedImage, homography, Size(image1.cols + image2.cols, image1.rows));
    
    // // Copy the first image to the output image
    // Mat outputImage = Mat::zeros(image1.rows, image1.cols + image2.cols, image1.type());
    // image1.copyTo(outputImage(Rect(0, 0, image1.cols, image1.rows)));
    
    // // Copy the warped second image to the output image
    // Mat half(outputImage, Rect(image1.cols, 0, image2.cols, image2.rows));
    // warpedImage.copyTo(half);
    
    // // Display the output image
    // imshow("Stitched Image", outputImage);
    // waitKey(0);
    
    return 0;
}
