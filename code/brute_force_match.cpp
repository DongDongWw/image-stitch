#include <opencv2/opencv.hpp>
#include <bitset>
using namespace cv;
using namespace std;
typedef vector<uint32_t>  Desctype;

void brute_force_match(const vector<Desctype> &desc1, const vector<Desctype> &desc2, vector<DMatch> &matches) {
  const int d_max = 40;

  for (int i = 0; i < desc1.size(); ++i) {
    cv::DMatch m{i, 0, 256};
    for (int j = 0; j < desc2.size(); ++j) {
      int distance = 0;
      for (int k = 0; k < 8; k++) {
        uint32_t xor_desc = (desc1[i][k] ^ desc2[j][k]);
		    bitset<32> xor_bits(xor_desc);
        distance += xor_bits.count();
      }
      if (distance < d_max && distance < m.distance) {
        m.distance = distance;
        m.trainIdx = j;
      }
    }
    if (m.distance < d_max) {
      matches.push_back(m);
    }
  }
}