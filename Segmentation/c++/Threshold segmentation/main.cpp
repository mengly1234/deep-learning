#include<iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

// 使用rgb值来分割
void rgb_segmation(cv::Mat& img) {

	int H = img.rows;
	int W = img.cols;
	//std::cout << img.at<cv::Vec3b>(0,0) << std::endl;

	cv::Mat visual(H, W, CV_8UC3);

	for (int h = 0; h < H; h++) {
		for (int w = 0; w < W; w++) {
			cv::Vec3b piexl = img.at<cv::Vec3b>(h, w);
			int R = static_cast<int>(piexl[2]);
			int G = static_cast<int>(piexl[1]);
			int B = static_cast<int>(piexl[0]);
			float ratio = B / float(R + G + B);
		
			if (ratio < 0.35) {
				cv::circle(visual, cv::Point(w, h), 1, cv::Scalar(0, 255, 0), -1);
			}
		}
	}

	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

	// 腐蚀
	cv::Mat erodedImage;
	cv::erode(visual, erodedImage, element);

	// 膨胀
	cv::Mat dilatedImage;
	cv::dilate(erodedImage, dilatedImage, element);

	cv::Mat result;
	cv::addWeighted(img, 1, dilatedImage, 0.2, 0, result);

	cv::imshow("result",result);
	cv::waitKey(0);

	cv::Mat result_y;
	cv::addWeighted(img, 1, visual, 0.2, 0, result_y);

	cv::imshow("result_y", result_y);
	cv::waitKey(0);
}

// 使用hsv值来分割
void hsv_segmation(cv::Mat img){

	cv::Mat img_hsv;

	cv::cvtColor(img, img_hsv, cv::COLOR_BGR2HSV);

	int H = img.rows;
	int W = img.cols;

	cv::Mat visual(H, W, CV_8UC3);

	for (int h = 0; h < H; h++) {
		for (int w = 0; w < W; w++) {
			cv::Vec3b hsv = img_hsv.at<cv::Vec3b>(h, w);
			// std::cout << int(hsv[0])*2 << " " << int(hsv[1])/255.0f << " " << int(hsv[2])/255.0f << " " << std::endl;
			if (float(hsv[0]) < 40) {
				cv::circle(visual, cv::Point(w, h), 1, cv::Scalar(0, 255, 0), -1);
			}
		}
	}

	cv::Mat result;
	cv::addWeighted(img, 1, visual, 0.2, 0, result);

	cv::imshow("result", result);
	cv::waitKey(0);

}

int main() {

	std::string ImgPath = "./img/*.bmp";

	// 用于存放从ImgPath中读取的图像名称
	std::vector<std::string> filepath;

	// 用于匹配所有图像的路径
	cv::glob(ImgPath, filepath);

	for (const auto& file_path : filepath) {
		// std::cout << file_path << std::endl;

		// 使用opencv读取图像
		cv::Mat img_data = cv::imread(file_path);
		
		// 使用bgr分割
		rgb_segmation(img_data);

		// 使用hsv分割
		hsv_segmation(img_data);
	}

}