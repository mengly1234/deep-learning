#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <filesystem>

static const int W = 640;
static const int H = 640;
static const int Box_sum = 25200; // 总的框数
static const int Box_data = 85; // 每个框的位置+置信度+类别
static const float CONF = 0.25;
static const float IOU = 0.45;
static const std::vector<std::string> cls_name = 
{ "person", "bicycle", "car", "motorcycle", "airplane", "bus","train","truck","boat","traffic light",
"fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant",
"bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
"kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",
"knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
"chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
"cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
"hair drier","toothbrush" };

void preprocess(cv::Mat& img, cv::Mat& img_input) {

	cv::resize(img, img, cv::Size(W, H));
	img_input = cv::dnn::blobFromImage(img, 1.0 / 255.0, cv::Size(W, H), cv::Scalar(), true);
}

void postprecess_conf(cv::Mat& result, std::vector<std::vector<float>>& out_conf_list, float conf = CONF) {
	
	float* data = (float*)result.data;
	for (int i = 0; i < Box_sum; i++) {
		if (data[4] > CONF) {
			std::vector<float> out_conf_data;
			for (int j = 0; j < Box_data; j++) {
				out_conf_data.push_back(data[j]);
			}
			out_conf_list.push_back(out_conf_data);
		}
		data = data + Box_data;
	}

	for (auto i = 0; i < out_conf_list.size(); i++) {
		// 算出该目标的类别
		out_conf_list[i][5] = std::max_element(out_conf_list[i].cbegin() + 5, out_conf_list[i].cend()) - (out_conf_list[i].cbegin() + 5);
		out_conf_list[i].resize(6);

		// x,y,w,h->x1,y1,x2,y2
		float x = out_conf_list[i][0];
		float y = out_conf_list[i][1];
		float w = out_conf_list[i][2];
		float h = out_conf_list[i][3];
		out_conf_list[i][0] = x - w / 2.0;
		out_conf_list[i][1] = y - h / 2.0;
		out_conf_list[i][2] = x + w / 2.0;
		out_conf_list[i][3] = y + h / 2.0;
	}
}

void classes_split(std::vector<std::vector<float>>& result_conf, std::vector<std::vector<std::vector<float>>>& result_cls) {
	;
	std::vector<int> cls_id;
	for (int i = 0; i < result_conf.size(); i++) {
		if (std::find(cls_id.begin(), cls_id.end(), (int)result_conf[i][5]) == cls_id.end()) {
			cls_id.push_back((int)result_conf[i][5]);
			std::vector<std::vector<float>> result_cls_data;
			result_cls.push_back(result_cls_data);
		}
		result_cls[std::find(cls_id.begin(), cls_id.end(), (int)result_conf[i][5]) - cls_id.begin()].push_back(result_conf[i]);
	}
}

void nms_process(std::vector<std::vector<float>>& result_cls) {

	int count = 0;
	std::vector<std::vector<float>> nms_result;

	while (count < result_cls.size()) {

		nms_result.clear();

		std::sort(result_cls.begin(), result_cls.end(), [](std::vector<float> x, std::vector<float> y)
			{ return x[4] > y[4]; });

		float x1 = 0, y1 = 0, x2 = 0, y2 = 0;

		for (int i = 0; i < result_cls.size(); i++) {
			if (i < count) {
				nms_result.push_back(result_cls[i]);
				continue;
			}
			if (i == count) {
				x1 = result_cls[i][0];
				y1 = result_cls[i][1];
				x2 = result_cls[i][2];
				y2 = result_cls[i][3];
				nms_result.push_back(result_cls[i]);
				continue;
			}
			if (result_cls[i][0] > x2 or result_cls[i][1] > y2 or result_cls[i][2] < x1 or result_cls[i][3] < y1) {
				nms_result.push_back(result_cls[i]);
			}
			else
			{
				float cover_x1 = std::max(x1, result_cls[i][0]);
				float cover_y1 = std::max(y1, result_cls[i][1]);
				float cover_x2 = std::min(x2, result_cls[i][2]);
				float cover_y2 = std::min(y2, result_cls[i][3]);

				float cover_s = (cover_x2 - cover_x1) * (cover_y2 - cover_y1);
				float sum_s = (x2 - x1) * (y2 - y1) + (result_cls[i][2] - result_cls[i][0]) * (result_cls[i][3] - result_cls[i][1]) - cover_s;

				if (cover_s / sum_s < IOU) {
					nms_result.push_back(result_cls[i]);
				}
			}
		}
		result_cls = nms_result;
		count++;
	}
}

void draw_box(cv::Mat& img, std::vector<std::vector<float>> box) {
	for (int i = 0; i < box.size(); i++) {
		cv::rectangle(img, cv::Point(box[i][0], box[i][1]), cv::Point(box[i][2], box[i][3]), cv::Scalar(0, 255, 0));
		
		//std::string label = std::to_string((int)box[i][5]);
		//cv::putText(img, label, cv::Point(box[i][0], box[i][1]), 1, 2, cv::Scalar(0, 255, 0), 1);
		
		int cls_index = (int)box[i][5];
		std::string name = cls_name[cls_index];
		cv::putText(img, name, cv::Point(box[i][0], box[i][1]), 1, 2, cv::Scalar(0, 255, 0), 1);
	}
}

void detect(cv::Mat& img) {
	cv::Mat img_input;
	preprocess(img, img_input);

	cv::dnn::Net net = cv::dnn::readNetFromONNX("yolov5s.onnx");

	net.setInput(img_input);

	std::vector<cv::Mat> img_output;
	std::vector<std::string> out_name = { "output0" };

	net.forward(img_output, out_name);

	cv::Mat result = img_output[0];

	std::vector<std::vector<float>> result_conf;

	postprecess_conf(result, result_conf);

	std::vector<std::vector<std::vector<float>>> result_cls;

	classes_split(result_conf, result_cls);

	for (int i = 0; i < result_cls.size(); i++) {
		nms_process(result_cls[i]);
		draw_box(img, result_cls[i]);
	}
	cv::imshow("test", img);
	cv::waitKey(100);
}

int main() {

	std::string folder = "D:/visual studio/Project/yolov5_onnx_detect_dnn/yolov5_onnx_detect/images";

	std::vector<cv::String> filename;
	cv::glob(folder, filename);
	
	for (int i = 0; i < filename.size(); i++) {
		cv::Mat img = cv::imread(filename[i]);
		detect(img);
	}
	return 0;
}