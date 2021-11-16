/** FaceDetection.cpp **/
//��openCV sample file "object_detection.cpp"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <string>
using namespace cv::dnn;
using namespace cv;
using namespace std;

//�e�������e
const int inWidth = 1920;
const int inHeight = 1080;

int main(int argc, char** argv)
//�s���w��
try {
	cout << "��cv::dnn,cv::VideoCapture²�ƫD���n��,�i�����H�y,�����j�a����300x300,���ѪR�׼ҫ����Ѳv���\n"
		<< "�n��release, �[�Wcv�u�ƴ�֩���\n";
	cin.get();
	Net net = readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel");

	// Start streaming from Camera
	setUseOptimized(1);
	VideoCapture cap(0);
	cap.set(CAP_PROP_FRAME_WIDTH, inWidth);
	cap.set(CAP_PROP_FRAME_HEIGHT, inHeight);
	cap.set(CAP_PROP_AUTOFOCUS, 1);//�۰ʹ�J
	string window_name = "��ܿ���" + to_string(inWidth) + "x" + to_string(inHeight);
	namedWindow(window_name);

	while (getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0) {
		Mat color_mat;
		cap >> color_mat;

		//Convert Mat to batch of images
		Mat tmp;
		resize(color_mat, tmp, Size(300, 300));
		//��color_mat��tmp
		Mat inputBlob = blobFromImage(color_mat, 1.0, color_mat.size(), Scalar(104.0, 177.0, 123.0), false);
		net.setInput(inputBlob, "data"); //set the network input
		Mat detection = net.forward(); //compute output
		//Network produces output blob with a shape 1xlxNx7 where N is the max.number of detections
		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, (float*)detection.data);
		for (int i = 0; i <= 3; i++)
			cout << detection.size[1] << "\t";
		cout << endl;
		float confidenceThreshold = 0.5;
		Vec3b color[] = { Vec3b(255, 255, 0), Vec3b(0, 255, 0), Vec3b(0, 255, 255), Vec3b(255, 100, 255) };
		int num = 0;
		//Every detection is a vector of values
		// [batchId, classld, confidence, left, top, right, bottom]
		for (int i = 0; i < detectionMat.rows; i++) {
			putText(color_mat, "Max size of Detections: " + to_string(detection.size[2]),
				Size(30, 30), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255));
			float  confidence = detectionMat.at<float>(i, 2);

			if (confidence > confidenceThreshold) {
				int x0 = (int)(detectionMat.at<float>(i, 3) * color_mat.cols);
				int y0 = (int)(detectionMat.at<float>(i, 4) * color_mat.rows);
				int x1 = (int)(detectionMat.at<float>(i, 5) * color_mat.cols);
				int y1 = (int)(detectionMat.at<float>(i, 6) * color_mat.rows);
				Rect object(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
				cout << object;
				string ss = "#" + to_string(num++) + " Prob=" + to_string(confidence);
				cout << ss << endl;

				rectangle(color_mat, object, color[num % 4], 2);
				int baseLine = 0;
				Size labelSize = getTextSize(ss, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

				Point LB(x0, y1 + labelSize.height);
				rectangle(color_mat, Rect(Point(LB.x, LB.y - labelSize.height),
					Size(labelSize.width, labelSize.height + baseLine)), color[num % 4], -1);
				putText(color_mat, ss, LB, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
			}
		}
		imshow(window_name, color_mat);

		int k = waitKey(1);
		if (k == 27) break;
		else if (k == (int)'+') imwrite("photo.png", color_mat);
	}
	destroyAllWindows();
	system("Pause");
	return 0;
}
catch (exception& e) {
	cerr << e.what();
	return 1;
}