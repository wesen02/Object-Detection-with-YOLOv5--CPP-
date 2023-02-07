#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <fstream>
#include <cmath>
#include <random>
#include <chrono>

using namespace cv;
using namespace std;
using namespace dnn;

struct Detection
{
    int class_id = 0;
    string className;
    float confidence = 0.0;
    Scalar color;
    Rect box;
};

class Yolo
{
public:
    Yolo();

private:
    Net net;

    void loadClasses();
    void cvVersion();
    void loadSource();
    void detect(Mat &frame);
    string modelPath = "./source/model/yolov5s.onnx";
    string dataPath = "./source/data/test.mp4";
    // string dataPath = "./source/data/bus.jpg";
    string classPath = "./source/classes/classes.txt";
    string colorPath = "./source/classes/colors.txt";

    float modelConfidenceThreshold = 0.25;
    float modelScoreThreshold = 0.45;
    float modelNMSThreshold = 0.5;

    bool letterBoxForSquare = false;

    Size2f modelShape = Size(640, 480);

    Mat formatToSquare(const Mat &source);

    vector<string> classes;

    vector<Detection> detections{};
    vector<Detection> output;

    vector<Scalar> colors;

    void drawPred(int classId, float conf, int left, int top,
                  int right, int bottom, Mat &frame);
    void readColors();
};
