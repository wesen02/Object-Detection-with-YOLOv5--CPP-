#include <main.h>

Yolo::Yolo()
{
    cvVersion();
    loadClasses();
    readColors();
    loadSource();
}

void Yolo::cvVersion()
{
    cout << "OpenCV version : " << CV_VERSION << endl;

    try
    {
        this->net = readNetFromONNX(modelPath);
        // cuda
        if (cuda::getCudaEnabledDeviceCount() > 0)
        {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
            cout << "This computer is using CUDA"
                 << "\n";
        }
        // cpu
        else
        {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            cout << "This computer is using CPU"
                 << "\n";
        }
    }
    catch (const Exception &e)
    {
        cerr << "Error Loading the model " << e.what()
             << "\n";
        return;
    }
    cout << "weights loaded successfully"
         << "\n";
}

void Yolo::loadClasses()
{
    ifstream inputFile(classPath);
    if (inputFile.is_open())
    {
        cout << "Classes file opened"
             << "\n";
        string classLine;
        while (std::getline(inputFile, classLine))
            classes.push_back(classLine);
        inputFile.close();
    }
}

void Yolo::loadSource()
{
    // Load video
    cv::VideoCapture video(dataPath);
    if (!video.isOpened())
    {
        cout << "Error opening video file"
             << "\n";
        return;
    }
    Mat frame;

    std::vector<std::string> parts;
    std::string part;
    for (auto c : dataPath)
    {
        if (c == '.')
        {
            parts.push_back(part);
            part.clear();
        }
        else
        {
            part += c;
        }
    }
    parts.push_back(part);

    string extension = parts.back();

    if (extension == "jpg" || extension == "png")
    {
        cout << "File is a .jpg image."
             << "\n";
        video >> frame;
        if (frame.empty())
            return;
        namedWindow("JPG Image", WINDOW_NORMAL);
        detect(frame);
        imshow("JPG Image", frame);
        waitKey(0);
    }
    else if (extension == "mp4")
    {
        cout << "File is a .mp4 video."
             << "\n";
        clock_t start;
        clock_t end;
        double ms, fpsLive, seconds;
        while (true)
        {
            start = clock();
            video >> frame;
            if (frame.empty())
                break;
            namedWindow("Video", WINDOW_NORMAL);
            detect(frame);

            end = clock();

            seconds = (double(end) - double(start)) / double(CLOCKS_PER_SEC);
            fpsLive = 1.0 / double(seconds);
            // cout << "FPS: " << fpsLive << setprecision(2) << "\n";

            putText(frame, "FPS: " + to_string(fpsLive), {50, 100},
                    FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5);
            imshow("Video", frame);
            if (waitKey(30) == 27)
                break;
        }
    }
    else
    {
        cout << "Unknown format"
             << "\n";
    }
}

void Yolo::detect(Mat &frame)
{
    Mat modelInput = frame;
    if (letterBoxForSquare && modelShape.width == modelShape.height)
        modelInput = formatToSquare(modelInput);

    Mat blob;
    blobFromImage(modelInput, blob, 1.0 / 255.0, modelShape,
                  cv::Scalar(), true, false); // Error occur here because of the size of the resolution
    net.setInput(blob);

    vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    int rows = outputs[0].size[1];
    int dimensions = outputs[0].size[2];

    bool yolov8 = false;
    if (dimensions % 2 == 0)
    {
        yolov8 = true;
        rows = outputs[0].size[2];
        dimensions = outputs[0].size[1];

        outputs[0] = outputs[0].reshape(1, dimensions);
        transpose(outputs[0], outputs[0]);
    }
    float *data = (float *)outputs[0].data;

    float x_factor = modelInput.cols / modelShape.width;
    float y_factor = modelInput.rows / modelShape.height;

    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];

        if (confidence >= modelConfidenceThreshold)
        {
            float *classes_scores = yolov8 ? data + 4 : data + 5;

            Mat scores(1, classes.size(), CV_32FC1, classes_scores);
            Point class_id;
            double max_class_score;

            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            if (max_class_score > modelScoreThreshold)
            {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);

                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(Rect(left, top, width, height));
            }
        }

        data += dimensions;
    }

    vector<int> nms_result;
    NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);

    // vector<Detection> detections{};
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {

        int idx = nms_result[i];

        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];

        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<int> dis(0, 255);
        result.color = cv::Scalar(dis(gen),
                                  dis(gen),
                                  dis(gen));

        result.className = classes[result.class_id];
        result.box = boxes[idx];

        detections.push_back(result);

        // cout << result.class_id;

        Rect box = boxes[idx];
        this->drawPred(class_ids[idx], confidences[idx], box.x, box.y,
                       box.x + box.width, box.y + box.height, frame, result.color);
    }
}

void Yolo::drawPred(int classId, float conf, int left, int top, int right,
                    int bottom, Mat &frame, Scalar color)
{
    rectangle(frame, Point(left, top), Point(right, bottom), colors[classId], 3);

    string label = format("%.2f", conf);
    label = this->classes[classId] + ":" + label;

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);

    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 2, colors[classId], 5);
}

Mat Yolo::formatToSquare(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    Mat result = Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(Rect(0, 0, col, row)));
    return result;
}

void Yolo::readColors()
{
    ifstream file(colorPath);
    if (!file.is_open())
    {
        cout << "Unable to open file: " << colorPath << "\n";
        return;
    }

    int r, g, b;
    while (file >> r >> g >> b)
    {
        colors.push_back(Scalar(b, g, r));
    }
    file.close();
}

int main(int argc, char **argv)
{
    Yolo test;

    return 0;
}