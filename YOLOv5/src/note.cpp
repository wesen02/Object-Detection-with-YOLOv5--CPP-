#include <main.h>

Yolo::Yolo()
{
    cvVersion();
    loadClasses();
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
    while (true)
    {
        video >> frame;
        if (frame.empty())
            break;
        namedWindow("Test", WINDOW_NORMAL);
        imshow("Test", frame);
        detect(frame);
        if (waitKey(30) == 27)
            break;
    }
}

void Yolo::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat &frame)
{
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 3);

    string label = format("%.2f", conf);
    label = this->classes[classId] + ":" + label;

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);

    putText(frame, label, Point(left, top), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
}

void Yolo::sigmoid(Mat *out, int length)
{
    float *pdata = (float *)(out->data);
    int i = 0;
    for (i = 0; i < length; i++)
    {
        pdata[i] = 1.0 / (1 + expf(-pdata[i]));
    }
}

void Yolo::detect(Mat &frame)
{
    Mat blob;

    blobFromImage(frame, blob, 1.0 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
    this->net.setInput(blob);
    vector<Mat> outs;
    this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

    // Generate proposals
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
    int n = 0, q = 0, i = 0, j = 0, nout = this->classes.size() + 5, c = 0;
    for (n = 0; n < 3; n++)
    {
        int num_grid_x = (int)(this->inpWidth / this->stride[n]);
        int num_grid_y = (int)(this->inpHeight / this->stride[n]);
        int area = num_grid_x * num_grid_y;

        this->sigmoid(&outs[n], 3 * nout * area);
        for (q = 0; q < 3; q++)
        {
            const float anchor_w = this->anchors[n][q * 2];
            const float anchor_h = this->anchors[n][q * 2 + 1];
            float *pdata = (float *)outs[n].data + q * nout * area;
            for (i = 0; i < num_grid_y; i++)
            {
                for (i = 0; i < num_grid_x; i++)
                {
                    float box_score = pdata[4 * area + i * num_grid_x + j];
                    if (box_score > this->objThreshold)
                    {
                        float max_class_score = 0, class_score = 0;
                        int max_class_id = 0;
                        for (c = 0; c < this->classes.size(); c++)
                        {
                            class_score = pdata[(c + 5) * area + i * num_grid_x + j];
                            if (class_score > max_class_score)
                            {
                                max_class_score = class_score;
                                max_class_id = c;
                            }
                        }

                        if (max_class_score > this->confThreshold)
                        {
                            float cx = (pdata[i * num_grid_x + j] * 2.f - 0.5f + j) * this->stride[n];
                            float cy = (pdata[area + i * num_grid_x + j] * 2.f - 0.5f + j) * this->stride[n];
                            float w = powf(pdata[2 * area + i * num_grid_x + j] * 2.f, 2.f) * anchor_w;
                            float h = powf(pdata[3 * area + i * num_grid_x + j] * 2.f, 2.f) * anchor_h;

                            int left = (cx - 0.5 * w) * ratiow;
                            int top = (cy - 0.5 * h) * ratioh;

                            classIds.push_back(max_class_id);
                            confidences.push_back(max_class_score);
                            boxes.push_back(Rect(left, top, (int)(w * ratiow), (int)(h * ratioh)));
                        }
                    }
                }
            }
        }
    }

    vector<int> indices;
    NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        this->drawPred(classIds[idx], confidences[idx], box.x, box.y,
                       box.x + box.width, box.y + box.height, frame);
    }
}

int main(int argc, char **argv)
{
    Yolo test;

    return 0;
}