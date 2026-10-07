// DetectedObject.h
#ifndef DETECTED_OBJECT_H
#define DETECTED_OBJECT_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct DetectedObject {
    int class_id;
    float confidence;
    std::string label;
    cv::Rect_<float> bbox;  // or cv::Rect2f depending on your OpenCV version
    cv::Rect box;            // integer rectangle coordinates
    
    // You might also need these methods:
    bool contains(const cv::Point2f& pt) const {
        return bbox.contains(pt);
    }
};

#endif // DETECTED_OBJECT_H
