#ifndef FRAME_H
#define FRAME_H

#include <vector>
#include <mutex>
#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"

#include <opencv2/opencv.hpp>
#include "DetectedObject.h"

namespace ORB_SLAM2
{

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;

class Frame
{
public:
    Frame();
    Frame(const Frame &frame);

    // Stereo

	Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp,
      ORBextractor* extractorLeft, ORBextractor* extractorRight,
      ORBVocabulary* voc,
      cv::Mat &K, cv::Mat &distCoef,
      const float &bf, const float &thDepth,
      const cv::Mat &exclude_mask);

    // RGB-D
    Frame(const cv::Mat &imGray, const cv::Mat &imDepth,
      const double &timeStamp,
      ORBextractor* extractor, ORBVocabulary* voc,
      cv::Mat &K, cv::Mat &distCoef,
      const float &bf, const float &thDepth,
      const cv::Mat &exclude_mask);
    // Monocular
    Frame(const cv::Mat &imGray, const double &timeStamp,
      ORBextractor* extractor, ORBVocabulary* voc,
      cv::Mat &K, cv::Mat &distCoef,
      const float &bf, const float &thDepth,
      const cv::Mat &exclude_mask, bool bIsWorkFrame = false);

    // ================= EXCLUDE MASK =================
    inline void SetExcludeMask(const cv::Mat &mask)
    {
        if (!mask.empty())
            mExcludeMask = mask.clone();
        else
            mExcludeMask.release();
    }

    inline const cv::Mat& GetExcludeMask() const
    {
        return mExcludeMask;
    }

    inline bool IsExcludedPixel(int x, int y) const
    {
        if (mExcludeMask.empty()) return false;
        if (x < 0 || y < 0 || x >= mExcludeMask.cols || y >= mExcludeMask.rows)
            return true;
        return mExcludeMask.at<uchar>(y, x) == 0;
    }

    // ORB
    void ExtractORB(int flag, const cv::Mat &im);
    void ComputeBoW();

    void SetPose(cv::Mat Tcw);
    void UpdatePoseMatrices();

    inline cv::Mat GetCameraCenter(){ return mOw.clone(); }
    inline cv::Mat GetRotationInverse(){ return mRwc.clone(); }

    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    std::vector<size_t> GetFeaturesInArea(
        const float &x, const float &y, const float &r,
        const int minLevel=-1, const int maxLevel=-1) const;

    void ComputeStereoMatches();
    void ComputeStereoFromRGBD(const cv::Mat &imDepth);
    cv::Mat UnprojectStereo(const int &i);

    // Objects (debug)
    void SetObjects(const std::vector<DetectedObject>& objects);
    const std::vector<DetectedObject>& GetObjects() const;

public:
    ORBVocabulary* mpORBvocabulary;
    ORBextractor* mpORBextractorLeft;
    ORBextractor* mpORBextractorRight;

    double mTimeStamp;

    cv::Mat mK;
    static float fx, fy, cx, cy, invfx, invfy;
    cv::Mat mDistCoef;

    float mbf;
    float mb;
    float mThDepth;

    int N;

    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
    std::vector<cv::KeyPoint> mvKeysUn;

    std::vector<float> mvuRight;
    std::vector<float> mvDepth;

    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    cv::Mat mDescriptors, mDescriptorsRight;

    std::vector<MapPoint*> mvpMapPoints;
    std::vector<bool> mvbOutlier;

    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    cv::Mat mTcw;
    static long unsigned int nNextId;
    // V1.6: nNextId is incremented by every Frame constructor call. Since the
    // OA-SLAM background LOST-recovery thread (Tracking::RecoveryThreadLoop)
    // now also constructs standalone Frame objects concurrently with the main
    // thread, this counter is no longer single-threaded — guard it explicitly
    // rather than relying on it never racing.
    static std::mutex mMutexNextId;
    long unsigned int mnId;

    KeyFrame* mpReferenceKF;

    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    std::vector<float> mvScaleFactors;
    std::vector<float> mvInvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    static float mnMinX, mnMaxX, mnMinY, mnMaxY;
    static bool mbInitialComputations;

    // 🔥 MASK LOẠI NGƯỜI & ẢNH XÁM CHO KEYFRAME PLACE DESCRIPTOR
    cv::Mat mExcludeMask;   // CV_8UC1, 0 = người, 255 = nền
    cv::Mat mImGray;        // Shallow reference to grayscale image for KeyFrame SIFT/Place extraction

private:

    void UndistortKeyPoints();
    void ComputeImageBounds(const cv::Mat &imLeft);
    void AssignFeaturesToGrid();

    cv::Mat mRcw, mtcw, mRwc, mOw;

    std::vector<DetectedObject> mObjects;
};

} // namespace ORB_SLAM2

#endif // FRAME_H