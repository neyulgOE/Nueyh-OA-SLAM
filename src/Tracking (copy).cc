/**This file is part of OA-SLAM.

Copyright (C) 2022 Matthieu Zins <matthieu.zins@inria.fr>
(Inria, LORIA, Université de Lorraine)
OA-SLAM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

OA-SLAM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with OA-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

/**This file is part of ORB-SLAM2.

Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
For more information see <https://github.com/raulmur/ORB_SLAM2>

ORB-SLAM2 is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ORB-SLAM2 is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Tracking.h"
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"
#include "Optimizer.h"
#include "PnPsolver.h"
#include "Ellipsoid.h"
#include "Ellipse.h"
#include "MapObject.h"
#include "LocalObjectMapping.h"
#include "Localization.h"
#include "System.h"
#include "Utils.h"
#include "ARViewer.h"
#include <iostream>
#include <mutex>
#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <unistd.h>
#include <Eigen/Dense>
#include <unistd.h>
#include <dlib/optimization/max_cost_assignment.h>
#include <chrono>
#include <thread>
#include <omp.h>
//#define PERSON_CLASS_ID 110

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

bool is_inside_mask(float x, float y, const cv::Mat& mask_no_person) {
    int ix = static_cast<int>(std::round(x));
    int iy = static_cast<int>(std::round(y));
    if (ix < 0 || iy < 0 || ix >= mask_no_person.cols || iy >= mask_no_person.rows)
        return false;
    // Chỉ chấp nhận điểm nếu pixel tại đó > 0 (tức không nằm trên người)
    return mask_no_person.at<uchar>(iy, ix) > 0;
}

namespace ORB_SLAM2
{
Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0),
    mbUseMask(true)
{
    // Load camera parameters from settings file
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    mK = K;
    K_ = cvToEigenMatrix<double, float, 3, 3>(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    mDistCoef = DistCoef;

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    // Mono: mMinFrames=2 tránh tạo KF liên tiếp từng frame, nhưng không block quá lâu.
    // bEnoughMotion guard trong NeedNewKeyFrame() sẽ chặn spam khi camera đứng yên.
    // fps/2 = 7 frame cũ là quá cao ở 15fps: camera quay 90° xong trong 1.5s mà không
    // có anchor nào mới → khi nhìn chỗ mới hoàn toàn mất track.
    mMinFrames = (sensor == System::MONOCULAR) ? 2 : 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;
    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLocalObjectMapper(LocalObjectMapping *obj_mapper)
{
    local_object_mapper_ = obj_mapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

void Tracking::SetARViewer(ARViewer *pARViewer)
{
    mpARViewer = pARViewer;
}

cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft,
                                  const cv::Mat &imRectRight,
                                  const double &timestamp)
{
    // ==== 1. Chuyển ảnh sang grayscale ====
    mImGray = imRectLeft.clone();
    cv::Mat imGrayRight = imRectRight.clone();

    if (mImGray.channels() == 3)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
        }
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
        }
    }
    
    // TẠO MASK LOẠI TRỪ NGƯỜI CÓ KIỂM TRA BIẾN MBUSEMASK
    cv::Mat exclude_person_mask;

    if (mbUseMask) 
    {
        exclude_person_mask = cv::Mat::ones(mImGray.size(), CV_8UC1) * 255;

        for (const auto& det : current_frame_good_detections_)
        {
            if (det->category_id != PERSON_CLASS_ID) continue;

            if (!det->mask_no_person.empty()) {
                // JSON convention: mask_no_person = 255 tại người → setTo(0) xóa đúng vùng người
                exclude_person_mask.setTo(0, det->mask_no_person);
            }
            else if (!det->mask_object.empty()) {
                exclude_person_mask.setTo(0, det->mask_object);
            }
            else {
                // Không có mask polygon → dùng đúng bbox, không mở rộng để tránh xóa nền
                float xmin = det->bbox[0], ymin = det->bbox[1];
                float xmax = det->bbox[2], ymax = det->bbox[3];
                cv::Rect exact(
                    std::max(0, int(xmin)),
                    std::max(0, int(ymin)),
                    std::min(int(xmax - xmin), mImGray.cols - std::max(0, int(xmin))),
                    std::min(int(ymax - ymin), mImGray.rows - std::max(0, int(ymin)))
                );
                exclude_person_mask(exact).setTo(0);
            }
        }
        static cv::Mat kernel_stereo = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        cv::erode(exclude_person_mask, exclude_person_mask, kernel_stereo, cv::Point(-1,-1), 1);
    }
    else 
    {
        // Khi tắt mask trên Viewer, truyền mask rỗng để hệ thống không lọc gì cả
        exclude_person_mask = cv::Mat();
    }

    // ==== 4. TẠO FRAME (Giữ nguyên) ====
    mCurrentFrame = Frame(mImGray, imGrayRight, timestamp,
                          mpORBextractorLeft, mpORBextractorRight,
                          mpORBVocabulary, mK, mDistCoef,
                          mbf, mThDepth,
                          exclude_person_mask);

    // ==== 5. TRACKING (Giữ nguyên) ====
    Track();

    return mCurrentFrame.mTcw;
}

cv::Mat Tracking::GrabImageStereo_object(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp, const std::vector<Detection::Ptr>& detections, bool force_relocalize)
{
    current_frame_idx_ = (current_frame_idx_ + 1) % (std::numeric_limits<size_t>::max()-1);
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;   
    im_rgb_ = imRectLeft;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    ///////////////////////////////////////
    /*
    cv::Mat person_mask = cv::Mat::zeros(mImGray.size(), CV_8UC1);
    for (const auto& det : current_frame_good_detections_) {
        if (det->category_id == PERSON_CLASS_ID) {
            cv::bitwise_or(person_mask, det->mask, person_mask);
        }
    }
    cv::Mat exclude_person_mask = cv::Mat::ones(mImGray.size(), CV_8UC1) * 255;
    exclude_person_mask.setTo(0, person_mask);
    cv::imshow("personMask", person_mask);
    */
    current_frame_detections_ = detections;
    current_frame_good_detections_.clear();
    for (auto det : current_frame_detections_) {
        if (det->score > 0.5) {
            // if (det->category_id != 73 && det->score > 0.5 ||  det->score > 0.7) { // for table scene to ignore book on the nappe
            current_frame_good_detections_.push_back(det);
        }
    }

    cv::Mat exclude_person_mask;
    if (mbUseMask) {
        exclude_person_mask = cv::Mat(mImGray.size(), CV_8UC1, cv::Scalar(255));
        for (const auto& det : current_frame_good_detections_) {
            if (det->category_id != PERSON_CLASS_ID) continue;
            if (!det->mask_no_person.empty()) {
                exclude_person_mask.setTo(0, det->mask_no_person);
            } else if (!det->mask_object.empty()) {
                exclude_person_mask.setTo(0, det->mask_object);
            } else {
                float xmin = det->bbox[0], ymin = det->bbox[1];
                float xmax = det->bbox[2], ymax = det->bbox[3];
                cv::Rect exact(
                    std::max(0, int(xmin)),
                    std::max(0, int(ymin)),
                    std::min(int(xmax - xmin), mImGray.cols - std::max(0, int(xmin))),
                    std::min(int(ymax - ymin), mImGray.rows - std::max(0, int(ymin)))
                );
                exclude_person_mask(exact).setTo(0);
            }
        }
        static cv::Mat kernel_stereo_obj = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        cv::erode(exclude_person_mask, exclude_person_mask, kernel_stereo_obj, cv::Point(-1,-1), 1);
    }
    //cv::imshow("Mask-feed-slam", exclude_person_mask);
    ////////////////////////
    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,exclude_person_mask);
    std::vector<cv::KeyPoint> filteredKeys;

    if (force_relocalize)
    {
        auto t1 = high_resolution_clock::now();

        bool bOK = false;
        std::cout << "[FORCE RELOCALIZE] Frame " << current_frame_idx_ << std::endl;

        // ================== ƯU TIÊN RELocalization BẰNG OBJECTS ==================
        int relocMode = mpSystem->GetRelocalizationMode();

        if (relocMode == RELOC_OBJECTS || relocMode == RELOC_OBJECTS_POINTS)
        {
            std::cout << "→ Trying RelocalizationFromObjects first...\n";
            bOK = RelocalizationFromObjects(false);   // false = chỉ dùng objects
        }

        if (!bOK)
        {
            std::cout << "→ Objects failed, trying Relocalization with points...\n";
            bOK = Relocalization();
        }

        if (!bOK && relocMode == RELOC_OBJECTS_POINTS)
        {
            std::cout << "→ Trying combined Objects + Points...\n";
            bOK = RelocalizationFromObjects(true);   // true = kết hợp points
        }

        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms_double = t2 - t1;

        mpFrameDrawer->Update(this);
        if (bOK)
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mpSystem->relocalization_duration = ms_double.count();
        mpSystem->relocalization_status = bOK;

        if (bOK)
            std::cout << "✅ Force Relocalization SUCCESS!\n";
        else
            std::cout << "❌ Force Relocalization FAILED\n";
    }
    else
    {
        Track();
        
        /////////////////////////////////// Objects Tracking ///////////////////////////////////
        // Update mean depth
        if (mState == Tracking::OK) {
            Matrix34d Rt = cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw);
            double z_mean = 0.0;
            unsigned int z_nb = 0;
            for(size_t i = 0; i < mCurrentFrame.mvpMapPoints.size(); i++) {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP && !mCurrentFrame.mvbOutlier[i]) {
                    cv::Mat mp =  pMP->GetWorldPos();
                    Eigen::Vector4d p(mp.at<float>(0), mp.at<float>(1), mp.at<float>(2), 1.0);
                    Eigen::Vector3d p_cam = Rt * p;
                    z_mean += p_cam[2];
                    z_nb++;
                }
            }
            z_mean /= z_nb;
            // std::cout << "Mean depth = " << z_mean << "\n";
            current_mean_depth_ = z_mean;
        }
        
        std::cout << "Frame " << current_frame_idx_ << " ===========\n";
        // std::cout << "Created new KF: " << createdNewKeyFrame_ << "\n";
        std::cout << "Nb Object Tracks: " << objectTracks_.size() << "\n";
        std::cout << "Nb Map Objects  : " << mpMap->GetNumberMapObjects() << "\n";
        
        double MIN_2D_IOU_THRESH = 0.2;
        double MIN_3D_IOU_THRESH = 0.3;
        int TIME_DIFF_THRESH = 30;

        BBox2 img_bbox(0, 0, imRectLeft.cols, imRectLeft.rows);
        
        if (mState == Tracking::OK) {

            // Keep only detections with a certain score
            if (current_frame_good_detections_.size() != 0) {

                KeyFrame *kf = mpLastKeyFrame;
                if (!createdNewKeyFrame_)
                    kf = nullptr;
                // pre-compute all the projections of all ellipsoids which already reconstructed
                Matrix34d Rt = cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw);
                Matrix34d P;
                P = K_ * Rt;
                std::unordered_map<ObjectTrack::Ptr, BBox2> proj_bboxes;
                for (auto tr: objectTracks_) {
                    if (tr->GetStatus() == ObjectTrackStatus::INITIALIZED ||
                        tr->GetStatus() == ObjectTrackStatus::IN_MAP) {
                        MapObject* obj = tr->GetMapObject();
                        Eigen::Vector3d c = obj->GetEllipsoid().GetCenter();
                        double z = Rt.row(2).dot(c.homogeneous());
                        auto ell = obj->GetEllipsoid().project(P);
                        BBox2 bb = ell.ComputeBbox();
                        if (bboxes_intersection(bb, img_bbox) < 0.3 * bbox_area(bb)) {
                            continue;
                        }
                        proj_bboxes[tr] = ell.ComputeBbox();

                        // Check occlusions and keep only the nearest
                        std::unordered_set<ObjectTrack::Ptr> hidden;
                        for (auto it : proj_bboxes) {
                            if (it.first != tr && bboxes_iou(it.second, bb) > 0.9) {
                                Eigen::Vector3d c2 = it.first->GetMapObject()->GetEllipsoid().GetCenter();
                                double z2 = Rt.row(2).dot(c2.homogeneous());
                                if (z < z2) {
                                    // remove z2
                                    hidden.insert(it.first);
                                } else {
                                    // remove z
                                    hidden.insert(tr);
                                }
                                break;
                            }
                        }
                        for (auto hid : hidden) {
                            proj_bboxes.erase(hid);
                        }
                    }
                }

                // find possible tracks
                std::vector<ObjectTrack::Ptr> possible_tracks;
                for (auto tr : objectTracks_) {
                    auto bb = tr->GetLastBbox();
                    if (tr->GetLastObsFrameId() + 60 >= current_frame_idx_ &&
                        bboxes_intersection(bb, img_bbox) >= 0.3 * bbox_area(bb)) {
                        possible_tracks.push_back(tr);
                    } else if (proj_bboxes.find(tr) != proj_bboxes.end()) {
                        possible_tracks.push_back(tr);
                    }
                }

                // ------------------- Phần tối ưu: song song + ROI -------------------
                std::vector<std::unordered_set<MapPoint*>> assoc_map_points(current_frame_good_detections_.size());
                std::vector<Eigen::Vector3d> mean_coords(current_frame_good_detections_.size(), Eigen::Vector3d::Zero());
                std::vector<int> point_counts(current_frame_good_detections_.size(), 0);

                // Chuyển Tcw chỉ MỘT lần
                Eigen::Matrix4d Tcw_eigen = cvToEigenMatrix<double, float, 4, 4>(mCurrentFrame.mTcw);

                // Parallel processing cho từng detection
                #pragma omp parallel for schedule(dynamic)
                for (size_t i = 0; i < current_frame_good_detections_.size(); ++i) {
                    const auto& det = current_frame_good_detections_[i];
                    if (det->mask_object.empty()) continue;

                    // Tính ROI mở rộng
                    cv::Rect roi = cv::boundingRect(det->mask_object);
                    roi.x      = std::max(0, roi.x - 15);
                    roi.y      = std::max(0, roi.y - 15);
                    roi.width  = std::min(static_cast<int>(mImGray.cols) - roi.x, roi.width + 30);
                    roi.height = std::min(static_cast<int>(mImGray.rows) - roi.y, roi.height + 30);

                    Eigen::Vector3d local_sum = Eigen::Vector3d::Zero();
                    int local_count = 0;
                    std::unordered_set<MapPoint*> local_assoc;

                    for (size_t j = 0; j < mCurrentFrame.mvKeysUn.size(); ++j) {
                        if (!mCurrentFrame.mvpMapPoints[j]) continue;

                        const cv::Point2f& pt = mCurrentFrame.mvKeysUn[j].pt;

                        // Skip nếu ngoài ROI
                        if (pt.x < roi.x || pt.x >= roi.x + roi.width ||
                            pt.y < roi.y || pt.y >= roi.y + roi.height) {
                            continue;
                        }

                        if (is_inside_mask(pt.x, pt.y, det->mask_object)) {
                            MapPoint* mp = mCurrentFrame.mvpMapPoints[j];
                            local_assoc.insert(mp);

                            cv::Mat pos_w = mp->GetWorldPos();
                            Eigen::Vector4d Pw(pos_w.at<float>(0), pos_w.at<float>(1), pos_w.at<float>(2), 1.0);
                            Eigen::Vector3d Pc = (Tcw_eigen * Pw).head<3>();
                            local_sum += Pc;
                            ++local_count;
                        }
                    }

                    assoc_map_points[i] = std::move(local_assoc);
                    if (local_count > 0) {
                        mean_coords[i] = local_sum / local_count;
                    }
                    point_counts[i] = local_count;
                }

                // Gán mean_coor và in log có kiểm soát
                for (size_t i = 0; i < current_frame_good_detections_.size(); ++i) {
                    current_frame_good_detections_[i]->mean_coor = mean_coords[i];

                    // Chỉ in log mỗi 5 frame để tránh spam console
                    if (current_frame_idx_ % 5 == 0) {
                        std::cout << "[Frame " << current_frame_idx_ << "] "
                                  << "[Detection " << i << "] "
                                  << "points: " << point_counts[i] << " | "
                                  << "mean_coor (camera): " << mean_coords[i].transpose() << std::endl;

                        double dist = mean_coords[i].norm();
                        std::cout << "  → Distance: " << dist << " meters" << std::endl;
                    }
                }
                /////////////////////////////////////

                // Try to match detections to existing object track based on the associated map points
                int THRESHOLD_NB_MATCH = 10;
                std::vector<int> matched_by_points(current_frame_good_detections_.size(), -1);
                std::vector<std::vector<size_t>> nb_matched_points(current_frame_good_detections_.size(), std::vector<size_t>());
                for (size_t i = 0; i < current_frame_good_detections_.size(); ++i) {
                    int det_cat = current_frame_good_detections_[i]->category_id;
                    size_t max_nb_matches = 0;
                    size_t best_matched_track = 0;
                    for (size_t j = 0; j < possible_tracks.size(); ++j) {
                        auto tr_map_points = possible_tracks[j]->GetAssociatedMapPoints();
                        size_t n = count_set_map_intersection(assoc_map_points[i], tr_map_points);
                        if (n > max_nb_matches) {
                            max_nb_matches = n;
                            best_matched_track = j;
                        }
                        if ( det_cat != (int)(possible_tracks[j]->GetCategoryId()) )
                            n = 0;
                        nb_matched_points[i].push_back(n);
                    }

                    if ( ((int)max_nb_matches) > ((int)THRESHOLD_NB_MATCH) &&
                        current_frame_good_detections_[i]->category_id == possible_tracks[best_matched_track]->GetCategoryId()) {
                        matched_by_points[i] = best_matched_track;
                    }
                }
                int m = std::max(possible_tracks.size(), current_frame_good_detections_.size());
                dlib::matrix<long> cost = dlib::zeros_matrix<long>(m, m);
                std::vector<long> assignment(m, std::numeric_limits<long>::max()); // Important to have it in 'long', max_int is used to force assignment of tracks already matched using points
                if (current_frame_good_detections_.size() > 0)
                {
                    // std::cout << "Hungarian algorithm size " << m << "\n";
                    for (size_t di = 0; di < current_frame_good_detections_.size(); ++di) {
                        auto det = current_frame_good_detections_[di];

                        for (size_t ti = 0; ti < possible_tracks.size(); ++ti) {
                            auto tr = possible_tracks[ti];
                            if (tr->GetCategoryId() == det->category_id) {
                                double iou_2d = 0;
                                double iou_3d = 0;

                                if (tr->GetLastObsFrameId() + TIME_DIFF_THRESH >= current_frame_idx_)
                                    iou_2d = bboxes_iou(tr->GetLastBbox(), det->bbox);

                                if (proj_bboxes.find(tr) != proj_bboxes.end())
                                    iou_3d = bboxes_iou(proj_bboxes[tr], det->bbox);

                                if (iou_2d < MIN_2D_IOU_THRESH) iou_2d = 0;
                                if (iou_3d < MIN_3D_IOU_THRESH) iou_3d = 0;

                                // std::cout << "2D: " << iou_2d << "\n";
                                // std::cout << "3D: " << iou_3d << "\n";
                                cost(di, ti) = std::max(iou_2d, iou_3d) * 1000;
                            }
                        }

                        if (matched_by_points[di] != -1) {
                            cost(di, matched_by_points[di]) = std::numeric_limits<int>::max();
                        }
                    }

                    // for (size_t i = 0; i < current_frame_good_detections_.size(); ++i) {
                    //     for (size_t j = 0; j < possible_tracks.size(); ++j) {
                    //         // std::cout << i << " " << j << " " << nb_matched_points[i][j] << "\n";
                    //         cost(i, j) += nb_matched_points[i][j] * 1000;
                    //     }
                    // }

                    assignment = dlib::max_cost_assignment(cost); // solve
                }
                std::vector<ObjectTrack::Ptr> new_tracks;
                for (size_t di = 0; di < current_frame_good_detections_.size(); ++di) {
                    auto det = current_frame_good_detections_[di];
                    auto assigned_track_idx = assignment[di];
                    if (assigned_track_idx >= static_cast<long>(possible_tracks.size()) || cost(di, assigned_track_idx) == 0) {
                        // assigned to non-existing => means not assigned
                        auto tr = ObjectTrack::CreateNewObjectTrack(det->category_id, det->bbox, det->score, Rt,
                                                                    current_frame_idx_, this, kf);
                        // std::cout << "create new track " << tr->GetId() << "\n";
                        new_tracks.push_back(tr);
                    } else {
                        ObjectTrack::Ptr associated_track = possible_tracks[assigned_track_idx];
                        associated_track->AddDetection(det->bbox, det->score, Rt, current_frame_idx_, kf);
                        if (kf && associated_track->GetStatus() == ObjectTrackStatus::IN_MAP) {
                            // std::cout << "Add modified objects" << std::endl;
                            if (local_object_mapper_)
                                local_object_mapper_->InsertModifiedObject(associated_track->GetMapObject());
                        }
                    }
                }

                for (auto tr : new_tracks)
                    objectTracks_.push_back(tr);
                if (!mbOnlyTracking) {
                    for (auto& tr : objectTracks_) {
                        if (tr->GetLastObsFrameId() == current_frame_idx_) {
                            // Try reconstruct from points
                            if ((tr->GetNbObservations() > 10 && tr->GetStatus() == ObjectTrackStatus::ONLY_2D) ||
                                (tr->GetNbObservations() % 2 == 0 && tr->GetStatus() == ObjectTrackStatus::INITIALIZED)) {
                                // tr->ReconstructFromSamplesEllipsoid();
                                // tr->ReconstructFromSamplesCenter();

                            bool status_rec = tr->ReconstructFromCenter(); // try to reconstruct and change status to INITIALIZED if success
                            // tr->ReconstructFromLandmarks(mpMap);
                            // tr->ReconstructCrocco(false); // not working
                            if (status_rec)
                                tr->OptimizeReconstruction(mpMap);
                            }
                        }

                        // Try to optimize objects and insert in the map
                        if (tr->GetNbObservations() >= 20 && tr->GetStatus() == ObjectTrackStatus::INITIALIZED) {
                            tr->OptimizeReconstruction(mpMap);
                            // std::cout << "First opimitzation done.\n";
                            auto checked = tr->CheckReprojectionIoU(0.3);
                            // std::cout << "Check reprojection " << checked << ".\n";
                            if (checked) {
                                // Add object to map
                                tr->InsertInMap(mpMap);
                                // Add object in the local object mapping thread to run a fusion checking
                                if (local_object_mapper_)
                                    local_object_mapper_->InsertModifiedObject(tr->GetMapObject());
                            } else {
                                tr->SetIsBad(); // or only reset to ONLY_2D ?
                            }
                        }
                    }
                }
            }

            if (!mbOnlyTracking) {
                // Remove objects that are not tracked anymore and not initialized or in the map
                for (ObjectTrack::Ptr tr : objectTracks_) {
                    if (static_cast<int>(tr->GetLastObsFrameId()) < static_cast<int>(current_frame_idx_) - TIME_DIFF_THRESH
                        && tr->GetStatus() != ObjectTrackStatus::IN_MAP) {
                            tr->SetIsBad();
                        }
                }
            }
        }

        // std::cout << "Object Tracks: " << objectTracks_.size() << "\n";
        mpFrameDrawer->Update(this);
        
        
    }

    // ── SEED LK BRIDGE cho GrabImageStereo_object ───────────────────────────
    mPrevGray = mImGray;
    if (mState == OK)
    {
        mvPrevPts.clear();
        mvPrevMPs.clear();
        for (int i = 0; i < mCurrentFrame.N; ++i)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if (pMP && !mCurrentFrame.mvbOutlier[i] && !pMP->isBad()
                && pMP->Observations() >= 2)
            {
                mvPrevPts.push_back(mCurrentFrame.mvKeysUn[i].pt);
                mvPrevMPs.push_back(pMP);
            }
        }
    }
    // ─────────────────────────────────────────────────────────────────────────

    return mCurrentFrame.mTcw;
}

cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(
    mImGray, imDepth, timestamp,
    mpORBextractorLeft, mpORBVocabulary,
    mK, mDistCoef, mbf, mThDepth,
    cv::Mat()   // exclude_mask
    );

    Track();

    return mCurrentFrame.mTcw;
}

cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp,
                                      const std::vector<Detection::Ptr>& detections, bool force_relocalize)
{
    current_frame_idx_ = (current_frame_idx_ + 1) % (std::numeric_limits<size_t>::max()-1);
    mImGray = im;
    im_rgb_ = im;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }
    else if(mImGray.channels()==1)
    {
        // [SỬA MỚI] Clone ảnh nếu đầu vào đã là ảnh xám để tránh ghi đè vùng nhớ của biến const 'im'
        mImGray = mImGray.clone(); 
    }

    // ================================================================
    // PREPROCESSING: Tính texture score → CLAHE adaptive → Unsharp mask
    //
    // Texture score = variance of Laplacian:
    //   - Tường trắng / ít texture: score < 50
    //   - Scene bình thường:        score >= 50
    //
    // Khi low texture:
    //   - CLAHE mạnh hơn (clipLimit=3.0, tile=4×4) để kéo edge ra
    //   - mpIniORBextractor (2x features) để lấy nhiều điểm hơn
    // ================================================================

    // === Bước 1: Tính texture score TRƯỚC khi enhance ===
    {
        cv::Mat lap;
        cv::Laplacian(mImGray, lap, CV_64F);
        cv::Scalar mean_l, std_l;
        cv::meanStdDev(lap, mean_l, std_l);
        current_texture_score_ = std_l[0] * std_l[0]; // variance of Laplacian
        if (current_frame_idx_ % 10 == 0)
            std::cout << "[Texture] score=" << current_texture_score_ << "\n";
    }

    bool bLowTexture = (current_texture_score_ < 50.0);

    // === Bước 2: CLAHE adaptive ===
    {
        static cv::Ptr<cv::CLAHE> clahe_strong = cv::createCLAHE(3.0, cv::Size(4, 4));
        static cv::Ptr<cv::CLAHE> clahe_normal = cv::createCLAHE(1.5, cv::Size(8, 8));
        if (bLowTexture)
            clahe_strong->apply(mImGray, mImGray);
        else
            clahe_normal->apply(mImGray, mImGray);
    }

    // === Bước 3: Unsharp mask nhẹ ===
    {
        cv::Mat blurred;
        cv::GaussianBlur(mImGray, blurred, cv::Size(0, 0), 2.0);
        cv::addWeighted(mImGray, 1.3, blurred, -0.3, 0, mImGray);
    }
    // ================================================================

    // === Xử lý detections và tạo mask loại người TRƯỚC khi tạo Frame ===
    current_frame_detections_ = detections;
    current_frame_good_detections_.clear();
    for (auto det : current_frame_detections_) {
        if (det->score > 0.6f) { 
            current_frame_good_detections_.push_back(det);
        }
    }

    // TẠO MASK RỖNG MẶC ĐỊNH
    cv::Mat exclude_person_mask;

    if (mbUseMask) 
    {
        // [SỬA MỚI] Khởi tạo trực tiếp bằng cv::Scalar(255) sẽ chạy nhanh hơn phép nhân ma trận
        exclude_person_mask = cv::Mat(mImGray.size(), CV_8UC1, cv::Scalar(255));

        for (const auto& det : current_frame_good_detections_)
        {
            if (det->category_id != PERSON_CLASS_ID)  // Chỉ xử lý người
                continue;

            std::cout << "[Mask] Processing person det cat=" << det->category_id
                      << " score=" << det->score
                      << " bbox=[" << det->bbox[0] << "," << det->bbox[1]
                      << "," << det->bbox[2] << "," << det->bbox[3] << "]"
                      << " has_polygon=" << (!det->mask_no_person.empty())
                      << std::endl;

            if (!det->mask_no_person.empty())
            {
                exclude_person_mask.setTo(0, det->mask_no_person);
            }
            else if (!det->mask_object.empty())
            {
                exclude_person_mask.setTo(0, det->mask_object);
            }
            else
            {
                float xmin = det->bbox[0], ymin = det->bbox[1];
                float xmax = det->bbox[2], ymax = det->bbox[3];
                cv::Rect exact(
                    std::max(0, int(xmin)),
                    std::max(0, int(ymin)),
                    std::min(int(xmax - xmin), mImGray.cols - std::max(0, int(xmin))),
                    std::min(int(ymax - ymin), mImGray.rows - std::max(0, int(ymin)))
                );
                exclude_person_mask(exact).setTo(0);
            }
        }
        
        // kernel nhỏ: không ăn vào trần/tường phía trên người
        static cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::erode(exclude_person_mask, exclude_person_mask, kernel, cv::Point(-1,-1), 1);

        int nonzero = cv::countNonZero(exclude_person_mask);
        int total = exclude_person_mask.rows * exclude_person_mask.cols;
        std::cout << "[Mask] Pixels kept after mask: " << nonzero << "/" << total
                  << " (" << (100.0f * nonzero / total) << "%)" << std::endl;
    }

    // Truyền mask vào Frame
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
    {
        mCurrentFrame = Frame(
            mImGray, timestamp,
            mpIniORBextractor, mpORBVocabulary,
            mK, mDistCoef, mbf, mThDepth,
            exclude_person_mask 
        );
    }
    else
    {
        mCurrentFrame = Frame(
            mImGray, timestamp,
            mpORBextractorLeft, mpORBVocabulary,
            mK, mDistCoef, mbf, mThDepth,
            exclude_person_mask 
        );
    }

    if (force_relocalize)
    {
        auto t1 = high_resolution_clock::now();

        bool bOK = false;
        if (mpSystem->GetRelocalizationMode() == RELOC_POINTS) {
            std::cout << "Relocalize with points.\n";
            bOK = Relocalization();
        } else if (mpSystem->GetRelocalizationMode() == RELOC_OBJECTS) {
            std::cout << "Relocalize with objects.\n";
            bOK = RelocalizationFromObjects(false);
        } else if (mpSystem->GetRelocalizationMode() == RELOC_OBJECTS_POINTS) {
            std::cout << "Relocalize with objects and points.\n";
            bOK = Relocalization();
            if (!bOK)
                bOK = RelocalizationFromObjects(true);
        }

        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms_double = t2 - t1;

        mpFrameDrawer->Update(this);
        if (bOK)
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mpSystem->relocalization_duration = ms_double.count();
        mpSystem->relocalization_status = bOK;
    }
    else
    {
        // ==========================================
        // 1. CHẠY THUẦN SLAM (Cốt lõi)
        // ==========================================
        Track();

        // ==========================================
        // 2. CẬP NHẬT CHỈ SỐ CƠ BẢN (KHÔNG VẼ OBJECT)
        // ==========================================
        if (mState == Tracking::OK) {
            Matrix34d Rt = cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw);
            double z_mean = 0.0;
            unsigned int z_nb = 0;
            for(size_t i = 0; i < mCurrentFrame.mvpMapPoints.size(); i++) {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP && !mCurrentFrame.mvbOutlier[i]) {
                    cv::Mat mp =  pMP->GetWorldPos();
                    Eigen::Vector4d p(mp.at<float>(0), mp.at<float>(1), mp.at<float>(2), 1.0);
                    Eigen::Vector3d p_cam = Rt * p;
                    z_mean += p_cam[2];
                    z_nb++;
                }
            }
            if (z_nb > 0) z_mean /= z_nb;
            current_mean_depth_ = z_mean;
        }

        std::cout << "Frame " << current_frame_idx_ << " ===========\n";
        //std::cout << "[Tối ưu] Đã tắt Object Tracking (Hungarian, Ellipsoid) trên Monocular để giảm lag.\n";

        // ==========================================
        // 3. CẬP NHẬT BẢN VẼ
        // ==========================================
        mpFrameDrawer->Update(this);
    }

    // Cập nhật AR Viewer
    if (mpARViewer) {
        mpARViewer->UpdateFrame(im_rgb_);
        if (mCurrentFrame.mTcw.rows == 4)
            mpARViewer->SetCurrentCameraPose(cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw));
    }

    // ── SEED LK BRIDGE: cập nhật mPrevGray + mvPrevPts sau mỗi frame OK ──────
    // Bug trước: mPrevGray không bao giờ được gán ngoài TrackWithOpticalFlow()
    // → LK bridge luôn thấy mPrevGray.empty() = true → luôn bail sớm
    // Fix: seed unconditionally sau mỗi frame để frame sau có dữ liệu track
    mPrevGray = mImGray;
    if (mState == OK)
    {
        // Chỉ seed mvPrevPts khi track tốt, tránh dùng pose rác làm anchor LK
        mvPrevPts.clear();
        mvPrevMPs.clear();
        for (int i = 0; i < mCurrentFrame.N; ++i)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if (pMP && !mCurrentFrame.mvbOutlier[i] && !pMP->isBad()
                && pMP->Observations() >= 2)
            {
                mvPrevPts.push_back(mCurrentFrame.mvKeysUn[i].pt);
                mvPrevMPs.push_back(pMP);
            }
        }
    }
    // ─────────────────────────────────────────────────────────────────────────

    return mCurrentFrame.mTcw;
}

void Tracking::Track()
{
    createdNewKeyFrame_ = false;
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
        {
            // ── GUARD: Nếu map đã đủ lớn (>10 KF) mà vẫn vào NOT_INITIALIZED
            //    nghĩa là SoftReset đã đặt lại state nhưng map vẫn còn.
            //    Chỉ cần Relocalize, không được xóa map rồi init lại.
            if (mpMap->KeyFramesInMap() > 10)
            {
                bool bOK = false;
                if (mpSystem->GetRelocalizationMode() == RELOC_OBJECTS ||
                    mpSystem->GetRelocalizationMode() == RELOC_OBJECTS_POINTS)
                    bOK = RelocalizationFromObjects(false);
                if (!bOK)
                    bOK = Relocalization();
                if (bOK)
                {
                    mState = OK;
                    mnConsecutiveLostFrames = 0;
                    mbInertiaMode = false;
                    mnInertiaFrames = 0;
                    cout << "[Track] Recovered via Relocalization (map preserved).\n";
                }
                else
                {
                    // Vẫn không reloc được → giữ state LOST chứ không init lại
                    mState = LOST;
                }
            }
            else
            {
                MonocularInitialization();
            }
        }

        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK = false;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    // Phát hiện quay nhanh → dùng multi-seed search
                    double rot_deg = ComputeVelocityRotDeg();
                    bool bLargeRotation = (rot_deg > kLargeRotThreshDeg);
                    if (bLargeRotation)
                        std::cout << "[Track] Large rot " << rot_deg << "deg → MultiSeed\n";

                    if (bLargeRotation)
                    {
                        // CHIẾN LƯỢC MULTI-SEED (giống ORB-SLAM3):
                        // 1. Thử tất cả covisible KF làm seed → chọn kết quả tốt nhất
                        // 2. Fallback LK optical flow — sống sót qua motion blur khi cua gắt
                        // 3. Fallback: TrackReferenceKeyFrame thuần
                        // Lý do: khi quay 90°, pose cũ (mLastFrame) hoàn toàn sai hướng
                        // → optimizer bị kéo vào local min → tracking gãy
                        bOK = TrackReferenceKeyFrameMultiSeed();
                        if (bOK)
                        {
                            // Multi-seed OK: clear velocity cũ (không hợp lệ sau quay)
                            mVelocity = cv::Mat();
                            std::cout << "[Track] LargeRot recovered via MultiSeed\n";
                        }
                        else
                        {
                            // Fallback 1: LK optical flow — sống sót qua motion blur khi cua gắt
                            // MultiSeed fail thường vì ảnh nhòe, LK bám gradient mà không cần descriptor
                            bOK = TrackWithOpticalFlow();
                            if (bOK)
                            {
                                mVelocity = cv::Mat();
                                std::cout << "[Track] LargeRot recovered via LK bridge\n";
                            }
                            else
                            {
                                // Fallback 2: BoW thuần với ref KF
                                bOK = TrackReferenceKeyFrame();
                                if (!bOK)
                                {
                                    mVelocity = cv::Mat();
                                    std::cout << "[Track] LargeRot LOST, velocity cleared\n";
                                }
                            }
                        }
                    }
                    else
                    {
                        bOK = TrackWithMotionModel();
                        if(!bOK)
                            bOK = TrackReferenceKeyFrame();
                        // ── LK BRIDGE FALLBACK ──────────────────────────────
                        // Khi cả MotionModel lẫn RefKF đều fail (blur, occlusion,
                        // fast pan): thử track bằng optical flow.
                        if (!bOK)
                        {
                            bOK = TrackWithOpticalFlow();
                            if (bOK)
                                std::cout << "[Track] LK bridge saved tracking (normal path)!\n";
                        }
                        // ────────────────────────────────────────────────────
                    }
                }
            }
            else
            {
                // ── Quick recovery (< full relocalization) ──────────────────
                // Thử các KF gần nhất trước — nhanh hơn BoW query toàn DB
                // Đặc biệt hữu ích khi quay nhanh: KF covisible có góc nhìn gần đúng
                if (!bOK && mpReferenceKF && !mLastFrame.mTcw.isIdentity())
                {
                    std::vector<KeyFrame*> vpQR;
                    vpQR.push_back(mpReferenceKF);
                    auto covisQR = mpReferenceKF->GetBestCovisibilityKeyFrames(5);
                    for (auto* k : covisQR) if (k && !k->isBad()) vpQR.push_back(k);

                    ORBmatcher matcherQR(0.9, true);
                    for (auto* pKFqr : vpQR)
                    {
                        mCurrentFrame.SetPose(mLastFrame.mTcw);
                        std::set<MapPoint*> sFoundQR;
                        int nmQR = matcherQR.SearchByProjection(
                            mCurrentFrame, pKFqr, sFoundQR, 25, 100);
                        if (nmQR >= 12)
                        {
                            Optimizer::PoseOptimization(&mCurrentFrame);
                            int nInQR = 0;
                            for (int i = 0; i < mCurrentFrame.N; i++)
                                if (mCurrentFrame.mvpMapPoints[i]
                                    && !mCurrentFrame.mvbOutlier[i]) nInQR++;
                            if (nInQR >= 10)
                            {
                                std::cout << "[Track] Quick-reloc OK ("
                                          << nInQR << " inliers)\n";
                                bOK = true;
                                break;
                            }
                        }
                    }
                }
                // ────────────────────────────────────────────────────────────

                // ── LK BRIDGE: thử trước khi chạy full BoW relocalization ──
                // Full reloc tốn ~50-100ms; LK chỉ ~2-5ms.
                // Nếu mất track ngắn (người che, blur 1-3 frame) LK thường đủ.
                if (!bOK)
                {
                    bOK = TrackWithOpticalFlow();
                    if (bOK)
                        std::cout << "[Track] LK bridge recovered from LOST state!\n";
                }
                // ───────────────────────────────────────────────────────────

                if (!bOK)
                if (mpSystem->GetRelocalizationMode() == RELOC_POINTS) {
                    std::cout << "Relocalize with points.\n";
                    bOK = Relocalization();
                } else if (mpSystem->GetRelocalizationMode() == RELOC_OBJECTS) {
                    std::cout << "Relocalize with objects.\n";
                    bOK = RelocalizationFromObjects(false);
                } else if (mpSystem->GetRelocalizationMode() == RELOC_OBJECTS_POINTS) {
                    std::cout << "Relocalize with objects and points.\n";
                    bOK = Relocalization();
                    if (!bOK)
                        bOK = RelocalizationFromObjects(true);
                }

                if (bOK) {
                    std::cout << "Relocalization is OK\n";
                }
                else {
                    std::cout << "Relocalization failed\n";
                    
                    // ==================================================================
                    // CƠ CHẾ ÉP BUỘC RE-TRACKING (KHI CHẠY SLAM MODE)
                    // Thay vì dùng pose cũ của referenceKF (sai khi camera đã quay),
                    // thử từng KF trong local map theo thứ tự covisibility
                    // ==================================================================
                    if(mpReferenceKF)
                    {
                        // Lấy danh sách KF ứng viên: referenceKF + các KF covisible gần nhất
                        std::vector<KeyFrame*> vpCandidates;
                        vpCandidates.push_back(mpReferenceKF);
                        auto covisibles = mpReferenceKF->GetBestCovisibilityKeyFrames(5);
                        for(auto* kf : covisibles)
                            if(kf && !kf->isBad())
                                vpCandidates.push_back(kf);

                        bool bFoundGoodKF = false;
                        for(auto* pKFcand : vpCandidates)
                        {
                            std::cout << "Trying KF " << pKFcand->mnId << " as pose seed...\n";
                            mCurrentFrame.SetPose(pKFcand->GetPose());
                            
                            ORBmatcher matcher(0.9, true);
                            std::set<MapPoint*> sFound;
                            int nmatches = matcher.SearchByProjection(
                                mCurrentFrame, pKFcand, sFound, 20, 100);
                            
                            if(nmatches >= 15) {
                                Optimizer::PoseOptimization(&mCurrentFrame);
                                // Đếm inlier thực sau optimize
                                int nInliers = 0;
                                for(int i = 0; i < mCurrentFrame.N; i++)
                                    if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                        nInliers++;
                                if(nInliers >= 10) {
                                    bOK = true;
                                    bFoundGoodKF = true;
                                    std::cout << "Re-Tracking OK with KF " << pKFcand->mnId
                                              << " (" << nInliers << " inliers)\n";
                                    break;
                                }
                            }
                        }

                        if(bOK) {
                            cout << "Re-Tracking Success! Map recovered.\n";
                            bOK = TrackLocalMap();
                        }
                        else {
                            // Tất cả candidates đều thất bại
                            // ĐỨNG YÊN tại pose cuối hợp lệ — KHÔNG bay theo velocity cũ
                            // velocity cũ sai hướng khi camera đã quay góc lớn
                            std::cout << "All KF candidates failed — holding last valid pose.\n";
                            if(!mLastFrame.mTcw.isIdentity())
                                mCurrentFrame.SetPose(mLastFrame.mTcw);
                            // bOK = false → mState = LOST, nhưng pose không bị lung lay
                        }
                    }
                    // ==================================================================
                }
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated
            if(mState==LOST)
            {
                if (mpSystem->GetRelocalizationMode() == RELOC_POINTS) {
                    std::cout << "Relocalize with points.\n";
                    bOK = Relocalization();
                } else if (mpSystem->GetRelocalizationMode() == RELOC_OBJECTS) {
                    std::cout << "Relocalize with objects.\n";
                    bOK = RelocalizationFromObjects(false);
                } else if (mpSystem->GetRelocalizationMode() == RELOC_OBJECTS_POINTS) {
                    std::cout << "Relocalize with objects and points.\n";
                    bOK = Relocalization();
                    if (!bOK)
                        bOK = RelocalizationFromObjects(true);
                }

                // ==================================================================
                // CƠ CHẾ ÉP BUỘC RE-TRACKING (KHI CHẠY LOCALIZATION MODE)
                // ==================================================================
                if(!bOK && mpReferenceKF)
                {
                    std::vector<KeyFrame*> vpCandidates;
                    vpCandidates.push_back(mpReferenceKF);
                    auto covisibles = mpReferenceKF->GetBestCovisibilityKeyFrames(5);
                    for(auto* kf : covisibles)
                        if(kf && !kf->isBad())
                            vpCandidates.push_back(kf);

                    for(auto* pKFcand : vpCandidates)
                    {
                        mCurrentFrame.SetPose(pKFcand->GetPose());
                        ORBmatcher matcher(0.9, true);
                        std::set<MapPoint*> sFound;
                        int nmatches = matcher.SearchByProjection(
                            mCurrentFrame, pKFcand, sFound, 20, 100);
                        if(nmatches >= 15) {
                            Optimizer::PoseOptimization(&mCurrentFrame);
                            int nInliers = 0;
                            for(int i = 0; i < mCurrentFrame.N; i++)
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                    nInliers++;
                            if(nInliers >= 10) {
                                bOK = true;
                                cout << "Re-Tracking Success (Loc mode)! KF "
                                     << pKFcand->mnId << "\n";
                                break;
                            }
                        }
                    }

                    if(!bOK && !mLastFrame.mTcw.isIdentity())
                        mCurrentFrame.SetPose(mLastFrame.mTcw); // đứng yên
                }
                // ==================================================================
            }
            else if(!mbVO)
            {
                // In last frame we tracked enough MapPoints in the map
                if(!mVelocity.empty())
                {
                    bOK = TrackWithMotionModel();
                }
                else
                {
                    bOK = TrackReferenceKeyFrame();
                }

                // CỜ SINH TỒN — chỉ dùng khi mbOnlyTracking
                // Khi không track được: đứng yên tại pose cuối hợp lệ
                // KHÔNG dùng velocity để "bay tiếp" vì velocity có thể sai hướng
                // (ví dụ camera đã quay 90° nhưng velocity vẫn là vector thẳng cũ)
                if(!bOK) 
                {
                    if (!mLastFrame.mTcw.isIdentity()) {
                        mCurrentFrame.SetPose(mLastFrame.mTcw);
                        bOK = true;
                        mbVO = true; // Báo đang "đi mù" để hạn chế tạo KF
                    }
                }
            }
            else
            {
                bool bOKMM = false;
                bool bOKReloc = false;
                vector<MapPoint*> vpMPsMM;
                vector<bool> vbOutMM;
                cv::Mat TcwMM;
                if(!mVelocity.empty())
                {
                    bOKMM = TrackWithMotionModel();
                    vpMPsMM = mCurrentFrame.mvpMapPoints;
                    vbOutMM = mCurrentFrame.mvbOutlier;
                    TcwMM = mCurrentFrame.mTcw;
                }
                bOKReloc = Relocalization();

                if(bOKMM && !bOKReloc)
                {
                    mCurrentFrame.SetPose(TcwMM);
                    mCurrentFrame.mvpMapPoints = vpMPsMM;
                    mCurrentFrame.mvbOutlier = vbOutMM;

                    if(mbVO)
                    {
                        for(int i =0; i<mCurrentFrame.N; i++)
                        {
                            if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                            {
                                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                            }
                        }
                    }
                }
                else if(bOKReloc)
                {
                    mbVO = false;
                }

                bOK = bOKReloc || bOKMM;
            }
        }
        
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
                bOK = TrackLocalMap();
        }
        else
        {
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }
        
        // =========================================================
        // ----> CHÈN ĐOẠN "RESET MỀM" <----
        // =========================================================
        /*if(bOK && mnMatchesInliers < 12) 
        {
            cout << "Soft Reset: Low inliers (" << mnMatchesInliers << "), forcing re-track..." << endl;
            SoftResetTracking(); 
            bOK = false;         
        }*/
        // =========================================================

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Cập nhật bộ đếm lost/inertia
        if (mState == OK)
        {
            mnConsecutiveLostFrames = 0;
            mbInertiaMode = false;
            mnInertiaFrames = 0;
        }
        else
        {
            mnConsecutiveLostFrames++;
        }

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model - ORB-SLAM2 standard
            if(!mLastFrame.mTcw.isIdentity())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.block<3,3>(0,0));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.block<3,1>(0,3));
                cv::Mat newVel = mCurrentFrame.mTcw * LastTwc;

                // Sanity check: velocity rotation > 15°/frame là bất thường, clear
                // 15fps: quay 90° trong 1.5s = 6°/frame bình thường
                // 15° = tốc độ quay ~135°/s, vượt mức này velocity bắt đầu poison
                // 30° cũ quá cao: garbage velocity tích lũy ~5 frame trước khi bị catch
                cv::Mat Rnv = newVel.block<3,3>(0,0);
                double trnv = Rnv.at<float>(0,0)+Rnv.at<float>(1,1)+Rnv.at<float>(2,2);
                double rotnv = std::acos(std::max(-1.0,std::min(1.0,(trnv-1.0)/2.0)))*180.0/M_PI;
                if (rotnv > 15.0)
                {
                    std::cout << "[Track] Velocity sanity fail: " << rotnv << "deg → cleared\n";
                    mVelocity = cv::Mat();
                }
                else
                    mVelocity = newVel;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame()) {
                CreateNewKeyFrame();
                createdNewKeyFrame_ = true;
            }

            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        // Chỉ reset khi: map thực sự nhỏ (≤5 KF) VÀ đã lost liên tục nhiều frame
        // Tránh reset khi vật thể che tạm thời
        if(mState==LOST)
        {
            const int nKFs = mpMap->KeyFramesInMap();
            // Chỉ reset khi: map thực sự chưa build (<=3 KF)
            // VÀ lost lâu (>=30 frame = 1 giây ở 30fps)
            // Tránh reset sớm khi người che tạm thời (thường chỉ 10-20 frame)
            if(nKFs <= 3 && mnConsecutiveLostFrames >= 30)
            {
                cout << "Track lost soon after init, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
            else if(nKFs > 3)
            {
                if (mnConsecutiveLostFrames % 30 == 0)
                    cout << "[Track] LOST for " << mnConsecutiveLostFrames
                         << " frames (" << nKFs << " KFs preserved)\n";
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.isIdentity() && mCurrentFrame.mpReferenceKF)
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }
}

void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));    // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

void Tracking::MonocularInitialization()
{
    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    else
    {
        // Try to initialize
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.block<3,3>(0,0));
            tcw.copyTo(Tcw.block<3,1>(0,3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // ── SCALE CONSISTENCY CHECK ──────────────────────────────────────────
    // Nếu đã từng init thành công trước đó, kiểm tra scale của lần này
    // có hợp lý không. Nếu lệch > 10x thì scene/parallax bất thường → bỏ qua
    if (mLastInitMedianDepth > 0.0f)
    {
        // Sau khi scale về 1.0, medianDepth * invMedianDepth = 1.0 luôn.
        // Ta so sánh raw medianDepth (trước scale) với lần trước.
        // Lần trước cũng đã scale về 1.0, nên mLastInitMedianDepth = 1.0.
        // Không thể so sánh trực tiếp → chỉ check medianDepth có hợp lý (>0, <100m scale)
        // Nếu medianDepth quá nhỏ (camera gần tường) hoặc quá lớn → parallax sai
        if (medianDepth < 0.01f || medianDepth > 1000.0f)
        {
            cout << "Scale inconsistent (medianDepth=" << medianDepth
                 << "), discarding this initialization..." << endl;
            Reset();
            return;
        }
    }
    mLastInitMedianDepth = 1.0f; // Sau scale luôn = 1.0
    // ─────────────────────────────────────────────────────────────────────

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

// ─── Helper: tính rotation angle (degree) từ velocity matrix ───────────────
double Tracking::ComputeVelocityRotDeg() const
{
    if (mVelocity.empty()) return 0.0;
    cv::Mat Rv = mVelocity.block<3,3>(0,0);
    double tr = Rv.at<float>(0,0)+Rv.at<float>(1,1)+Rv.at<float>(2,2);
    return std::acos(std::max(-1.0,std::min(1.0,(tr-1.0)/2.0)))*180.0/M_PI;
}

// ─── Multi-seed TrackReferenceKeyFrame ──────────────────────────────────────
// Thay vì chỉ dùng mpReferenceKF làm seed, thử N KF tốt nhất trong covisibility.
// Cách tiếp cận này giống ORB-SLAM3: khi large rotation, search rộng hơn.
bool Tracking::TrackReferenceKeyFrameMultiSeed()
{
    mCurrentFrame.ComputeBoW();

    // Xây danh sách candidate KF theo covisibility
    std::vector<KeyFrame*> vpCandKFs;
    if (mpReferenceKF)
    {
        vpCandKFs.push_back(mpReferenceKF);
        auto covis = mpReferenceKF->GetBestCovisibilityKeyFrames(kMaxMultiSeedKF);
        for (auto* kf : covis)
            if (kf && !kf->isBad()) vpCandKFs.push_back(kf);
    }
    // Thêm local KF nếu còn ít
    for (auto* kf : mvpLocalKeyFrames)
    {
        if ((int)vpCandKFs.size() >= kMaxMultiSeedKF) break;
        if (kf && !kf->isBad())
            vpCandKFs.push_back(kf);
    }

    ORBmatcher matcher(0.75, true);
    int bestInliers = 0;
    cv::Mat bestTcw;
    vector<MapPoint*> bestMapPoints;

    for (auto* pKF : vpCandKFs)
    {
        if (!pKF || pKF->isBad()) continue;

        vector<MapPoint*> vpMPM;
        int nm = matcher.SearchByBoW(pKF, mCurrentFrame, vpMPM);
        if (nm < 10) continue;

        // CHỈ dùng pose của KF làm seed — KHÔNG dùng mLastFrame
        // Lý do: khi large rotation, mLastFrame.pose sai ~90° so với góc thực
        // → optimizer bị kéo vào local minimum gần mLastFrame thay vì pose đúng
        // KF pose là ground truth đã được bundle-adjusted → đáng tin hơn nhiều
        mCurrentFrame.mvpMapPoints = vpMPM;
        for (int i = 0; i < mCurrentFrame.N; i++) mCurrentFrame.mvbOutlier[i] = false;
        mCurrentFrame.SetPose(pKF->GetPose());
        Optimizer::PoseOptimization(&mCurrentFrame);

        int nInliers = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
            if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i]
                && mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nInliers++;

        if (nInliers > bestInliers)
        {
            bestInliers = nInliers;
            bestTcw = mCurrentFrame.mTcw;
            bestMapPoints = mCurrentFrame.mvpMapPoints;
            if (nInliers >= 30) goto found_good;
        }
    }

found_good:
    if (bestInliers < 10 || bestTcw.empty())
        return false;

    // Áp kết quả tốt nhất
    mCurrentFrame.SetPose(bestTcw);
    mCurrentFrame.mvpMapPoints = bestMapPoints;
    for (int i = 0; i < mCurrentFrame.N; i++) mCurrentFrame.mvbOutlier[i] = false;
    // Re-optimize với pose tốt nhất để loại outlier chính xác hơn
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers và đếm inlier cuối
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                mCurrentFrame.mvpMapPoints[i] = nullptr;
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }

    std::cout << "[MultiSeed] bestInliers=" << bestInliers
              << " nmatchesMap=" << nmatchesMap << "\n";
    return nmatchesMap >= 10;
}

bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<10)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;

    // Chọn seed tốt nhất: nếu velocity đang quay >10°, dùng refKF pose
    double rotDeg = ComputeVelocityRotDeg();
    bool bUseRefKFSeed = (rotDeg > 10.0 && mpReferenceKF);

    if (bUseRefKFSeed)
        mCurrentFrame.SetPose(mpReferenceKF->GetPose());
    else
        mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Nếu inlier thấp, thử seed còn lại
    {
        int nI = 0;
        for(int i=0;i<mCurrentFrame.N;i++)
            if(mCurrentFrame.mvpMapPoints[i]&&!mCurrentFrame.mvbOutlier[i]
               &&mCurrentFrame.mvpMapPoints[i]->Observations()>0) nI++;
        if (nI < 15)
        {
            mCurrentFrame.mvpMapPoints = vpMapPointMatches;
            for(int i=0;i<mCurrentFrame.N;i++) mCurrentFrame.mvbOutlier[i]=false;
            if (bUseRefKFSeed)
                mCurrentFrame.SetPose(mLastFrame.mTcw);
            else if (mpReferenceKF)
                mCurrentFrame.SetPose(mpReferenceKF->GetPose());
            Optimizer::PoseOptimization(&mCurrentFrame);
        }
    }

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=30;
    else
        th=15;
    // Nếu người che nhiều (mask loại >40% pixel), tăng th để tìm
    // điểm trong vùng còn lại (trần, tường bên) rộng hơn
    {
        const cv::Mat& emask = mCurrentFrame.GetExcludeMask();
        if (!emask.empty())
        {
            int nz = cv::countNonZero(emask);
            float kept = static_cast<float>(nz) / (emask.rows * emask.cols);
            if (kept < 0.6f) // người chiếm >40% frame
                th = std::min(th * 2, 100);
        }
    }

    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR); 
    }

    // Nếu vẫn ít match và velocity đang có rotation đáng kể → seed bằng refKF pose
    // 15fps: motion model seed sai hướng khi quay ≥ 4-5°/frame, refKF pose ổn định hơn
    if (nmatches < 20 && mpReferenceKF && !mVelocity.empty())
    {
        double rotDeg = ComputeVelocityRotDeg();
        if (rotDeg > 3.0)   // 3° ở 15fps = bắt đầu quay vừa
        {
            fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),
                 static_cast<MapPoint*>(NULL));
            mCurrentFrame.SetPose(mpReferenceKF->GetPose());
            nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);
            if (nmatches < 20)
                nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
            std::cout << "[TrackMM] refKF-seed fallback: " << nmatches
                      << " matches (vel_rot=" << rotDeg << "deg)\n";
        }
    }

    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }   

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{
    UpdateLocalMap();
    SearchLocalPoints();

    // === CHỈ LỌC MAPPOINT NẰM TRÊN NGƯỜI NẾU MB_USE_MASK = TRUE ===
    if (mbUseMask) 
    {
        const cv::Mat& exclude_mask = mCurrentFrame.GetExcludeMask();
        int removed = 0;

        if (!exclude_mask.empty())
        {
            // Tìm y_top nhỏ nhất (cao nhất trên ảnh) của tất cả bbox người
            // để nhận diện điểm "chắc chắn trên trần" ngay cả khi obs thấp
            float min_person_top = static_cast<float>(exclude_mask.rows);
            for (const auto& det : current_frame_good_detections_)
            {
                if (det->category_id != PERSON_CLASS_ID) continue;
                float bbox_top = det->bbox[1];
                if (bbox_top < min_person_top) min_person_top = bbox_top;
            }
            // Vùng an toàn: ít nhất 20px trên đỉnh bbox người cao nhất
            // Điểm trong vùng này chắc chắn không phải người dù obs thấp
            const float safe_ceiling_y = min_person_top - 20.0f;

            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                const cv::KeyPoint& kp = mCurrentFrame.mvKeysUn[i];
                int cx = cvRound(kp.pt.x);
                int cy = cvRound(kp.pt.y);
                if (cx < 0 || cy < 0 || cx >= exclude_mask.cols || cy >= exclude_mask.rows)
                    continue;
                if (exclude_mask.at<uchar>(cy, cx) != 0)
                    continue; // pixel này không phải vùng người → giữ lại

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if (pMP && !pMP->isBad())
                {
                    // Điểm đã được quan sát nhiều lần (bundle-adjusted ổn định): giữ vô điều kiện
                    if (pMP->Observations() >= 5)
                        continue;

                    // Điểm ở vùng trần/tường phía trên tất cả bbox người: giữ lại
                    // Đây là điểm "50-50" bị miss trước đây: trần nhà đôi khi detect được,
                    // nhưng nếu pixel rơi vào vùng exclude_mask do erosion quá rộng thì bị xóa oan
                    if (static_cast<float>(cy) < safe_ceiling_y && pMP->Observations() >= 2)
                        continue;

                    // obs 3-4 nhưng không ở trên trần: vẫn kiểm tra từng det người
                    // để tránh xóa điểm thực sự trên tường bên cạnh người
                    if (pMP->Observations() >= 3)
                    {
                        bool definitely_on_person = false;
                        for (const auto& det : current_frame_good_detections_)
                        {
                            if (det->category_id != PERSON_CLASS_ID) continue;
                            // Chỉ xóa nếu pixel nằm BÊN TRONG bbox người (không phải chỉ gần)
                            if (cx >= det->bbox[0] && cx <= det->bbox[2] &&
                                cy >= det->bbox[1] && cy <= det->bbox[3])
                            {
                                definitely_on_person = true;
                                break;
                            }
                        }
                        if (!definitely_on_person)
                            continue; // nằm ngoài bbox nhưng trong mask vì erode → giữ lại
                    }
                }
                mCurrentFrame.mvbOutlier[i] = true;
                removed++;
            }
        }

        if (removed > 0)
        {
            std::cout << "[TrackLocalMap] Mask ON: Removed " << removed
                      << " potential dynamic points." << std::endl;
        }
    }
    else 
    {
        // Khi mbUseMask = false, không thực hiện đánh dấu Outlier dựa trên mask
    }

    // === TỐI ƯU POSE SAU KHI ĐÃ LỌC ===
    Optimizer::PoseOptimization(&mCurrentFrame);

    // === ĐẾM INLIER ỔN ĐỊNH TRƯỚC KHI QUYẾT ĐỊNH (để tính ngưỡng linh hoạt) ===
    // Đếm riêng điểm "có anchor trong map" (obs>0) để biết mask có xóa quá nhiều không
    int nStableInliers = 0;
    mnMatchesInliers = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if (!pMP) continue;

        if (!mCurrentFrame.mvbOutlier[i])
        {
            pMP->IncreaseFound();
            if (!mbOnlyTracking)
            {
                if (pMP->Observations() > 0)
                {
                    mnMatchesInliers++;
                    if (pMP->Observations() >= 3) nStableInliers++;
                }
            }
            else
                mnMatchesInliers++;
        }
        else if (mSensor == System::STEREO)
        {
            mCurrentFrame.mvpMapPoints[i] = nullptr;
        }
    }

    // Ngưỡng inlier linh hoạt:
    // - Bình thường: >= 8
    // - Khi người che nhiều (mask xóa >40% điểm, nStableInliers thấp): hạ xuống 5
    // - Khi low texture (tường trắng) vừa quay ra: hạ xuống 6 để không false-LOST
    //   ngay khi bắt đầu thấy phòng trở lại
    int inlierThresh = 8;
    if (mbUseMask && nStableInliers < 15 && mnMatchesInliers > 0)
    {
        // Kiểm tra xem có người đang che nhiều không
        bool bHeavyOcclusion = false;
        for (const auto& det : current_frame_good_detections_)
        {
            if (det->category_id != PERSON_CLASS_ID) continue;
            float bboxArea = (det->bbox[2]-det->bbox[0]) * (det->bbox[3]-det->bbox[1]);
            float frameArea = static_cast<float>(mImGray.cols * mImGray.rows);
            if (bboxArea / frameArea > 0.25f) { bHeavyOcclusion = true; break; }
        }
        if (bHeavyOcclusion)
        {
            inlierThresh = 5;
            std::cout << "[TrackLocalMap] Heavy occlusion: inlierThresh=" << inlierThresh
                      << " stableInliers=" << nStableInliers << "\n";
        }
    }
    // Low-texture case: vừa quay khỏi tường, ORB features vẫn thưa
    // Không liên quan người → kiểm tra riêng
    if (inlierThresh > 6 && current_texture_score_ < 50.0 && nStableInliers < 12)
    {
        inlierThresh = 6;
        std::cout << "[TrackLocalMap] LowTexture fallback: inlierThresh=6"
                  << " texture=" << current_texture_score_
                  << " stableInliers=" << nStableInliers << "\n";
    }

    if (mnMatchesInliers < inlierThresh)
    {
        mbInertiaMode = false;
        mnInertiaFrames = 0;
        return false;
    }

    // Đủ inlier
    mbInertiaMode = false;
    mnInertiaFrames = 0;

    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 15)
        return false;

    return true;
}

bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)
        return false;

    // Không tạo KF khi đang chạy bằng quán tính
    if(mbInertiaMode)
        return false;

    // Absolute inlier guard: khi không có người che, cần ≥15 inlier
    // Khi người che nhiều (>25% frame), hạ về 10 để vẫn tạo được anchor KF
    {
        int nkfInlierThresh = 15;
        if (mbUseMask)
        {
            for (const auto& det : current_frame_good_detections_)
            {
                if (det->category_id != PERSON_CLASS_ID) continue;
                float bboxArea = (det->bbox[2]-det->bbox[0]) * (det->bbox[3]-det->bbox[1]);
                float frameArea = static_cast<float>(
                    (mImGray.empty() ? 480 : mImGray.rows) *
                    (mImGray.empty() ? 640 : mImGray.cols));
                if (bboxArea / frameArea > 0.25f) { nkfInlierThresh = 10; break; }
            }
        }
        if (mnMatchesInliers < nkfInlierThresh)
            return false;
    }

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
    {
        thRefRatio = 0.9f;
        // Khi đang quay rõ rệt: cho phép tạo KF dễ hơn để có nhiều anchor
        // 15fps: 5° đã là quay rõ (4°/frame ở 90°/1.5s), khớp với kLargeRotThreshDeg
        double rotNKF = ComputeVelocityRotDeg();
        if (rotNKF > 5.0)
        {
            thRefRatio = 0.75f;
            std::cout << "[NKF] Rotation=" << rotNKF << "deg → thRefRatio=0.75\n";
        }
        // Ít texture → tạo KF sớm hơn để có nhiều anchor trước khi quay đi
        if (current_texture_score_ < 50.0)
        {
            thRefRatio = std::min(thRefRatio, 0.80f);
            std::cout << "[NKF] LowTexture score=" << current_texture_score_
                      << " → thRefRatio=" << thRefRatio << "\n";
        }
    }

    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    const bool c1c = mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose);
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    // Kiểm tra camera có thực sự di chuyển không (chỉ Monocular)
    bool bEnoughMotion = true;
    if (mSensor==System::MONOCULAR && !mVelocity.empty())
    {
        cv::Mat tv=mVelocity.block<3,1>(0,3);
        double trans=cv::norm(tv);
        double rot=ComputeVelocityRotDeg();
        // rot>2° hoặc trans đủ lớn mới tạo KF
        bEnoughMotion=(rot>2.0||trans>0.003);
    }

    if((c1a||c1b||c1c)&&c2&&bEnoughMotion)
    {
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    // ─── Cập nhật baseline inlier (chỉ dùng để log) ─────────────────────────
    mnLastKFInliers = mnMatchesInliers;
    // ────────────────────────────────────────────────────────────────────────

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 3;
        if(mSensor==System::RGBD) th=3;
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2) th=5;
        if (!mVelocity.empty())
        {
            cv::Mat Rv=mVelocity.block<3,3>(0,0);
            double tr=Rv.at<float>(0,0)+Rv.at<float>(1,1)+Rv.at<float>(2,2);
            double rot=std::acos(std::max(-1.0,std::min(1.0,(tr-1.0)/2.0)))*180.0/M_PI;
            if(rot>5.0) th=std::min(10,th+(int)(rot/5.0));
        }
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::RelocalizationFromObjects(bool use_points)
{
    std::vector<BBox2, Eigen::aligned_allocator<BBox2>> bboxes;
    std::vector<size_t> bboxes_categories;
    for (auto det : current_frame_good_detections_) {
        if (det->score > 0.1) {
            bboxes.push_back(det->bbox);
            bboxes_categories.push_back(det->category_id);
        }
    }
    std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>> ellipsoids;
    std::vector<size_t> ellipsoids_categories;
    auto objects = mpMap->GetAllMapObjects();
    for (auto* obj : objects) {
        ellipsoids.push_back(obj->GetEllipsoid());
        ellipsoids_categories.push_back(obj->GetTrack()->GetCategoryId());
    }
    std::cout << "Nb detections / ellipsoids = " << bboxes.size() << " / " << ellipsoids.size() << "\n";

    auto [best_index, poses, scores, used_pairs, used_inliers] = ORB_SLAM2::solveP3P_ransac(ellipsoids, ellipsoids_categories, bboxes, bboxes_categories, K_, 0.2);

    if (best_index == -1 || poses.size() == 0 || scores[best_index] > bboxes.size())
        return false; // reloc failed

    std::vector<int> indices(poses.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&scores](int a, int b) {
        return scores[a] < scores[b];
    });

    cv::Mat best_Rt_objects;
    for (int i = 0; i < ((int)indices.size()) && scores[indices[i]] < bboxes.size(); ++i) {
        int best_index = indices[i]; // current index

        const auto& pose_est = poses[best_index];
        Eigen::Matrix3d o_est = pose_est.block<3, 3>(0, 0);
        Eigen::Vector3d p_est = pose_est.col(3);
        Matrix34d Rt_est;
        Rt_est << o_est.transpose(), -o_est.transpose() * p_est;
        cv::Mat Rt(4, 4, CV_32F, 0.0);
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j)
                Rt.at<float>(i, j) = Rt_est(i, j);
        Rt.at<float>(3, 3) = 1.0;
        mCurrentFrame.SetPose(Rt);

        // Optimize camera pose by minimizing object reprojection error
        std::vector<Ellipse, Eigen::aligned_allocator<Ellipse>> ellipses_inliers;
        std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>> ellipsoids_inliers;
        for (auto x : used_inliers[best_index]) {
            Ellipse ellipse = Ellipse::FromBbox(bboxes[x.first]);
            ellipses_inliers.push_back(ellipse);
            ellipsoids_inliers.push_back(ellipsoids[x.second]);
        }

        Rt = OptimizePoseFromObjects(ellipses_inliers, ellipsoids_inliers, Rt, K_);

        if (i == 0) {
            best_Rt_objects = Rt;
        }
        // Match keypoints from the current frame to map points using the estimated pose
        auto map_points = mpMap->GetAllMapPoints();
        ORBmatcher matcher(0.9, true);
        int nb_matches = matcher.SearchByProjectionAllPoints(mCurrentFrame, map_points, 10, 100);
        std::cout << nb_matches << " matched points by projection.\n";

        if (nb_matches > 10) // the pose is considered as good, we can stop
        {
            if (use_points) {
                // further optimization
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                std::cout << "Optimized pose using matched map points.\n";
            }

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
            return true;
        }
    }
    mCurrentFrame.SetPose(best_Rt_objects);
    mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

    return true;
}

bool Tracking::Relocalization()
{
    mLastProcessedState = mState;

    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame); 

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<10)  // giảm từ 15→10 cho scene ít texture
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,8,500,4,0.5,5.991);  // min 8, max 500 iter
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                mCurrentFrame.mTcw = Tcw;

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                // Dưới 10 điểm thì bỏ qua luôn, coi như góc đó quá mù
                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // Nếu có ít điểm (từ 10 đến 20), thử tìm kiếm quanh vùng đó xem có gỡ gạc thêm được điểm nào không
                if(nGood<20)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=20) // Đã sửa 50 thành 20
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // Tìm được hòm hòm rồi (trên 15) nhưng vẫn chưa đủ 20, ráng tìm kỹ hơn ở phạm vi hẹp
                        if(nGood>15 && nGood<20) // Đã sửa 30, 50 thành 15, 20
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Tối ưu lần cuối
                            if(nGood+nadditional>=20) // Đã sửa 50 thành 20
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }
                
                // NẾU TỔNG SỐ ĐIỂM CHẤT LƯỢNG >= 20 THÌ CHỐT ĐÓNG VÒNG LẶP/RELOCALIZE!
                if(nGood>=20) // Đã sửa 50 thành 20
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }
}

void Tracking::Reset()
{
    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    if (local_object_mapper_) {
        cout << "Reseting Local Object Mapper...";
        local_object_mapper_->RequestReset();
        cout << " done" << endl;
    }

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    objectTracks_.clear();
    mpMap->clear();
    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    // Reset các counter chống drift
    mbInertiaMode = false;
    mnInertiaFrames = 0;
    mnConsecutiveLostFrames = 0;
    mLastInitMedianDepth = -1.0f;
    mnLastKFInliers = 0;
    mLastTrackedScale = 1.0;
    mnStableScaleFrames = 0;

    // Reset LK bridge state — tránh dùng ảnh/điểm cũ từ session trước
    mPrevGray.release();
    mvPrevPts.clear();
    mvPrevMPs.clear();
    mvLKErrors.clear();
    mnLKFallbackFrames = 0;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    mK = K;

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    mDistCoef = DistCoef;

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}


void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}
void Tracking::SoftResetTracking()
{
    cout << "Soft Reset: Low inliers, triggering Relocalization..." << endl;
    
    mVelocity = cv::Mat(); 
    mbInertiaMode = false;
    mnInertiaFrames = 0;
    mnLastKFInliers = 0;      // reset quality baseline khi soft-reset
    mState = LOST;

    // Clear LK state: seed cũ không còn phù hợp sau soft-reset
    mPrevGray.release();
    mvPrevPts.clear();
    mvPrevMPs.clear();
    mvLKErrors.clear();
    mnLKFallbackFrames = 0;
}

// =============================================================================
// LK OPTICAL FLOW BRIDGE
// =============================================================================
// Chạy khi ORB matching thất bại (blur, occlusion, fast pan, low texture).
// Pipeline:
//   1. calcOpticalFlowPyrLK: track mvPrevPts từ mPrevGray → mImGray
//   2. Forward-backward check để loại điểm không ổn định
//   3. Epipolar check nhẹ (projection error) để loại điểm sai 3D
//   4. PnP (solvePnPRansac) từ 3D(MapPoint) ↔ 2D(tracked pt)
//   5. Optimizer::PoseOptimization để refine
//   6. Cập nhật mvPrevPts/mvPrevMPs cho frame tiếp theo
// =============================================================================
bool Tracking::TrackWithOpticalFlow()
{
    // Cần có dữ liệu frame trước
    if (mPrevGray.empty() || mvPrevPts.empty() || mvPrevMPs.empty())
    {
        std::cout << "[LK] No prev frame data, skip.\n";
        return false;
    }

    // ── 1. FORWARD TRACK ────────────────────────────────────────────────────
    std::vector<cv::Point2f> curr_pts;
    std::vector<uchar> status_fwd;
    std::vector<float> err_fwd;

    cv::calcOpticalFlowPyrLK(
        mPrevGray, mImGray,
        mvPrevPts, curr_pts,
        status_fwd, err_fwd,
        cv::Size(21, 21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01)
    );

    // ── 2. BACKWARD CHECK (forward-backward consistency) ────────────────────
    std::vector<cv::Point2f> back_pts;
    std::vector<uchar> status_bwd;
    std::vector<float> err_bwd;

    cv::calcOpticalFlowPyrLK(
        mImGray, mPrevGray,
        curr_pts, back_pts,
        status_bwd, err_bwd,
        cv::Size(21, 21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01)
    );

    // Lưu LK errors để dùng trong ComputeDynamicScores
    mvLKErrors.assign(mvPrevPts.size(), 999.f);

    // ── 3. LỌC ĐIỂM HỢP LỆ ─────────────────────────────────────────────────
    // Điều kiện giữ lại:
    //   a) forward track OK
    //   b) backward track OK
    //   c) forward-backward error < 2px (consistency)
    //   d) LK error < 30 (không bị blur/occlude quá mạnh)
    //   e) điểm vẫn nằm trong ảnh
    const float FB_THRESH    = 2.0f;   // forward-backward consistency (px)
    const float ERR_THRESH   = 30.0f;  // LK photometric error

    std::vector<cv::Point2f> pts3d_src;   // 2D undistorted tại prev (không dùng)
    std::vector<cv::Point3f> pts_3d;      // 3D world coords từ MapPoint
    std::vector<cv::Point2f> pts_2d;      // 2D obs tại curr frame
    std::vector<MapPoint*>   valid_mps;   // MapPoint tương ứng

    for (size_t i = 0; i < mvPrevPts.size(); ++i)
    {
        if (!status_fwd[i] || !status_bwd[i]) continue;

        float fb_err = cv::norm(mvPrevPts[i] - back_pts[i]);
        if (fb_err > FB_THRESH) continue;
        if (err_fwd[i] > ERR_THRESH) continue;

        // Điểm phải nằm trong ảnh
        const cv::Point2f& cp = curr_pts[i];
        if (cp.x < 0 || cp.y < 0 ||
            cp.x >= mImGray.cols || cp.y >= mImGray.rows) continue;

        // Kiểm tra MapPoint còn hợp lệ không
        MapPoint* pMP = mvPrevMPs[i];
        if (!pMP || pMP->isBad()) continue;

        cv::Mat wpos = pMP->GetWorldPos();
        pts_3d.emplace_back(wpos.at<float>(0), wpos.at<float>(1), wpos.at<float>(2));
        pts_2d.push_back(cp);
        valid_mps.push_back(pMP);
        mvLKErrors[i] = err_fwd[i];
    }

    std::cout << "[LK] Valid pts after FB check: " << pts_3d.size()
              << " / " << mvPrevPts.size() << "\n";

    if ((int)pts_3d.size() < 8)
    {
        std::cout << "[LK] Too few valid pts, fallback to relocalize.\n";
        return false;
    }

    // ── 4. PnP RANSAC ────────────────────────────────────────────────────────
    cv::Mat rvec, tvec;
    std::vector<int> inliers_idx;

    // Dùng pose cũ làm initial guess nếu có
    bool use_extrinsic_guess = false;
    if (!mLastFrame.mTcw.isIdentity())
    {
        cv::Mat R_last = mLastFrame.mTcw.block<3,3>(0,0);
        cv::Mat t_last = mLastFrame.mTcw.block<3,1>(0,3);
        cv::Rodrigues(R_last, rvec);
        tvec = t_last.clone();
        use_extrinsic_guess = true;
    }

    bool pnp_ok = cv::solvePnPRansac(
        pts_3d, pts_2d,
        mK, mDistCoef,
        rvec, tvec,
        use_extrinsic_guess,
        100,    // iterations
        4.0f,   // reprojection error threshold (px)
        0.99,   // confidence
        inliers_idx,
        cv::SOLVEPNP_ITERATIVE
    );

    if (!pnp_ok || (int)inliers_idx.size() < 6)
    {
        std::cout << "[LK] PnP failed or too few inliers ("
                  << inliers_idx.size() << ").\n";
        return false;
    }

    std::cout << "[LK] PnP inliers: " << inliers_idx.size() << "\n";

    // ── 5. TẠO POSE TỪ PnP → ĐƯA VÀO FRAME ────────────────────────────────
    cv::Mat R_mat;
    cv::Rodrigues(rvec, R_mat);
    R_mat.convertTo(R_mat, CV_32F);
    tvec.convertTo(tvec, CV_32F);

    cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
    R_mat.copyTo(Tcw.block<3,3>(0,0));
    tvec.copyTo(Tcw.block<3,1>(0,3));
    mCurrentFrame.SetPose(Tcw);

    // Gán MapPoints cho frame hiện tại từ inlier set
    std::fill(mCurrentFrame.mvpMapPoints.begin(),
              mCurrentFrame.mvpMapPoints.end(), nullptr);
    std::fill(mCurrentFrame.mvbOutlier.begin(),
              mCurrentFrame.mvbOutlier.end(), false);

    // Tạo map idx: pts_2d[i] → keypoint index trong mCurrentFrame
    // Dùng nearest neighbour đơn giản (O(N*M) nhưng N, M nhỏ ở đây)
    for (int idx : inliers_idx)
    {
        const cv::Point2f& p = pts_2d[idx];
        float best_d = 4.0f; // ngưỡng gán (px)
        int best_k = -1;
        for (int k = 0; k < mCurrentFrame.N; ++k)
        {
            const cv::Point2f& kp = mCurrentFrame.mvKeysUn[k].pt;
            float d = std::hypot(kp.x - p.x, kp.y - p.y);
            if (d < best_d) { best_d = d; best_k = k; }
        }
        if (best_k >= 0)
            mCurrentFrame.mvpMapPoints[best_k] = valid_mps[idx];
    }

    // ── 6. POSE OPTIMIZATION (refine với BA nhỏ) ────────────────────────────
    Optimizer::PoseOptimization(&mCurrentFrame);

    int n_inliers_final = 0;
    for (int i = 0; i < mCurrentFrame.N; ++i)
    {
        if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
            ++n_inliers_final;
    }

    std::cout << "[LK] Final inliers after PoseOpt: " << n_inliers_final << "\n";

    if (n_inliers_final < 6)
        return false;

    // ── 7. CẬP NHẬT mvPrevPts/mvPrevMPs + mPrevGray cho frame tiếp theo ───────
    // mPrevGray cũng phải được cập nhật ở đây: nếu LK được gọi liên tiếp nhiều frame,
    // mà mPrevGray không update thì frame N+2 sẽ track từ ảnh frame N-1 (stale) → drift
    mPrevGray = mImGray;
    mvPrevPts.clear();
    mvPrevMPs.clear();
    for (int i = 0; i < mCurrentFrame.N; ++i)
    {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if (pMP && !mCurrentFrame.mvbOutlier[i] && !pMP->isBad())
        {
            mvPrevPts.push_back(mCurrentFrame.mvKeysUn[i].pt);
            mvPrevMPs.push_back(pMP);
        }
    }

    mnLKFallbackFrames++;
    return true;
}

// =============================================================================
// DYNAMIC FEATURE SCORING
// =============================================================================
// Gán dynamic score cho mỗi feature của mCurrentFrame.
// Score = max(semantic, flow, temporal) trong [0, 1].
//   semantic:  nằm trong person bbox → penalty cao
//   flow:      LK error cao → penalty
//   temporal:  MapPoint hay bị outlier → penalty
// Feature có score > kDynScoreRejectThresh → đánh dấu outlier.
// Cách này "mềm" hơn hard masking:
//   - Feature nằm trong bbox NHƯNG ổn định nhiều frame → giữ lại
//   - Feature ngoài bbox NHƯNG LK error cao → vẫn loại được
// =============================================================================
void Tracking::ComputeDynamicScores()
{
    const int N = mCurrentFrame.N;
    mDynScores.assign(N, 0.0f);

    // Không cần làm gì nếu không dùng mask và không có LK errors
    if (!mbUseMask && mvLKErrors.empty())
        return;

    // Tính bounding boxes người một lần
    struct PersonBBox { float x1, y1, x2, y2; };
    std::vector<PersonBBox> person_bboxes;
    if (mbUseMask)
    {
        for (const auto& det : current_frame_good_detections_)
        {
            if (det->category_id != PERSON_CLASS_ID) continue;
            person_bboxes.push_back({
                det->bbox[0], det->bbox[1],
                det->bbox[2], det->bbox[3]
            });
        }
    }

    // Build LK error lookup: MapPoint* → normalized error
    // Chỉ có nếu LK bridge đã chạy
    float max_lk_err = 1.0f;
    for (float e : mvLKErrors)
        if (e < 500.f) max_lk_err = std::max(max_lk_err, e);

    std::unordered_map<MapPoint*, float> lk_error_map;
    if (!mvLKErrors.empty() && !mvPrevMPs.empty())
    {
        size_t lk_n = std::min(mvPrevMPs.size(), mvLKErrors.size());
        for (size_t i = 0; i < lk_n; ++i)
        {
            if (mvPrevMPs[i] && mvLKErrors[i] < 500.f)
                lk_error_map[mvPrevMPs[i]] = mvLKErrors[i] / max_lk_err;
        }
    }

    int n_rejected = 0;

    for (int i = 0; i < N; ++i)
    {
        const cv::Point2f& pt = mCurrentFrame.mvKeysUn[i].pt;
        float score = 0.0f;

        // ── A. SEMANTIC SCORE (person bbox) ─────────────────────────────────
        // Trong center bbox: penalty cao. Near border: penalty thấp hơn.
        for (const auto& bb : person_bboxes)
        {
            if (pt.x < bb.x1 || pt.x > bb.x2 ||
                pt.y < bb.y1 || pt.y > bb.y2) continue;

            // Normalize vị trí trong bbox → score [0.3, 0.8]
            float bw = bb.x2 - bb.x1;
            float bh = bb.y2 - bb.y1;
            float rel_x = (pt.x - bb.x1) / std::max(bw, 1.f);
            float rel_y = (pt.y - bb.y1) / std::max(bh, 1.f);
            // Center nhận penalty cao hơn border
            float cx_dist = std::abs(rel_x - 0.5f) * 2.f; // 0=center, 1=edge
            float cy_dist = std::abs(rel_y - 0.5f) * 2.f;
            float border_factor = std::min(cx_dist, cy_dist); // 0=center, 1=edge
            float sem = 0.8f - 0.5f * border_factor; // center→0.8, edge→0.3
            score = std::max(score, sem);
            break;
        }

        // ── B. FLOW INSTABILITY SCORE (LK error) ────────────────────────────
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if (pMP)
        {
            auto it = lk_error_map.find(pMP);
            if (it != lk_error_map.end())
            {
                float flow_score = it->second * 0.5f; // LK error → [0, 0.5]
                score = std::max(score, flow_score);
            }
        }

        // ── C. TEMPORAL INSTABILITY (MapPoint observation health) ───────────
        // MapPoint với obs thấp + đã từng bị outlier nhiều → không đáng tin
        if (pMP && !pMP->isBad())
        {
            int obs = pMP->Observations();
            float found_ratio = pMP->GetFoundRatio();
            // found_ratio thấp = point này hay bị miss → instability cao
            if (obs > 0 && found_ratio < 0.5f)
            {
                float temporal = (1.0f - found_ratio) * 0.4f; // max 0.4
                score = std::max(score, temporal);
            }
        }

        mDynScores[i] = score;

        // ── REJECT nếu score cao, nhưng bảo vệ điểm đã ổn định nhiều frame ─
        if (score > kDynScoreRejectThresh)
        {
            // Điểm đã được bundle-adjusted nhiều lần: giữ lại dù score cao
            // (ví dụ: trần nhà xuất hiện đúng bbox người do góc nhìn)
            if (pMP && pMP->Observations() >= 5)
                continue;

            mCurrentFrame.mvbOutlier[i] = true;
            if (pMP) mCurrentFrame.mvpMapPoints[i] = nullptr;
            ++n_rejected;
        }
    }

    if (n_rejected > 0)
        std::cout << "[DynScore] Rejected " << n_rejected << "/" << N
                  << " features (thresh=" << kDynScoreRejectThresh << ")\n";
}

} //namespace ORB_SLAM2