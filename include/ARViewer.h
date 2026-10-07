/**
 * This file is part of OA-SLAM.
 *
 * Copyright (C) 2022 Matthieu Zins <matthieu.zins@inria.fr>
 * (Inria, LORIA, Université de Lorraine)
 */

#ifndef ARVIEWER_H
#define ARVIEWER_H

#include "System.h"
#include "FrameDrawer.h"
#include "Map.h"
#include "Tracking.h"

#include <mutex>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <unordered_map>

namespace ORB_SLAM2
{

class ARViewer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    ARViewer(System* pSystem, FrameDrawer* pFrameDrawer, Map* pMap, 
             Tracking* pTracking, const std::string &strSettingPath);

    void Run();
    void UpdateFrame(cv::Mat img);

    // Control functions - PHẢI CÓ để linker không báo lỗi
    void RequestFinish();
    bool CheckFinish();
    void SetFinish();

    void RequestStop();
    bool Stop();
    bool isStopped();
    void Release();

    bool isFinished();
    bool isPaused();

    void SetProperties(bool disp_mesh, bool fix_size);

    void SetCurrentCameraPose(const Matrix34d& Rt);
    void DrawMapObjects();
    void DrawLines();

    void AddModel(const std::string& model, const Eigen::Matrix4d& Twm);

private:
    System* mpSystem;
    FrameDrawer* mpFrameDrawer;
    Map* mpMap;
    Tracking* mpTracker;

    // Timing
    double mT;

    // Camera parameters
    float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;
    double fx_, fy_, cx_, cy_;
    double w_, h_;
    double znear_ = 0.01;
    double zfar_  = 10000.0;

    Eigen::Matrix4d K_ = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d Rt_ = Eigen::Matrix4d::Identity();
    std::mutex mMutexCamera;

    cv::Mat frame_;

    // Control flags
    bool mbFinishRequested = false;
    bool mbFinished = true;
    bool mbStopped = true;
    bool mbStopRequested = false;
    bool mbPaused = false;

    std::mutex mMutexFinish;
    std::mutex mMutexStop;

    // AR settings
    bool disp_mesh_ = true;
    bool fix_mesh_size_ = true;
    bool use_class_col_ = false;
    bool force_pause_ = false;

    // Models and objects
    std::mutex mMutexModels;
    std::vector<std::string> models_;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> models_transf_;
    std::vector<bool> models_loaded_;
    std::unordered_map<MapObject*, double> size_by_object_;

    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> lines_;
};

} // namespace ORB_SLAM2

#endif // ARVIEWER_H
