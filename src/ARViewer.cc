/**
 * This file is part of OA-SLAM.
 *
 * Copyright (C) 2022 Matthieu Zins <matthieu.zins@inria.fr>
 * (Inria, LORIA, Université de Lorraine)
 */

#include "ARViewer.h"
#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include "shader.h"
#include "rendertree.h"
#include "ColorManager.h"
#include "Ellipsoid.h"

#include <mutex>
#include <future>
#include <unistd.h>

namespace ORB_SLAM2
{

ARViewer::ARViewer(System* pSystem, FrameDrawer *pFrameDrawer, Map *pMap, Tracking *pTracking, 
                   const std::string &strSettingPath)
    : mpSystem(pSystem), 
      mpFrameDrawer(pFrameDrawer), 
      mpMap(pMap), 
      mpTracker(pTracking),
      mbFinishRequested(false), 
      mbFinished(true), 
      mbStopped(true), 
      mbStopRequested(false), 
      mbPaused(false)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    float fps = fSettings["Camera.fps"];
    if (fps < 1) fps = 30;
    mT = 1e3 / fps;

    w_ = fSettings["Camera.width"];
    h_ = fSettings["Camera.height"];
    if (w_ < 1 || h_ < 1)
    {
        w_ = 640;
        h_ = 480;
    }

    mViewpointX = fSettings["Viewer.ViewpointX"];
    mViewpointY = fSettings["Viewer.ViewpointY"];
    mViewpointZ = fSettings["Viewer.ViewpointZ"];
    mViewpointF = fSettings["Viewer.ViewpointF"];

    fx_ = fSettings["Camera.fx"];
    fy_ = fSettings["Camera.fy"];
    cx_ = fSettings["Camera.cx"];
    cy_ = fSettings["Camera.cy"];

    K_ << 2 * fx_ / w_,  0,   (w_ - 2 * cx_) / w_, 0, 
          0, 2 * fy_ / h_, (-h_ + 2 * cy_) / h_, 0,
          0, 0, (-zfar_ - znear_)/(zfar_ - znear_), -2 * zfar_ * znear_ / (zfar_ - znear_),
          0, 0, -1, 0;
}

void ARViewer::UpdateFrame(cv::Mat img)
{
    img.copyTo(frame_);
}

// ====================== RENDER HELPER ======================
void RenderToViewportFlipY(pangolin::GlTexture& tex)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    float d = 0.99999;

    GLfloat sq_vert[] = { -1,-1, d, 1,-1, d,  1, 1, d,  -1, 1, d };
    glVertexPointer(3, GL_FLOAT, 0, sq_vert);
    glEnableClientState(GL_VERTEX_ARRAY);

    GLfloat sq_tex[]  = { 0,1,  1,1,  1,0,  0,0  };
    glTexCoordPointer(2, GL_FLOAT, 0, sq_tex);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glEnable(GL_TEXTURE_2D);
    tex.Bind();
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisable(GL_TEXTURE_2D);
}

// ====================== DRAW FUNCTIONS ======================
void ARViewer::DrawMapObjects()
{
    const std::vector<MapObject*> objects = mpMap->GetAllMapObjects();

    glPointSize(1);
    glLineWidth(2);

    for (auto *obj : objects) 
    {
        cv::Scalar c = obj->GetTrack()->GetColor();
        glColor3f(static_cast<double>(c(2)) / 255,
                  static_cast<double>(c(1)) / 255,
                  static_cast<double>(c(0)) / 255);

        const Ellipsoid& ell = obj->GetEllipsoid();
        auto pts = ell.GeneratePointCloud();
        int i = 0;
        while (i < pts.rows()) 
        {
            glBegin(GL_LINE_STRIP);
            for (int k = 0; k < 50; ++k, ++i)
            {
                glVertex3f(pts(i, 0), pts(i, 1), pts(i, 2));
            }
            glEnd();
        }
    }
}

void ARViewer::DrawLines() 
{
    glLineWidth(1);
    for (const auto& l : lines_) 
    {
        const auto& a = l.first;
        const auto& b = l.second;
        pangolin::glDrawLine(a[0], a[1], a[2], b[0], b[1], b[2]);
    }
}

// ====================== MAIN RUN FUNCTION ======================
void ARViewer::Run()
{
    std::cout << "Run AR Viewer" << std::endl;
    mbFinished = false;
    mbStopped = false;

    double w = w_;
    double h = h_;

    pangolin::CreateWindowAndBind("OA-SLAM: AR Viewer", w, h);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlMatrix K(K_);
    pangolin::OpenGlMatrix Rt_init(Rt_);

    pangolin::OpenGlRenderState s_cam(K, Rt_init);

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -w / h)
        .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::GlTexture bg_tex_(w_, h_, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

    // Tắt mesh để tránh lỗi attribute nx/ny/nz
    disp_mesh_ = false;
    std::cout << "[ARViewer] Mesh rendering is DISABLED to avoid shader attribute error." << std::endl;

    // ====================== gl_to_cv MATRIX ======================
    Eigen::Matrix4d gl_to_cv = Eigen::Matrix4d::Identity();
    gl_to_cv <<  1,  0,  0,  0,
                 0, -1,  0,  0,
                 0,  0, -1,  0,
                 0,  0,  0,  1;

    // ====================== MAIN LOOP ======================
    while (!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (frame_.cols == static_cast<int>(w_) && frame_.rows == static_cast<int>(h_))
        {
            bg_tex_.Upload(frame_.data, GL_BGR, GL_UNSIGNED_BYTE);

            // Build scene from MapObjects (chỉ vẽ ellipsoid)
            const std::vector<MapObject*> objects = mpMap->GetAllMapObjects();
            RenderNode root;  // không dùng mesh nữa
            root.edges.clear();

            pangolin::OpenGlMatrix Rt_gl(Rt_);
            s_cam.SetModelViewMatrix(gl_to_cv * Rt_gl);

            if (d_cam.IsShown())
            {
                d_cam.Activate(s_cam);

                // Chỉ vẽ ellipsoid (không vẽ mesh)
                DrawMapObjects();

                // Vẽ background image
                glColor3f(1.0f, 1.0f, 1.0f);
                RenderToViewportFlipY(bg_tex_);

                glDisable(GL_CULL_FACE);
            }
        }

        pangolin::FinishFrame();

        if (Stop())
        {
            while (isStopped())
                usleep(3000);
        }

        if (CheckFinish())
            break;
    }

    SetFinish();
}
// ====================== CONTROL FUNCTIONS ======================

void ARViewer::RequestFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool ARViewer::CheckFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void ARViewer::SetFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool ARViewer::isFinished()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinished;
}

void ARViewer::RequestStop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    if (!mbStopped)
        mbStopRequested = true;
}

bool ARViewer::Stop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    std::unique_lock<std::mutex> lock2(mMutexFinish);

    if (mbFinishRequested)
        return false;
    else if (mbStopRequested)
    {
        mbStopped = true;
        mbStopRequested = false;
        return true;
    }
    return false;
}

bool ARViewer::isStopped()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mbStopped;
}

void ARViewer::Release()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    mbStopped = false;
}

bool ARViewer::isPaused()
{
    return mbPaused;
}

void ARViewer::SetProperties(bool disp_mesh, bool fix_size)
{
    disp_mesh_ = disp_mesh;
    fix_mesh_size_ = fix_size;
}

void ARViewer::SetCurrentCameraPose(const Matrix34d& Rt)
{
    std::unique_lock<std::mutex> lock(mMutexCamera);
    Rt_.block<3, 4>(0, 0) = Rt;
}

} // namespace ORB_SLAM2
