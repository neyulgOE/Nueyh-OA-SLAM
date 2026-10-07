/**
 * OA-SLAM Tracking V1
 *
 * Base: Tracking_v0.cc (ORB-SLAM2 / OA-SLAM original)
 *
 * V1 additions over V0:
 *   1. Dynamic mask — fix: all detections, kernel 5×5 (not 13×13)
 *   2. Adaptive CLAHE — texture score drives clip limit
 *   3. TrackLocalMap — proper mask filter protecting ceiling MPs,
 *                       flexible inlier threshold under occlusion
 *   4. SearchLocalPoints — adaptive search radius when occluded
 *   5. TrackReferenceKeyFrameMultiSeed — large-rotation recovery
 *   6. TrackWithOpticalFlow — LK bridge, MapPoints only, no AI depth
 *   7. Track() — velocity sanity check, LK fallback wiring,
 *                multi-seed wiring, conservative reset condition
 *   8. NeedNewKeyFrame — motion guard (mono), velocity sanity
 *   9. LK seed update at end of every frame
 *
 * Deliberately NOT ported from the experimental branch:
 *   PseudoDepth (thread, map, injection, scale recovery)
 *   Inertia mode
 *   Dynamic feature scoring
 *   SoftReset
 */

#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "Map.h"
#include "Initializer.h"
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
#include <dlib/optimization/max_cost_assignment.h>
#include <chrono>
#include <thread>
#include <omp.h>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

// ---------------------------------------------------------------------------
// Helper: is a 2-D point inside a non-zero mask?
// ---------------------------------------------------------------------------
static bool is_inside_mask(float x, float y, const cv::Mat& mask)
{
    int ix = static_cast<int>(std::round(x));
    int iy = static_cast<int>(std::round(y));
    if (ix < 0 || iy < 0 || ix >= mask.cols || iy >= mask.rows)
        return false;
    return mask.at<uchar>(iy, ix) > 0;
}

namespace ORB_SLAM2
{

// ===========================================================================
// BuildPersonMask
// ===========================================================================
// Creates a CV_8UC1 mask: 255 = usable region, 0 = person region.
// Uses all person detections (V0 bug: only used detections[0]).
// kernel_size 5 instead of 13 — avoids eating into ceiling features.
// ===========================================================================
cv::Mat Tracking::BuildPersonMask(const cv::Size& sz, int kernel_size) const
{
    cv::Mat mask(sz, CV_8UC1, cv::Scalar(255));

    for (const auto& det : current_frame_good_detections_)
    {
        if (det->category_id != PERSON_CLASS_ID) continue;

        if (!det->mask_no_person.empty())
        {
            mask.setTo(0, det->mask_no_person);
        }
        else if (!det->mask_object.empty())
        {
            mask.setTo(0, det->mask_object);
        }
        else
        {
            // Exact bbox — no expansion (expansion ate into non-person regions)
            float xmin = det->bbox[0], ymin = det->bbox[1];
            float xmax = det->bbox[2], ymax = det->bbox[3];
            cv::Rect r(
                std::max(0, int(xmin)),
                std::max(0, int(ymin)),
                std::min(int(xmax - xmin), sz.width  - std::max(0, int(xmin))),
                std::min(int(ymax - ymin), sz.height - std::max(0, int(ymin)))
            );
            if (r.width > 0 && r.height > 0)
                mask(r).setTo(0);
        }
    }

    if (kernel_size > 0)
    {
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
        cv::erode(mask, mask, kernel, cv::Point(-1,-1), 1);
    }

    return mask;
}

// ===========================================================================
// Constructor
// ===========================================================================
Tracking::Tracking(System* pSys, ORBVocabulary* pVoc,
                   FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer,
                   Map* pMap, KeyFrameDatabase* pKFDB,
                   const string& strSettingPath, const int sensor)
    : mState(NO_IMAGES_YET), mSensor(sensor),
      mbOnlyTracking(false), mbVO(false),
      mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB),
      mpInitializer(static_cast<Initializer*>(NULL)),
      mpSystem(pSys), mpViewer(NULL),
      mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer),
      mpMap(pMap), mnLastRelocFrameId(0),
      mbUseMask(true)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);
    K_ = cvToEigenMatrix<double, float, 3, 3>(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0) { DistCoef.resize(5); DistCoef.at<float>(4) = k3; }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if (fps == 0) fps = 30;

    // V1: mono uses mMinFrames=2 to avoid back-to-back KF spam while still
    // inserting quickly after a turn.  Stereo/RGBD keep 0 (V0 default).
    mMinFrames = (sensor == System::MONOCULAR) ? 2 : 0;
    mMaxFrames = static_cast<int>(fps);

    cout << "\nCamera Parameters:\n"
         << "- fx: " << fx  << "\n- fy: " << fy
         << "\n- cx: " << cx << "\n- cy: " << cy
         << "\n- k1: " << DistCoef.at<float>(0)
         << "\n- k2: " << DistCoef.at<float>(1);
    if (DistCoef.rows == 5)
        cout << "\n- k3: " << DistCoef.at<float>(4);
    cout << "\n- p1: " << DistCoef.at<float>(2)
         << "\n- p2: " << DistCoef.at<float>(3)
         << "\n- fps: " << fps << "\n";

    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;
    cout << "- color order: " << (mbRGB ? "RGB" : "BGR") << "\n";

    int   nFeatures    = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int   nLevels      = fSettings["ORBextractor.nLevels"];
    int   fIniThFAST   = fSettings["ORBextractor.iniThFAST"];
    int   fMinThFAST   = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    if (sensor == System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    if (sensor == System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    // V1.2: low-texture extractor — floor 5/3 để tránh threshold=0 (FAST sẽ
    // trả về nhiễu thuần, không phải corner thật). Mono only: đây là trường
    // hợp camera cầm tay quay vào trần/tường trắng giữa góc cua.
    if (sensor == System::MONOCULAR)
    {
        int fIniThFASTLow = std::max(5, fIniThFAST / 2);
        int fMinThFASTLow = std::max(3, fMinThFAST / 2);
        mpORBextractorLowTexture = new ORBextractor(nFeatures, fScaleFactor, nLevels,
                                                     fIniThFASTLow, fMinThFASTLow);
        std::cout << "[LowTextureExtractor] iniTh=" << fIniThFASTLow
                  << " minTh=" << fMinThFASTLow << "\n";
    }

    cout << "\nORB Extractor Parameters:\n"
         << "- Number of Features: " << nFeatures
         << "\n- Scale Levels: "     << nLevels
         << "\n- Scale Factor: "     << fScaleFactor
         << "\n- Initial Fast Threshold: " << fIniThFAST
         << "\n- Minimum Fast Threshold: " << fMinThFAST << "\n";

    if (sensor == System::STEREO || sensor == System::RGBD)
    {
        mThDepth = mbf * (float)fSettings["ThDepth"] / fx;
        cout << "\nDepth Threshold (Close/Far Points): " << mThDepth << "\n";
    }

    if (sensor == System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if (fabs(mDepthMapFactor) < 1e-5) mDepthMapFactor = 1;
        else                               mDepthMapFactor = 1.0f / mDepthMapFactor;
    }

    // V1.1: load configurable quality thresholds (defaults = original V1 behaviour)
    LoadQualityThresholds(strSettingPath);

    // V1.6: launch the persistent background LOST-recovery worker.
    StartRecoveryThread();
}

// ===========================================================================
// Destructor  — V1.6: stop + join the background recovery thread
// ===========================================================================
Tracking::~Tracking()
{
    mbStopRecoveryThread.store(true);
    mbRecoveryHasWork.store(true); // wake the thread up if it's waiting
    mRecoveryCondVar.notify_all();
    if (mRecoveryThread.joinable())
        mRecoveryThread.join();
}

// ===========================================================================
// LoadQualityThresholds  — V1.1 addition (ported from v0.1)
// ===========================================================================
void Tracking::LoadQualityThresholds(const string& strSettingPath)
{
    cv::FileStorage fs(strSettingPath, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cout << "[QualityThresholds] Cannot open settings file, using defaults.\n";
        return;
    }
    auto readInt = [&fs](const char* key, int& target) {
        if (!fs[key].empty()) target = (int)fs[key];
    };
    readInt("Tracking.RelocMinGoodInliers",     mRelocMinGoodInliers);
    readInt("Tracking.KFMinInliersNoOcclusion", mKFMinInliersNoOcclusion);
    readInt("Tracking.KFMinInliersOcclusion",   mKFMinInliersOcclusion);
    readInt("Tracking.ProjRelocMinInliers",     mProjRelocMinInliers);

    std::cout << "[QualityThresholds] RelocMinGood=" << mRelocMinGoodInliers
              << " KFMinNoOcc=" << mKFMinInliersNoOcclusion
              << " KFMinOcc="   << mKFMinInliersOcclusion
              << " ProjRelocMinInliers=" << mProjRelocMinInliers << "\n";
}

// ===========================================================================
// StartRecoveryThread  — V1.6 addition
// ===========================================================================
void Tracking::StartRecoveryThread()
{
    mbStopRecoveryThread.store(false);
    mbRecoveryHasWork.store(false);
    mbRecoveryThreadBusy.store(false);
    mRecoveryThread = std::thread(&Tracking::RecoveryThreadLoop, this);
    std::cout << "[RecoveryThread] Background LOST-recovery worker started.\n";
}

// ===========================================================================
// OfferRecoverySnapshotIfIdle  — V1.6 addition (called from main thread)
// ===========================================================================
// Non-blocking. If the background worker is currently busy with a previous
// attempt, this does nothing — the new snapshot is simply dropped. This IS
// the throttle: at most one recovery attempt is ever in flight, so the main
// thread never piles up backlog and never waits on the heavy chain.
// ===========================================================================
void Tracking::OfferRecoverySnapshotIfIdle()
{
    if (mbRecoveryThreadBusy.load())
        return; // worker still chewing on the previous snapshot — drop this one

    if (mImGray.empty())
        return;

    {
        std::lock_guard<std::mutex> lock(mRecoverySnapshotMutex);
        mRecoverySnapshotGray            = mImGray.clone();
        mRecoverySnapshotGoodDetections  = current_frame_good_detections_;
        // V1.6 fix: use current_frame_idx_ (Tracking's own monotonic counter,
        // the same one printed as "Frame N ===========" in the logs), NOT
        // mCurrentFrame.mnId. mCurrentFrame.mnId comes from Frame::nNextId,
        // a counter shared with EVERY Frame ever constructed — including the
        // standalone Frame the background thread builds for itself. Comparing
        // against that made staleness checks meaningless (e.g. logs showing
        // "frame 268" while the main thread was displaying "Frame 256").
        // current_frame_idx_ is private to Tracking and only ever advanced by
        // the main thread, so it's the correct, race-free reference for "how
        // many main-thread frames have passed since this snapshot was taken".
        mRecoverySnapshotFrameId         = current_frame_idx_;
    }
    mbRecoveryHasWork.store(true);
    mRecoveryCondVar.notify_one();
}

// ===========================================================================
// ConsumePublishedRecoveryResultIfAny  — V1.6 addition (called from main thread)
// ===========================================================================
// Non-blocking. Picks up whatever the background worker last published (if
// anything new), and — only if still relevant — applies it to mCurrentFrame
// / mState right here in the main thread. This is the ONLY place a background
// recovery result is allowed to touch tracking state, so all of Track()'s
// existing single-threaded invariants on Frame/Map are preserved.
// ===========================================================================
bool Tracking::ConsumePublishedRecoveryResultIfAny()
{
    bool ready = false, success = false;
    cv::Mat tcw;
    std::vector<MapPoint*> mps;
    size_t resultFrameId = 0;
    KeyFrame* refKF = nullptr;

    {
        std::lock_guard<std::mutex> lock(mRecoveryResultMutex);
        if (!mbRecoveryResultReady) return false;
        ready          = true;
        success        = mbRecoveryResultSuccess;
        tcw            = mRecoveryResultTcw;
        mps            = mRecoveryResultMapPoints;
        resultFrameId  = mRecoveryResultFrameId;
        refKF          = mRecoveryResultRefKF;
        mbRecoveryResultReady = false; // consumed
    }

    if (!ready || !success)
        return false;

    // Stale check: the background attempt was seeded from an older frame's
    // image. If the camera has moved a lot since, or many frames have passed,
    // applying this pose could be worse than just letting Inertia/Quick-covis
    // keep coasting. A few frames of staleness is fine (recovery usually takes
    // 1-3 frames longer than the snapshot anyway); a large gap is suspect.
    //
    // V1.6 fix: compare against current_frame_idx_ (Tracking's own monotonic
    // counter — the same one printed as "Frame N ===========" in the logs),
    // NOT mCurrentFrame.mnId. The latter is a Frame-object id drawn from the
    // globally-shared Frame::nNextId counter, which the background thread's
    // own standalone Frame construction also advances — comparing against it
    // produced nonsensical ages (e.g. a "frame 268" snapshot appearing while
    // the main thread was only at "Frame 256").
    const size_t age = (current_frame_idx_ > resultFrameId)
                            ? (current_frame_idx_ - resultFrameId) : 0;
    if (age > 30)
    {
        std::cout << "[RecoveryThread] Discarding stale result (age=" << age
                  << " frames) — re-arming for a fresh attempt.\n";
        return false;
    }

    mCurrentFrame.SetPose(tcw);
    if (mps.size() == (size_t)mCurrentFrame.N)
        mCurrentFrame.mvpMapPoints = mps;
    fill(mCurrentFrame.mvbOutlier.begin(), mCurrentFrame.mvbOutlier.end(), false);

    if (refKF && !refKF->isBad())
    {
        mpReferenceKF = refKF;
        mCurrentFrame.mpReferenceKF = refKF;
    }

    mnLastRelocFrameId    = mCurrentFrame.mnId;
    mnPostResetLostFrames = 0; // V1.5 watchdog: recovered, clear it

    std::cout << "[RecoveryThread] Applied background-recovered pose "
              << "(snapshot age=" << age << " frames).\n";
    return true;
}

// ===========================================================================
// RecoveryThreadLoop  — V1.6 addition (runs on mRecoveryThread)
// ===========================================================================
// Owns its own Frame copy at all times — never touches mCurrentFrame, never
// touches current_frame_good_detections_/current_frame_idx_, never takes
// mpMap->mMutexMapUpdate for writing (PoseOptimization on a local Frame only
// reads MapPoint positions and calls MapPoint's own internally-locked
// accessors — same pattern already used safely by the existing multi-seed /
// projection-reloc functions, just now off the main thread).
// ===========================================================================
void Tracking::RecoveryThreadLoop()
{
    using namespace std::chrono;

    while (!mbStopRecoveryThread.load())
    {
        {
            std::unique_lock<std::mutex> lock(mRecoverySnapshotMutex);
            mRecoveryCondVar.wait(lock, [this]{
                return mbRecoveryHasWork.load() || mbStopRecoveryThread.load();
            });
        }

        if (mbStopRecoveryThread.load())
            break;

        mbRecoveryHasWork.store(false);
        mbRecoveryThreadBusy.store(true);
        auto t_start = steady_clock::now();

        // ── Pull snapshot ────────────────────────────────────────────────────
        cv::Mat gray;
        std::vector<Detection::Ptr> goodDets;
        size_t frameId = 0;
        {
            std::lock_guard<std::mutex> lock(mRecoverySnapshotMutex);
            gray     = mRecoverySnapshotGray.clone();
            goodDets = mRecoverySnapshotGoodDetections;
            frameId  = mRecoverySnapshotFrameId;
        }

        bool   success = false;
        cv::Mat resultTcw;
        std::vector<MapPoint*> resultMPs;
        KeyFrame* resultRefKF = nullptr;

        if (!gray.empty() && mpORBextractorLeft)
        {
            // Build an exclusion mask locally from the snapshot's own
            // detections — does NOT call BuildPersonMask() (that reads the
            // shared current_frame_good_detections_ member, which the main
            // thread mutates every frame; reading it here would race).
            cv::Mat localMask(gray.size(), CV_8UC1, cv::Scalar(255));
            for (const auto& det : goodDets)
            {
                if (det->category_id != PERSON_CLASS_ID) continue;
                if (!det->mask_no_person.empty())
                    localMask.setTo(0, det->mask_no_person);
                else if (!det->mask_object.empty())
                    localMask.setTo(0, det->mask_object);
                else
                {
                    float xmin = det->bbox[0], ymin = det->bbox[1];
                    float xmax = det->bbox[2], ymax = det->bbox[3];
                    cv::Rect r(std::max(0, int(xmin)), std::max(0, int(ymin)),
                               std::min(int(xmax-xmin), localMask.cols - std::max(0,int(xmin))),
                               std::min(int(ymax-ymin), localMask.rows - std::max(0,int(ymin))));
                    if (r.width > 0 && r.height > 0) localMask(r).setTo(0);
                }
            }
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5));
            cv::erode(localMask, localMask, kernel, cv::Point(-1,-1), 1);

            // Build a standalone Frame — independent of mCurrentFrame.
            // Note: Frame's static mbInitialComputations/grid-bounds setup only
            // runs on the very first Frame ever constructed in the process; by
            // the time tracking can be LOST, at least one frame has already
            // gone through StereoInitialization/MonocularInitialization on the
            // main thread, so that static state is already finalized and this
            Frame workFrame(gray, 0.0, mpORBextractorLeft, mpORBVocabulary,
                             mK, mDistCoef, mbf, mThDepth, localMask, true);

            if (workFrame.N > 0)
            {
                workFrame.ComputeBoW();

                // ── Stage 1: BoW-independent projection over ALL map KFs ────
                // Cheapest-of-the-heavy options and doesn't depend on BoW DB
                // scoring well, which is exactly the failure mode logged
                // (ProjReloc consistently short of threshold with the
                // covisibility-limited candidate set).
                std::vector<KeyFrame*> vpAllKFs = mpMap->GetAllKeyFrames();
                int bestInliers = 0;
                cv::Mat bestTcw;
                std::vector<MapPoint*> bestMPs;
                KeyFrame* bestKF = nullptr;
                ORBmatcher matcher(0.9, true);

                for (auto* pKF : vpAllKFs)
                {
                    if (!pKF || pKF->isBad()) continue;
                    workFrame.SetPose(pKF->GetPose());
                    fill(workFrame.mvpMapPoints.begin(), workFrame.mvpMapPoints.end(), nullptr);
                    fill(workFrame.mvbOutlier.begin(), workFrame.mvbOutlier.end(), false);

                    std::set<MapPoint*> sFound;
                    int nm = matcher.SearchByProjection(workFrame, pKF, sFound, 30, 100);
                    if (nm < 15) continue;

                    Optimizer::PoseOptimization(&workFrame);

                    int nIn = 0;
                    for (int i = 0; i < workFrame.N; i++)
                        if (workFrame.mvpMapPoints[i] && !workFrame.mvbOutlier[i]) nIn++;

                    if (nIn > bestInliers)
                    {
                        bestInliers = nIn;
                        bestTcw     = workFrame.mTcw.clone();
                        bestMPs     = workFrame.mvpMapPoints;
                        bestKF      = pKF;
                        if (nIn >= 50) break;
                    }
                }

                const int minInliers = std::max(15, mProjRelocMinInliers / 2);
                if (bestInliers >= minInliers && !bestTcw.empty())
                {
                    workFrame.SetPose(bestTcw);
                    workFrame.mvpMapPoints = bestMPs;
                    fill(workFrame.mvbOutlier.begin(), workFrame.mvbOutlier.end(), false);
                    Optimizer::PoseOptimization(&workFrame);

                    int nFinal = 0;
                    for (int i = 0; i < workFrame.N; i++)
                    {
                        if (!workFrame.mvpMapPoints[i]) continue;
                        if (workFrame.mvbOutlier[i]) workFrame.mvpMapPoints[i] = nullptr;
                        else nFinal++;
                    }

                    if (nFinal >= minInliers / 2)
                    {
                        success     = true;
                        resultTcw   = workFrame.mTcw.clone();
                        resultMPs   = workFrame.mvpMapPoints;
                        resultRefKF = bestKF;
                        std::cout << "[RecoveryThread] Stage1 ProjReloc OK via KF "
                                  << (bestKF ? bestKF->mnId : -1)
                                  << " inliers=" << bestInliers
                                  << " final=" << nFinal << "\n";
                    }
                }

                // ── Stage 2: full BoW Relocalization (only if stage 1 failed) ─
                // Heaviest step; only worth it if the cheap wide projection
                // scan above couldn't find the camera anywhere in the map.
                if (!success)
                {
                    std::vector<KeyFrame*> vpCandidateKFs =
                        mpKeyFrameDB->DetectRelocalizationCandidates(&workFrame);

                    if (!vpCandidateKFs.empty())
                    {
                        const int nKFs = (int)vpCandidateKFs.size();
                        ORBmatcher matcherBoW(0.75, true);
                        std::vector<std::vector<MapPoint*>> vvpMapPointMatches(nKFs);
                        std::vector<bool> vbDiscarded(nKFs, false);
                        std::vector<PnPsolver*> vpPnPsolvers(nKFs, nullptr);
                        int nCandidates = 0;

                        for (int i = 0; i < nKFs; i++)
                        {
                            KeyFrame* pKF = vpCandidateKFs[i];
                            if (pKF->isBad()) { vbDiscarded[i] = true; continue; }
                            int nmatches = matcherBoW.SearchByBoW(pKF, workFrame, vvpMapPointMatches[i]);
                            if (nmatches < 10) { vbDiscarded[i] = true; continue; }
                            PnPsolver* pSolver = new PnPsolver(workFrame, vvpMapPointMatches[i]);
                            pSolver->SetRansacParameters(0.99, 8, 500, 4, 0.5, 5.991);
                            vpPnPsolvers[i] = pSolver;
                            nCandidates++;
                        }

                        bool bMatch = false;
                        ORBmatcher matcher2(0.9, true);
                        const int minGood = std::max(15, mRelocMinGoodInliers / 2);

                        while (nCandidates > 0 && !bMatch)
                        {
                            for (int i = 0; i < nKFs && !bMatch; i++)
                            {
                                if (vbDiscarded[i]) continue;
                                std::vector<bool> vbInliers; int nInliers; bool bNoMore;
                                cv::Mat Tcw = vpPnPsolvers[i]->iterate(5, bNoMore, vbInliers, nInliers);
                                if (bNoMore) { vbDiscarded[i] = true; nCandidates--; }
                                if (Tcw.empty()) continue;

                                Tcw.copyTo(workFrame.mTcw);
                                workFrame.UpdatePoseMatrices();
                                std::set<MapPoint*> sFound;
                                for (int j = 0; j < (int)vbInliers.size(); j++)
                                {
                                    if (vbInliers[j]) {
                                        workFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                                        sFound.insert(vvpMapPointMatches[i][j]);
                                    } else workFrame.mvpMapPoints[j] = nullptr;
                                }

                                int nGood = Optimizer::PoseOptimization(&workFrame);
                                if (nGood < 10) continue;

                                for (int io = 0; io < workFrame.N; io++)
                                    if (workFrame.mvbOutlier[io]) workFrame.mvpMapPoints[io] = nullptr;

                                if (nGood < minGood)
                                {
                                    int nadditional = matcher2.SearchByProjection(
                                        workFrame, vpCandidateKFs[i], sFound, 10, 100);
                                    if (nadditional + nGood >= minGood)
                                        nGood = Optimizer::PoseOptimization(&workFrame);
                                }

                                if (nGood >= minGood)
                                {
                                    bMatch      = true;
                                    success     = true;
                                    resultTcw   = workFrame.mTcw.clone();
                                    resultMPs   = workFrame.mvpMapPoints;
                                    resultRefKF = vpCandidateKFs[i];
                                    std::cout << "[RecoveryThread] Stage2 BoW Relocalization OK, "
                                              << "nGood=" << nGood << "\n";
                                }
                            }
                        }

                        for (auto* s : vpPnPsolvers) delete s;
                    }
                }
            }
        }

        // ── Publish result ───────────────────────────────────────────────────
        {
            std::lock_guard<std::mutex> lock(mRecoveryResultMutex);
            mbRecoveryResultReady   = true;
            mbRecoveryResultSuccess = success;
            mRecoveryResultTcw      = resultTcw;
            mRecoveryResultMapPoints = resultMPs;
            mRecoveryResultFrameId  = frameId;
            mRecoveryResultRefKF   = resultRefKF;
        }

        if (!success)
            std::cout << "[RecoveryThread] Background attempt failed for frame "
                      << frameId << ".\n";

        // ── Throttle ──────────────────────────────────────────────────────────
        auto elapsed = duration_cast<milliseconds>(steady_clock::now() - t_start).count();
        if (elapsed < kRecoveryThreadMinIntervalMs)
            std::this_thread::sleep_for(milliseconds(kRecoveryThreadMinIntervalMs - elapsed));

        mbRecoveryThreadBusy.store(false);
    }
}

// ===========================================================================
// Setters
// ===========================================================================
void Tracking::SetLocalMapper(LocalMapping* pLocalMapper)       { mpLocalMapper = pLocalMapper; }
void Tracking::SetLocalObjectMapper(LocalObjectMapping* obj)    { local_object_mapper_ = obj; }
void Tracking::SetLoopClosing(LoopClosing* pLoopClosing)        { mpLoopClosing = pLoopClosing; }
void Tracking::SetViewer(Viewer* pViewer)                       { mpViewer = pViewer; }
void Tracking::SetARViewer(ARViewer* pARViewer)                 { mpARViewer = pARViewer; }

// ===========================================================================
// GrabImageStereo  (no detections — simple path)
// ===========================================================================
cv::Mat Tracking::GrabImageStereo(const cv::Mat& imRectLeft,
                                  const cv::Mat& imRectRight,
                                  const double& timestamp)
{
    mImGray = imRectLeft.clone();
    cv::Mat imGrayRight = imRectRight.clone();

    auto toGray = [&](cv::Mat& m) {
        if (m.channels() == 3)
            cvtColor(m, m, mbRGB ? CV_RGB2GRAY : CV_BGR2GRAY);
        else if (m.channels() == 4)
            cvtColor(m, m, mbRGB ? CV_RGBA2GRAY : CV_BGRA2GRAY);
    };
    toGray(mImGray);
    toGray(imGrayRight);

    cv::Mat exclude_mask;
    if (mbUseMask)
        exclude_mask = BuildPersonMask(mImGray.size(), 5);

    mCurrentFrame = Frame(mImGray, imGrayRight, timestamp,
                          mpORBextractorLeft, mpORBextractorRight,
                          mpORBVocabulary, mK, mDistCoef,
                          mbf, mThDepth, exclude_mask);
    Track();
    return mCurrentFrame.mTcw.clone();
}

// ===========================================================================
// GrabImageStereo_object  (with detections)
// ===========================================================================
cv::Mat Tracking::GrabImageStereo_object(const cv::Mat& imRectLeft,
                                         const cv::Mat& imRectRight,
                                         const double& timestamp,
                                         const std::vector<Detection::Ptr>& detections,
                                         bool force_relocalize)
{
    current_frame_idx_ = (current_frame_idx_ + 1) % (std::numeric_limits<size_t>::max() - 1);
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;
    imRectLeft.copyTo(im_rgb_);

    auto toGray = [&](cv::Mat& m) {
        if (m.channels() == 3)
            cvtColor(m, m, mbRGB ? CV_RGB2GRAY : CV_BGR2GRAY);
        else if (m.channels() == 4)
            cvtColor(m, m, mbRGB ? CV_RGBA2GRAY : CV_BGRA2GRAY);
    };
    toGray(mImGray);
    toGray(imGrayRight);

    // Filter detections
    current_frame_detections_ = detections;
    current_frame_good_detections_.clear();
    for (auto& det : current_frame_detections_)
        if (det->score > 0.5f)
            current_frame_good_detections_.push_back(det);

    // V1: build mask from ALL person detections, kernel=5
    cv::Mat exclude_mask;
    if (mbUseMask)
        exclude_mask = BuildPersonMask(mImGray.size(), 5);

    mCurrentFrame = Frame(mImGray, imGrayRight, timestamp,
                          mpORBextractorLeft, mpORBextractorRight,
                          mpORBVocabulary, mK, mDistCoef,
                          mbf, mThDepth, exclude_mask);

    if (force_relocalize)
    {
        auto t1 = high_resolution_clock::now();
        bool bOK = false;

        int relocMode = mpSystem->GetRelocalizationMode();
        if (relocMode == RELOC_OBJECTS || relocMode == RELOC_OBJECTS_POINTS)
            bOK = RelocalizationFromObjects(false);
        if (!bOK)
            bOK = Relocalization();
        if (!bOK && relocMode == RELOC_OBJECTS_POINTS)
            bOK = RelocalizationFromObjects(true);

        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms = t2 - t1;

        mpFrameDrawer->Update(this);
        if (bOK) mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
        mpSystem->relocalization_duration = ms.count();
        mpSystem->relocalization_status   = bOK;
    }
    else
    {
        Track();

        // ── Update mean depth ────────────────────────────────────────────────
        if (mState == Tracking::OK)
        {
            Matrix34d Rt = cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw);
            double z_mean = 0.0; unsigned int z_nb = 0;
            for (size_t i = 0; i < mCurrentFrame.mvpMapPoints.size(); i++) {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if (pMP && !mCurrentFrame.mvbOutlier[i]) {
                    cv::Mat mp = pMP->GetWorldPos();
                    Eigen::Vector4d p(mp.at<float>(0), mp.at<float>(1), mp.at<float>(2), 1.0);
                    z_mean += (Rt * p)[2]; z_nb++;
                }
            }
            if (z_nb > 0) current_mean_depth_ = z_mean / z_nb;
        }

        std::cout << "Frame " << current_frame_idx_ << " ===========\n"
                  << "Nb Object Tracks: " << objectTracks_.size() << "\n"
                  << "Nb Map Objects  : " << mpMap->GetNumberMapObjects() << "\n";

        // ── Object tracking (unchanged from V0) ──────────────────────────────
        const double MIN_2D_IOU_THRESH = 0.2;
        const double MIN_3D_IOU_THRESH = 0.3;
        const int    TIME_DIFF_THRESH  = 30;
        BBox2 img_bbox(0, 0, imRectLeft.cols, imRectLeft.rows);

        if (mState == Tracking::OK && !current_frame_good_detections_.empty())
        {
            KeyFrame* kf = createdNewKeyFrame_ ? mpLastKeyFrame : nullptr;

            Matrix34d Rt = cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw);
            Matrix34d P  = K_ * Rt;

            std::unordered_map<ObjectTrack::Ptr, BBox2> proj_bboxes;
            for (auto tr : objectTracks_)
            {
                if (tr->GetStatus() != ObjectTrackStatus::INITIALIZED &&
                    tr->GetStatus() != ObjectTrackStatus::IN_MAP) continue;
                MapObject* obj = tr->GetMapObject();
                Eigen::Vector3d c = obj->GetEllipsoid().GetCenter();
                double z = Rt.row(2).dot(c.homogeneous());
                auto ell = obj->GetEllipsoid().project(P);
                BBox2 bb = ell.ComputeBbox();
                if (bboxes_intersection(bb, img_bbox) < 0.3 * bbox_area(bb)) continue;
                proj_bboxes[tr] = bb;
                std::unordered_set<ObjectTrack::Ptr> hidden;
                for (auto it : proj_bboxes) {
                    if (it.first != tr && bboxes_iou(it.second, bb) > 0.9) {
                        Eigen::Vector3d c2 = it.first->GetMapObject()->GetEllipsoid().GetCenter();
                        double z2 = Rt.row(2).dot(c2.homogeneous());
                        hidden.insert(z < z2 ? it.first : tr);
                        break;
                    }
                }
                for (auto h : hidden) proj_bboxes.erase(h);
            }

            std::vector<ObjectTrack::Ptr> possible_tracks;
            for (auto tr : objectTracks_) {
                auto bb = tr->GetLastBbox();
                if (tr->GetLastObsFrameId() + 60 >= current_frame_idx_ &&
                    bboxes_intersection(bb, img_bbox) >= 0.3 * bbox_area(bb))
                    possible_tracks.push_back(tr);
                else if (proj_bboxes.count(tr))
                    possible_tracks.push_back(tr);
            }

            // Map point association (parallel)
            size_t nDet = current_frame_good_detections_.size();
            std::vector<std::unordered_set<MapPoint*>> assoc_map_points(nDet);
            std::vector<Eigen::Vector3d> mean_coords(nDet, Eigen::Vector3d::Zero());
            std::vector<int> point_counts(nDet, 0);
            Eigen::Matrix4d Tcw_eigen = cvToEigenMatrix<double, float, 4, 4>(mCurrentFrame.mTcw);

            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < nDet; ++i) {
                const auto& det = current_frame_good_detections_[i];
                if (det->mask_object.empty()) continue;
                cv::Rect roi = cv::boundingRect(det->mask_object);
                roi.x      = std::max(0, roi.x - 15);
                roi.y      = std::max(0, roi.y - 15);
                roi.width  = std::min(int(mImGray.cols) - roi.x, roi.width  + 30);
                roi.height = std::min(int(mImGray.rows) - roi.y, roi.height + 30);
                Eigen::Vector3d local_sum = Eigen::Vector3d::Zero();
                int local_count = 0;
                std::unordered_set<MapPoint*> local_assoc;
                for (size_t j = 0; j < mCurrentFrame.mvKeysUn.size(); ++j) {
                    if (!mCurrentFrame.mvpMapPoints[j]) continue;
                    const cv::Point2f& pt = mCurrentFrame.mvKeysUn[j].pt;
                    if (pt.x < roi.x || pt.x >= roi.x + roi.width ||
                        pt.y < roi.y || pt.y >= roi.y + roi.height) continue;
                    if (is_inside_mask(pt.x, pt.y, det->mask_object)) {
                        MapPoint* mp = mCurrentFrame.mvpMapPoints[j];
                        local_assoc.insert(mp);
                        cv::Mat pos_w = mp->GetWorldPos();
                        Eigen::Vector4d Pw(pos_w.at<float>(0), pos_w.at<float>(1), pos_w.at<float>(2), 1.0);
                        local_sum += (Tcw_eigen * Pw).head<3>();
                        ++local_count;
                    }
                }
                assoc_map_points[i] = std::move(local_assoc);
                if (local_count > 0) mean_coords[i] = local_sum / local_count;
                point_counts[i] = local_count;
            }

            for (size_t i = 0; i < nDet; ++i) {
                current_frame_good_detections_[i]->mean_coor = mean_coords[i];
                if (current_frame_idx_ % 5 == 0)
                    std::cout << "[Frame " << current_frame_idx_ << "] det " << i
                              << " pts=" << point_counts[i]
                              << " dist=" << mean_coords[i].norm() << "m\n";
            }

            // Hungarian matching
            const int THRESHOLD_NB_MATCH = 10;
            std::vector<int> matched_by_points(nDet, -1);
            std::vector<std::vector<size_t>> nb_matched_points(nDet);
            for (size_t i = 0; i < nDet; ++i) {
                int det_cat = current_frame_good_detections_[i]->category_id;
                size_t max_nb = 0, best_j = 0;
                for (size_t j = 0; j < possible_tracks.size(); ++j) {
                    auto tr_mps = possible_tracks[j]->GetAssociatedMapPoints();
                    size_t n = count_set_map_intersection(assoc_map_points[i], tr_mps);
                    if (n > max_nb) { max_nb = n; best_j = j; }
                    if (det_cat != (int)possible_tracks[j]->GetCategoryId()) n = 0;
                    nb_matched_points[i].push_back(n);
                }
                if ((int)max_nb > THRESHOLD_NB_MATCH &&
                    current_frame_good_detections_[i]->category_id ==
                    possible_tracks[best_j]->GetCategoryId())
                    matched_by_points[i] = best_j;
            }

            int m = std::max(possible_tracks.size(), nDet);
            dlib::matrix<long> cost = dlib::zeros_matrix<long>(m, m);
            std::vector<long> assignment(m, std::numeric_limits<long>::max());

            for (size_t di = 0; di < nDet; ++di) {
                auto det = current_frame_good_detections_[di];
                for (size_t ti = 0; ti < possible_tracks.size(); ++ti) {
                    auto tr = possible_tracks[ti];
                    if (tr->GetCategoryId() != det->category_id) continue;
                    double iou2d = 0, iou3d = 0;
                    if (tr->GetLastObsFrameId() + TIME_DIFF_THRESH >= current_frame_idx_)
                        iou2d = bboxes_iou(tr->GetLastBbox(), det->bbox);
                    if (proj_bboxes.count(tr))
                        iou3d = bboxes_iou(proj_bboxes[tr], det->bbox);
                    if (iou2d < MIN_2D_IOU_THRESH) iou2d = 0;
                    if (iou3d < MIN_3D_IOU_THRESH) iou3d = 0;
                    cost(di, ti) = std::max(iou2d, iou3d) * 1000;
                }
                if (matched_by_points[di] != -1)
                    cost(di, matched_by_points[di]) = std::numeric_limits<int>::max();
            }
            assignment = dlib::max_cost_assignment(cost);

            std::vector<ObjectTrack::Ptr> new_tracks;
            for (size_t di = 0; di < nDet; ++di) {
                auto det = current_frame_good_detections_[di];
                auto ati = assignment[di];
                if (ati >= (long)possible_tracks.size() || cost(di, ati) == 0) {
                    // V1.3: do NOT create a track for dynamic/moveable classes.
                    // Person, animals, vehicles, handheld items must never enter
                    // the map as objects — they would poison ObjectAnchor reloc.
                    if (!IsStaticAnchorClass(det->category_id)) continue;
                    auto tr = ObjectTrack::CreateNewObjectTrack(det->category_id, det->bbox,
                                    det->score, Rt, current_frame_idx_, this, kf);
                    new_tracks.push_back(tr);
                } else {
                    auto& atr = possible_tracks[ati];
                    atr->AddDetection(det->bbox, det->score, Rt, current_frame_idx_, kf);
                    if (kf && atr->GetStatus() == ObjectTrackStatus::IN_MAP && local_object_mapper_)
                        local_object_mapper_->InsertModifiedObject(atr->GetMapObject());
                }
            }
            for (auto tr : new_tracks) objectTracks_.push_back(tr);

            if (!mbOnlyTracking) {
                for (auto& tr : objectTracks_) {
                    // V1.3: extra guard — even if a dynamic-class track slipped in
                    // (from before this fix or from an existing saved map), prevent
                    // it from being reconstructed or inserted into the map.
                    if (!IsStaticAnchorClass(static_cast<int>(tr->GetCategoryId()))) {
                        if (tr->GetStatus() != ObjectTrackStatus::IN_MAP)
                            tr->SetIsBad();
                        continue;
                    }
                    if (tr->GetLastObsFrameId() == current_frame_idx_) {
                        if ((tr->GetNbObservations() > 10 && tr->GetStatus() == ObjectTrackStatus::ONLY_2D) ||
                            (tr->GetNbObservations() % 2 == 0 && tr->GetStatus() == ObjectTrackStatus::INITIALIZED)) {
                            bool ok = tr->ReconstructFromCenter();
                            if (ok) tr->OptimizeReconstruction(mpMap);
                        }
                    }
                    if (tr->GetNbObservations() >= 20 && tr->GetStatus() == ObjectTrackStatus::INITIALIZED) {
                        tr->OptimizeReconstruction(mpMap);
                        if (tr->CheckReprojectionIoU(0.3)) {
                            tr->InsertInMap(mpMap);
                            if (local_object_mapper_)
                                local_object_mapper_->InsertModifiedObject(tr->GetMapObject());
                        } else {
                            tr->SetIsBad();
                        }
                    }
                }
                for (auto tr : objectTracks_)
                    if ((int)tr->GetLastObsFrameId() < (int)current_frame_idx_ - TIME_DIFF_THRESH
                        && tr->GetStatus() != ObjectTrackStatus::IN_MAP)
                        tr->SetIsBad();
            }
        }

        mpFrameDrawer->Update(this);
    }

    // V1: seed LK bridge for next frame
    mImGray.copyTo(mPrevGray);
    if (mState == OK) {
        mvPrevPts.clear(); mvPrevMPs.clear();
        for (int i = 0; i < mCurrentFrame.N; ++i) {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if (pMP && !mCurrentFrame.mvbOutlier[i] && !pMP->isBad() && pMP->Observations() >= 2) {
                mvPrevPts.push_back(mCurrentFrame.mvKeysUn[i].pt);
                mvPrevMPs.push_back(pMP);
            }
        }
    }

    return mCurrentFrame.mTcw.clone();
}

// ===========================================================================
// GrabImageRGBD
// ===========================================================================
cv::Mat Tracking::GrabImageRGBD(const cv::Mat& imRGB, const cv::Mat& imD,
                                  const double& timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if (mImGray.channels() == 3)
        cvtColor(mImGray, mImGray, mbRGB ? CV_RGB2GRAY : CV_BGR2GRAY);
    else if (mImGray.channels() == 4)
        cvtColor(mImGray, mImGray, mbRGB ? CV_RGBA2GRAY : CV_BGRA2GRAY);

    if ((fabs(mDepthMapFactor-1.0f) > 1e-5) || imDepth.type() != CV_32F)
        imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

    mCurrentFrame = Frame(mImGray, imDepth, timestamp,
                          mpORBextractorLeft, mpORBVocabulary,
                          mK, mDistCoef, mbf, mThDepth, cv::Mat());
    Track();
    return mCurrentFrame.mTcw.clone();
}

// ===========================================================================
// GrabImageMonocular
// ===========================================================================
cv::Mat Tracking::GrabImageMonocular(const cv::Mat& im, const double& timestamp,
                                      const std::vector<Detection::Ptr>& detections,
                                      bool force_relocalize)
{
    current_frame_idx_ = (current_frame_idx_ + 1) % (std::numeric_limits<size_t>::max() - 1);
    mImGray = im;
    im.copyTo(im_rgb_);

    // Convert to grayscale
    if (mImGray.channels() == 3)
        cvtColor(mImGray, mImGray, mbRGB ? CV_RGB2GRAY : CV_BGR2GRAY);
    else if (mImGray.channels() == 4)
        cvtColor(mImGray, mImGray, mbRGB ? CV_RGBA2GRAY : CV_BGRA2GRAY);
    else
        mImGray = mImGray.clone(); // avoid writing to caller's memory

    // ── V1: Adaptive CLAHE based on texture score ────────────────────────────
    cv::Mat lap; // giữ lại để tái dùng cho zonal score, tránh tính Laplacian 2 lần
    {
        cv::Laplacian(mImGray, lap, CV_64F);
        cv::Scalar mean_l, std_l;
        cv::meanStdDev(lap, mean_l, std_l);
        current_texture_score_ = std_l[0] * std_l[0]; // variance of Laplacian (toàn frame)
    }

    // ── V1.2: zonal texture — % diện tích frame thuộc vùng low-texture ───────
    // Reduction thuần trên cùng `lap` vừa tính, O(rows*cols), không tốn thêm
    // 1 pass image processing nào. Không gán nhãn "trần/nền" — chỉ đo diện
    // tích thực sự đói texture, nên vẫn đúng kể cả khi camera đang nghiêng
    // mạnh giữa góc cua (lúc "trên ảnh = trần" không còn đúng nữa).
    {
        int n_low = 0, n_total = kTextureGridRows * kTextureGridCols;
        int cellH = std::max(1, lap.rows / kTextureGridRows);
        int cellW = std::max(1, lap.cols / kTextureGridCols);
        for (int r = 0; r < kTextureGridRows; ++r) {
            for (int c = 0; c < kTextureGridCols; ++c) {
                cv::Rect cell(c*cellW, r*cellH,
                              std::min(cellW, lap.cols - c*cellW),
                              std::min(cellH, lap.rows - r*cellH));
                if (cell.width <= 0 || cell.height <= 0) continue;
                cv::Scalar m, s;
                cv::meanStdDev(lap(cell), m, s);
                double cell_score = s[0] * s[0];
                if (cell_score < kLowTextureExtractorThresh)
                    n_low++;
            }
        }
        mLowTextureFraction = float(n_low) / float(n_total);
        if (current_frame_idx_ % 10 == 0)
            std::cout << "[ZonalTexture] lowFraction=" << mLowTextureFraction
                      << " globalScore=" << current_texture_score_ << "\n";
    }
    {
        static cv::Ptr<cv::CLAHE> clahe_strong = cv::createCLAHE(3.0, cv::Size(4, 4));
        static cv::Ptr<cv::CLAHE> clahe_normal = cv::createCLAHE(1.5, cv::Size(8, 8));
        if (current_texture_score_ < 50.0)
            clahe_strong->apply(mImGray, mImGray);
        else
            clahe_normal->apply(mImGray, mImGray);
    }
    // ─────────────────────────────────────────────────────────────────────────

    // Filter detections
    current_frame_detections_ = detections;
    current_frame_good_detections_.clear();
    for (auto& det : current_frame_detections_)
        if (det->score > 0.6f)
            current_frame_good_detections_.push_back(det);

    // V1: build mask from ALL person detections, kernel=5
    cv::Mat exclude_mask;
    if (mbUseMask)
        exclude_mask = BuildPersonMask(mImGray.size(), 5);

    // Create Frame (use ini extractor during initialisation)
    if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
    {
        mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary,
                              mK, mDistCoef, mbf, mThDepth, exclude_mask);
    }
    else
    {
        // V1.2: chọn extractor theo texture score/zonal fraction đã tính
        // TRƯỚC CLAHE phía trên. Dùng OR giữa global score và zonal fraction:
        // global score thấp bắt được trường hợp toàn frame phẳng; zonal
        // fraction>0.5 bắt thêm trường hợp global "tạm ổn" nhưng phần lớn
        // diện tích vẫn đói texture (vài mép vật kéo trung bình lên).
        ORBextractor* pExtractor = mpORBextractorLeft;
        if ((current_texture_score_ < kLowTextureExtractorThresh ||
             mLowTextureFraction > 0.5f) && mpORBextractorLowTexture)
        {
            pExtractor = mpORBextractorLowTexture;
            std::cout << "[LowTextureExtractor] score=" << current_texture_score_
                      << " lowFraction=" << mLowTextureFraction
                      << " -> using low-threshold extractor\n";
        }
        mCurrentFrame = Frame(mImGray, timestamp, pExtractor, mpORBVocabulary,
                              mK, mDistCoef, mbf, mThDepth, exclude_mask);
    }

    if (force_relocalize)
    {
        auto t1 = high_resolution_clock::now();
        bool bOK = false;

        int mode = mpSystem->GetRelocalizationMode();
        if (mode == RELOC_POINTS)
            bOK = Relocalization();
        else if (mode == RELOC_OBJECTS)
            bOK = RelocalizationFromObjects(false);
        else if (mode == RELOC_OBJECTS_POINTS) {
            bOK = Relocalization();
            if (!bOK) bOK = RelocalizationFromObjects(true);
        }

        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms = t2 - t1;
        mpFrameDrawer->Update(this);
        if (bOK) mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
        mpSystem->relocalization_duration = ms.count();
        mpSystem->relocalization_status   = bOK;
    }
    else
    {
        Track();

        // Update mean depth
        if (mState == Tracking::OK) {
            Matrix34d Rt = cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw);
            double z_mean = 0.0; unsigned int z_nb = 0;
            for (size_t i = 0; i < mCurrentFrame.mvpMapPoints.size(); i++) {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if (pMP && !mCurrentFrame.mvbOutlier[i]) {
                    cv::Mat mp = pMP->GetWorldPos();
                    Eigen::Vector4d p(mp.at<float>(0), mp.at<float>(1), mp.at<float>(2), 1.0);
                    z_mean += (Rt * p)[2]; z_nb++;
                }
            }
            if (z_nb > 0) current_mean_depth_ = z_mean / z_nb;
        }

        std::cout << "Frame " << current_frame_idx_ << " ===========\n";
        mpFrameDrawer->Update(this);
    }

    if (mpARViewer) {
        mpARViewer->UpdateFrame(im_rgb_);
        if (mCurrentFrame.mTcw.rows == 4)
            mpARViewer->SetCurrentCameraPose(
                cvToEigenMatrix<double, float, 3, 4>(mCurrentFrame.mTcw));
    }

    // V1: seed LK bridge
    mImGray.copyTo(mPrevGray);
    if (mState == OK) {
        mvPrevPts.clear(); mvPrevMPs.clear();
        for (int i = 0; i < mCurrentFrame.N; ++i) {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if (pMP && !mCurrentFrame.mvbOutlier[i] && !pMP->isBad() && pMP->Observations() >= 2) {
                mvPrevPts.push_back(mCurrentFrame.mvKeysUn[i].pt);
                mvPrevMPs.push_back(pMP);
            }
        }
    }

    return mCurrentFrame.mTcw.clone();
}

// ===========================================================================
// ComputeVelocityRotDeg
// ===========================================================================
double Tracking::ComputeVelocityRotDeg() const
{
    if (mVelocity.empty()) return 0.0;
    cv::Mat Rv = mVelocity.rowRange(0,3).colRange(0,3);
    double tr = Rv.at<float>(0,0) + Rv.at<float>(1,1) + Rv.at<float>(2,2);
    return std::acos(std::max(-1.0, std::min(1.0, (tr-1.0)/2.0))) * 180.0 / M_PI;
}

// ===========================================================================
// TryInertiaRecovery  — V1.2 addition
// ===========================================================================
// Ngoại suy mLastGoodVelocity lên pose của mLastFrame khi mọi cơ chế tracking
// chuẩn (motion model, ref-KF, MultiSeed, LK) đều fail. Chỉ "coast" khi đang
// thực sự quay (rotation đáng kể) — ngoại suy translation thuần lúc mất track
// thường là occlusion thật, không đáng đánh đổi. Không bao giờ tự nhận là
// tracking thành công: pose này chưa qua TrackLocalMap/PoseOptimization xác
// nhận bằng inlier thật, nên NeedNewKeyFrame() phải chặn riêng (xem dưới).
// ===========================================================================
bool Tracking::TryInertiaRecovery()
{
    if (mLastGoodVelocity.empty() || mLastFrame.mTcw.empty())
        return false;

    if (mnInertiaFrames >= kMaxInertiaFrames)
    {
        std::cout << "[Inertia] Cap reached (" << mnInertiaFrames
                  << " frames), giving up.\n";
        mbInertiaMode   = false;
        mnInertiaFrames = 0;   // sẵn sàng cho lần quay tiếp theo
        return false;
    }

    cv::Mat Rv = mLastGoodVelocity.rowRange(0,3).colRange(0,3);
    double tr  = Rv.at<float>(0,0)+Rv.at<float>(1,1)+Rv.at<float>(2,2);
    double rotDeg = std::acos(std::max(-1.0,std::min(1.0,(tr-1.0)/2.0)))*180.0/M_PI;
    if (rotDeg < 1.0)
        return false;

    // V1.8: prefer the averaged rotation-extrapolated pose when history is
    // available — PredictPoseWithRotationalVelocity() averages the last N
    // good velocities on SO(3), which gives a smoother extrapolation than
    // a single velocity step (mLastGoodVelocity * mLastFrame.mTcw) and
    // reduces the reprojection window miss rate during fast pans.
    // Fall back to the single-velocity formula when history is absent (e.g.
    // just after init or after a hard Reset).
    cv::Mat predictedPose;
    if (!mRotVelHistory.empty())
    {
        predictedPose = PredictPoseWithRotationalVelocity();
        if (!predictedPose.empty())
            std::cout << "[Inertia] Using rot-averaged prediction (history="
                      << mRotVelHistory.size() << ")\n";
    }
    if (predictedPose.empty())
        predictedPose = mLastGoodVelocity * mLastFrame.mTcw;

    mCurrentFrame.SetPose(predictedPose);
    mbInertiaMode = true;
    mnInertiaFrames++;

    std::cout << "[Inertia] Coasting frame " << mnInertiaFrames << "/"
              << kMaxInertiaFrames << " (rot=" << rotDeg << " deg/frame)\n";
    return true;
}

// ===========================================================================
// Track  — main dispatch
// ===========================================================================
void Tracking::Track()
{
    createdNewKeyFrame_ = false;

    if (mState == NO_IMAGES_YET)
        mState = NOT_INITIALIZED;

    mLastProcessedState = mState;

    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if (mState == NOT_INITIALIZED)
    {
        if (mSensor == System::STEREO || mSensor == System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();

        mpFrameDrawer->Update(this);
        if (mState != OK) return;
    }
    else
    {
        bool bOK = false;

        if (!mbOnlyTracking)
        {
            // ── SLAM mode ────────────────────────────────────────────────────
            if (mState == OK)
            {
                CheckReplacedInLastFrame();

                if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                {
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    // V1: check rotation magnitude
                    double rot_deg = ComputeVelocityRotDeg();
                    bool bLargeRot = (rot_deg > kLargeRotThreshDeg);

                    if (bLargeRot)
                    {
                        std::cout << "[Track] Large rot " << rot_deg
                                  << " deg -> MultiSeed\n";
                        bOK = TrackReferenceKeyFrameMultiSeed();
                        if (bOK) {
                            mVelocity = cv::Mat();
                            std::cout << "[Track] MultiSeed OK\n";
                        } else {
                            // Fallback 1: rotation-aware projection search
                            // (no BoW bias, uses averaged velocity as seed)
                            bOK = TrackByProjectionFastPan();
                            if (bOK) {
                                mVelocity = cv::Mat();
                                std::cout << "[Track] FastPan projection OK\n";
                            } else {
                                // Fallback 2: LK bridge
                                if (mSensor != System::MONOCULAR && mvPrevPts.size() >= 8) {
                                    bOK = TrackWithOpticalFlow();
                                    if (bOK) {
                                        mVelocity = cv::Mat();
                                        std::cout << "[Track] LK saved large-rot\n";
                                    }
                                }
                                if (!bOK) {
                                    bOK = TrackReferenceKeyFrame();
                                    if (!bOK) mVelocity = cv::Mat();
                                }
                            }
                        }
                    }
                    else
                    {
                        bOK = TrackWithMotionModel();
                        if (!bOK)
                            bOK = TrackReferenceKeyFrame();
                        // V1: LK fallback for normal-motion failures
                        if (mSensor != System::MONOCULAR && !bOK && mvPrevPts.size() >= 8) {
                            bOK = TrackWithOpticalFlow();
                            if (bOK)
                                std::cout << "[Track] LK bridge saved normal-path\n";
                        }
                    }
                }
            }
            else
            {
                // LOST — cheapest recovery first, most expensive last.
                // V1.1: order changed from v1 (full reloc first) to v0.1-style
                // (quick covisibility -> LK -> full reloc), since most LOST
                // episodes are brief occlusions/blur, not real relocalization.
                bool bOK_local = QuickCovisibilityRecovery();

                if (mSensor != System::MONOCULAR && !bOK_local && mvPrevPts.size() >= 8) {
                    bOK_local = TrackWithOpticalFlow();
                    if (bOK_local)
                        std::cout << "[Track] LK recovered from LOST\n";
                }

                // V1.8: rotation-seeded projection recovery — fast-pan LOST
                // scenario. Tried AFTER QuickCovisibilityRecovery and LK both
                // fail, BEFORE the expensive async background recovery chain.
                // Only runs when mRotVelHistory is non-empty (camera was
                // actively rotating just before going LOST). Uses a
                // rotation-extrapolated seed + wide 50px radius, and rebuilds
                // local map from covisibility KFs when mvpLocalMapPoints is
                // stale/empty (the normal case in the LOST branch).
                if (mSensor != System::MONOCULAR && !bOK_local)
                {
                    bOK_local = QuickRotationRecovery();
                    if (bOK_local)
                        std::cout << "[Track] QuickRotationRecovery recovered from LOST\n";
                }

                // V1.6: the heavy recovery steps (BoW-independent projection
                // over all map KFs, full BoW-based Relocalization()) no longer
                // run synchronously here. They were the actual source of the
                // "đơ" (freeze) — each LOST frame was paying their full cost
                // every single time, mostly failing, with no backoff. They now
                // run on a dedicated background thread (see RecoveryThreadLoop)
                // against its OWN Frame copy, throttled to at most one attempt
                // in flight (kRecoveryThreadMinIntervalMs apart). The main
                // thread here only does two cheap, non-blocking things:
                //   1. Check whether the background thread just published a
                //      successful pose for an earlier frame — if so, apply it.
                //   2. If still LOST, hand the background thread a fresh
                //      snapshot to work on NEXT, but only if it's idle.
                // Both calls return immediately; viewer frame rate is no
                // longer coupled to relocalization cost at all.
                bool bScaleRecoveryNeeded = false;
                if (!bOK_local)
                {
                    bOK_local = ConsumePublishedRecoveryResultIfAny();
                    if (bOK_local)
                    {
                        std::cout << "[Track] Background recovery thread "
                                  << "recovered from LOST\n";
                        bScaleRecoveryNeeded = true; // pose came from a stale-ish snapshot
                    }
                    else
                    {
                        OfferRecoverySnapshotIfIdle();
                    }
                }

                // V1.5: stale-recovery escalation watchdog.
                // Only tracked while we're inside a post-soft-reset recovery
                // window (mbPostResetBoost). Without async offload this used
                // to mean "retry with a brute-force scan"; now the background
                // thread already scans every KF in the map on its own — so the
                // watchdog's job is reduced to: if recovery is taking too long
                // even with the background thread helping, give up and fall
                // back to hard Reset() (observed to be reliable), since that's
                // strictly better than waiting forever on a soft reset that
                // was seeded badly.
                if (!bOK_local && mbPostResetBoost)
                {
                    mnPostResetLostFrames++;
                    if (mnPostResetLostFrames >= kStaleRecoveryGiveUpFrames)
                    {
                        std::cout << "[Track] Stale recovery exceeded "
                                  << kStaleRecoveryGiveUpFrames
                                  << " frames (background thread included), soft "
                                  << "reset failed -> falling back to hard Reset().\n";
                        mpSystem->Reset();
                        return;
                    }
                }
                else if (bOK_local && mbPostResetBoost)
                {
                    mnPostResetLostFrames = 0;
                }

                // V1.1: scale recovery — a pose recovered via the background
                // thread's BoW Relocalization stage can carry a different
                // scale than the map the camera was lost from. Re-anchor it,
                // same as before, just gated on the async path now.
                if (bOK_local && bScaleRecoveryNeeded)
                {
                    double sd = EstimateScaleFromDepth();
                    double so = EstimateScaleFromObjects(mCurrentFrame.mTcw);
                    std::vector<double> valid_scales;
                    if (sd > 0) valid_scales.push_back(sd);
                    if (so > 0) valid_scales.push_back(so);
                    if (!valid_scales.empty())
                    {
                        double final_scale;
                        if (valid_scales.size() == 2 &&
                            std::abs(valid_scales[0]-valid_scales[1]) < 0.3)
                            final_scale = (valid_scales[0]+valid_scales[1]) / 2.0;
                        else
                            final_scale = valid_scales[0];
                        ApplyScaleCorrection(final_scale);
                    }
                }

                bOK = bOK_local;

                if (bOK)  std::cout << "Relocalization OK\n";
                else      std::cout << "Relocalization failed\n";
            }
        }
        else
        {
            // ── Localization mode ─────────────────────────────────────────────
            if (mState == LOST)
            {
                int mode = mpSystem->GetRelocalizationMode();
                if (mode == RELOC_POINTS)
                    bOK = Relocalization();
                else if (mode == RELOC_OBJECTS)
                    bOK = RelocalizationFromObjects(false);
                else if (mode == RELOC_OBJECTS_POINTS) {
                    bOK = Relocalization();
                    if (!bOK) bOK = RelocalizationFromObjects(true);
                }
            }
            else if (!mbVO)
            {
                bOK = mVelocity.empty() ? TrackReferenceKeyFrame()
                                        : TrackWithMotionModel();
                // V1: survival — hold last pose rather than flying blind
                if (!bOK && !mLastFrame.mTcw.empty()) {
                    mCurrentFrame.SetPose(mLastFrame.mTcw);
                    bOK = true;
                    mbVO = true;
                }
            }
            else
            {
                bool bOKMM = false, bOKReloc = false;
                vector<MapPoint*> vpMPsMM;
                vector<bool> vbOutMM;
                cv::Mat TcwMM;
                if (!mVelocity.empty()) {
                    bOKMM  = TrackWithMotionModel();
                    vpMPsMM = mCurrentFrame.mvpMapPoints;
                    vbOutMM = mCurrentFrame.mvbOutlier;
                    TcwMM   = mCurrentFrame.mTcw.clone();
                }
                bOKReloc = Relocalization();
                if (bOKMM && !bOKReloc) {
                    mCurrentFrame.SetPose(TcwMM);
                    mCurrentFrame.mvpMapPoints = vpMPsMM;
                    mCurrentFrame.mvbOutlier   = vbOutMM;
                    if (mbVO)
                        for (int i = 0; i < mCurrentFrame.N; i++)
                            if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                } else if (bOKReloc) {
                    mbVO = false;
                }
                bOK = bOKReloc || bOKMM;
            }
        }

        // ── V1.2: inertia mode ───────────────────────────────────────────────
        // Thử ngay trước khi rơi vào TrackLocalMap(). Nếu TrackLocalMap() sau
        // đó vẫn không tìm được inlier thật, bOK quay lại false và mState=LOST
        // như cũ — inertia không che giấu lost, chỉ giữ pose liên tục cho
        // frame kế tiếp (giúp QuickCovisibilityRecovery có seed hợp lý).
        if (!mbOnlyTracking && !bOK)
            bOK = TryInertiaRecovery();

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // Track local map
        if (!mbOnlyTracking)  { if (bOK) bOK = TrackLocalMap(); }
        else                  { if (bOK && !mbVO) bOK = TrackLocalMap(); }

        mState = bOK ? OK : LOST;

        // V1: consecutive lost counter
        if (mState == OK)  mnConsecutiveLostFrames = 0;
        else               mnConsecutiveLostFrames++;

        // --- SOFT RESET ON PROLONGED LOST (PRESERVE MAP) ---
        if (mState == LOST && mnConsecutiveLostFrames >= 30)
        {
            std::cout << "[Track] LOST for " << mnConsecutiveLostFrames 
                      << " frames! Triggering MapPreservingReset\n";

            bool softOK = MapPreservingReset();
            if (!softOK)
            {
                std::cout << "[Track] Map too small for soft reset, falling back to System Reset\n";
                mpSystem->Reset();
            }
            return;
        }

        // V1.2: chỉ clear inertia khi tracking THẬT SỰ thành công (TrackLocalMap
        // đã chạy và pass với inlier thật), không phải chỉ vì TryInertiaRecovery
        // trả true ở trên.
        if (mState == OK) { mbInertiaMode = false; mnInertiaFrames = 0; }

        // V1.5: tracking genuinely OK again -> watchdog no longer needed.
        if (mState == OK) mnPostResetLostFrames = 0;

        if (bOK)
        {
            // V1: update velocity with sanity check
            if (!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                cv::Mat newVel = mCurrentFrame.mTcw * LastTwc;

                // Sanity: rotation > 15°/frame → garbage velocity → clear
                cv::Mat Rv = newVel.rowRange(0,3).colRange(0,3);
                double tr = Rv.at<float>(0,0)+Rv.at<float>(1,1)+Rv.at<float>(2,2);
                double rot = std::acos(std::max(-1.0,std::min(1.0,(tr-1.0)/2.0)))*180.0/M_PI;
                if (rot > 15.0) {
                    std::cout << "[Track] Velocity sanity fail: " << rot << " deg -> cleared\n";
                    mVelocity = cv::Mat();
                } else {
                    mVelocity = newVel;
                }
            }
            else mVelocity = cv::Mat();

            // V1.2: snapshot velocity tốt nhất gần đây — chỉ khi không rỗng,
            // tránh ghi đè bằng velocity rỗng ngay sau init/relocalize. Đây
            // là nguồn duy nhất TryInertiaRecovery() dùng để ngoại suy, nên
            // phải tách khỏi mVelocity (bị clear ngay khi có gì đó fail).
            if (!mVelocity.empty())
            {
                mLastGoodVelocity = mVelocity.clone();

                // V1.3: maintain rotation velocity history ring buffer
                mRotVelHistory.insert(mRotVelHistory.begin(), mVelocity.clone());
                if ((int)mRotVelHistory.size() > kRotVelHistLen)
                    mRotVelHistory.resize(kRotVelHistLen);
            }

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for (int i = 0; i < mCurrentFrame.N; i++) {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if (pMP && pMP->Observations() < 1) {
                    mCurrentFrame.mvbOutlier[i]   = false;
                    mCurrentFrame.mvpMapPoints[i] = nullptr;
                }
            }

            // Delete temporal MapPoints
            for (auto lit = mlpTemporalPoints.begin(); lit != mlpTemporalPoints.end(); lit++)
                delete *lit;
            mlpTemporalPoints.clear();

            // V1.1: snapshot pre-lost state — used by scale-guard / scale-recovery
            // when LK bridge / MultiSeed / Relocalization later save the track.
            if (mpReferenceKF && mnMatchesInliers >= 15)
            {
                float d = mpReferenceKF->ComputeSceneMedianDepth(2);
                if (d > 0) {
                    mPreLostMedianDepth = d;
                    mPreLostTcw = mCurrentFrame.mTcw.clone();
                    mnPreLostInliers = mnMatchesInliers;
                    mPreLostFrameId = mCurrentFrame.mnId;
                }
            }

            // V1.1: scale sanity check before allowing a new KF.
            // After a LK bridge / MultiSeed recovery the pose can carry a wrong
            // scale; inserting such a KF poisons the map permanently via BA.
            // Only meaningful for monocular (stereo/RGBD have absolute scale).
            bool bScaleSane = true;
            if (mSensor == System::MONOCULAR && mPreLostMedianDepth > 0 && mpReferenceKF)
            {
                float cur_d = mpReferenceKF->ComputeSceneMedianDepth(2);
                if (cur_d > 0)
                {
                    double ratio = cur_d / mPreLostMedianDepth;
                    if (ratio < 0.4 || ratio > 2.5)
                    {
                        bScaleSane = false;
                        std::cout << "[ScaleGuard] KF rejected: depth ratio=" << ratio
                                  << " (pre=" << mPreLostMedianDepth << " cur=" << cur_d << ")\n";
                    }
                }
            }

            if (bScaleSane && NeedNewKeyFrame()) {
                CreateNewKeyFrame();
                createdNewKeyFrame_ = true;
            }

            for (int i = 0; i < mCurrentFrame.N; i++)
                if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i] = nullptr;
        }

        // V1: conservative reset — only when map is truly tiny AND lost for a long time
        if (mState == LOST)
        {
            const int nKFs = mpMap->KeyFramesInMap();
            if (nKFs <= 3 && mnConsecutiveLostFrames >= 30) {
                cout << "Track lost soon after init, reseting...\n";
                mpSystem->Reset();
                return;
            } else if (nKFs > 3) {
                if (mnConsecutiveLostFrames % 30 == 0)
                    cout << "[Track] LOST for " << mnConsecutiveLostFrames
                         << " frames (" << nKFs << " KFs preserved)\n";
            }
        }

        if (!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store trajectory
    if (!mCurrentFrame.mTcw.empty() && mCurrentFrame.mpReferenceKF)
    {
        cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState == LOST);
    }
    else
    {
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState == LOST);
    }
}

// ===========================================================================
// StereoInitialization  — unchanged from V0
// ===========================================================================
void Tracking::StereoInitialization()
{
    if (mCurrentFrame.N > 500)
    {
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        KeyFrame* pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);
        mpMap->AddKeyFrame(pKFini);
        for (int i = 0; i < mCurrentFrame.N; i++) {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0) {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpMap);
                pNewMP->AddObservation(pKFini, i);
                pKFini->AddMapPoint(pNewMP, i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);
                mCurrentFrame.mvpMapPoints[i] = pNewMP;
            }
        }
        cout << "New map created with " << mpMap->MapPointsInMap() << " points\n";
        mpLocalMapper->InsertKeyFrame(pKFini);
        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->mvpKeyFrameOrigins.push_back(pKFini);
        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
        mState = OK;
    }
}

// ===========================================================================
// MonocularInitialization  — unchanged from V0
// ===========================================================================
void Tracking::MonocularInitialization()
{
    if (!mpInitializer)
    {
        if (mCurrentFrame.mvKeys.size() > 100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame    = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;
            if (mpInitializer) delete mpInitializer;
            mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);
            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
        }
        return;
    }

    if ((int)mCurrentFrame.mvKeys.size() <= 100)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
        fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
        return;
    }

    ORBmatcher matcher(0.9, true);
    int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame,
                                                   mvbPrevMatched, mvIniMatches, 100);
    if (nmatches < 100)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
        return;
    }

    cv::Mat Rcw, tcw;
    vector<bool> vbTriangulated;
    if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
    {
        for (size_t i = 0; i < mvIniMatches.size(); i++)
            if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
                mvIniMatches[i] = -1; nmatches--;
            }
        
        // Initialize at Identity
        mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
        cv::Mat Tcw_rel = cv::Mat::eye(4,4,CV_32F);
        Rcw.copyTo(Tcw_rel.rowRange(0,3).colRange(0,3));
        tcw.copyTo(Tcw_rel.rowRange(0,3).col(3));
        mCurrentFrame.SetPose(Tcw_rel);

        CreateInitialMapMonocular();
    }
}

// ===========================================================================
void Tracking::CreateInitialMapMonocular()
{
    KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);
    pKFini->ComputeBoW(); pKFcur->ComputeBoW();
    mpMap->AddKeyFrame(pKFini); mpMap->AddKeyFrame(pKFcur);

    for (size_t i = 0; i < mvIniMatches.size(); i++)
    {
        if (mvIniMatches[i] < 0) continue;
        cv::Mat worldPos(mvIniP3D[i]);
        MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);
        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);
        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);
        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]]   = false;
        mpMap->AddMapPoint(pMP);
    }
    pKFini->UpdateConnections(); pKFcur->UpdateConnections();
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points\n";

    Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

    float medianDepth    = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f / medianDepth;
    if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100) {
        cout << "Wrong initialization, reseting...\n";
        Reset(); return;
    }

    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3) * invMedianDepth;
    pKFcur->SetPose(Tc2w);

    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for (auto pMP : vpAllMapPoints)
        if (pMP) pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);
    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame   = pKFcur;
    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints  = mpMap->GetAllMapPoints();
    mpReferenceKF      = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;
    mLastFrame = Frame(mCurrentFrame);
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());
    mpMap->mvpKeyFrameOrigins.push_back(pKFini);
    mState = OK;
}

// ===========================================================================
// CheckReplacedInLastFrame  — unchanged from V0
// ===========================================================================
void Tracking::CheckReplacedInLastFrame()
{
    for (int i = 0; i < mLastFrame.N; i++) {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if (pMP) {
            MapPoint* pRep = pMP->GetReplaced();
            if (pRep) mLastFrame.mvpMapPoints[i] = pRep;
        }
    }
}

// ===========================================================================
// TrackReferenceKeyFrame  — V1: dual seed (KF pose vs lastFrame pose)
// ===========================================================================
bool Tracking::TrackReferenceKeyFrame()
{
    mCurrentFrame.ComputeBoW();
    ORBmatcher matcher(0.7, true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);
    if (nmatches < 10) return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;

    // V1: choose seed based on rotation
    double rotDeg = ComputeVelocityRotDeg();
    if (rotDeg > 10.0 && mpReferenceKF)
        mCurrentFrame.SetPose(mpReferenceKF->GetPose());
    else
        mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // If inlier count is low, try the other seed
    int nI = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
        if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i]
            && mCurrentFrame.mvpMapPoints[i]->Observations() > 0) nI++;

    if (nI < 15)
    {
        mCurrentFrame.mvpMapPoints = vpMapPointMatches;
        fill(mCurrentFrame.mvbOutlier.begin(), mCurrentFrame.mvbOutlier.end(), false);
        if (rotDeg > 10.0)
            mCurrentFrame.SetPose(mLastFrame.mTcw);
        else if (mpReferenceKF)
            mCurrentFrame.SetPose(mpReferenceKF->GetPose());
        Optimizer::PoseOptimization(&mCurrentFrame);
    }

    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++) {
        if (mCurrentFrame.mvpMapPoints[i]) {
            if (mCurrentFrame.mvbOutlier[i]) {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                mCurrentFrame.mvpMapPoints[i] = nullptr;
                mCurrentFrame.mvbOutlier[i]   = false;
                pMP->mbTrackInView    = false;
                pMP->mnLastFrameSeen  = mCurrentFrame.mnId;
                nmatches--;
            } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }
    return nmatchesMap >= 10;
}

// ===========================================================================
// TrackReferenceKeyFrameMultiSeed  — V1 addition
// ===========================================================================
bool Tracking::TrackReferenceKeyFrameMultiSeed()
{
    mCurrentFrame.ComputeBoW();

    std::vector<KeyFrame*> vpCands;
    if (mpReferenceKF) {
        vpCands.push_back(mpReferenceKF);
        auto covis = mpReferenceKF->GetBestCovisibilityKeyFrames(kMaxMultiSeedKF);
        for (auto* kf : covis)
            if (kf && !kf->isBad()) vpCands.push_back(kf);
    }
    for (auto* kf : mvpLocalKeyFrames)
        if ((int)vpCands.size() < kMaxMultiSeedKF && kf && !kf->isBad())
            vpCands.push_back(kf);

    ORBmatcher matcher(0.75, true);
    int bestInliers = 0;
    cv::Mat bestTcw;
    vector<MapPoint*> bestMPs;

    for (auto* pKF : vpCands)
    {
        if (!pKF || pKF->isBad()) continue;
        vector<MapPoint*> vpMPM;
        int nm = matcher.SearchByBoW(pKF, mCurrentFrame, vpMPM);
        if (nm < 10) continue;

        mCurrentFrame.mvpMapPoints = vpMPM;
        fill(mCurrentFrame.mvbOutlier.begin(), mCurrentFrame.mvbOutlier.end(), false);
        mCurrentFrame.SetPose(pKF->GetPose()); // KF pose, not lastFrame
        Optimizer::PoseOptimization(&mCurrentFrame);

        int nIn = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
            if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i]
                && mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nIn++;

        if (nIn > bestInliers) {
            bestInliers = nIn;
            bestTcw     = mCurrentFrame.mTcw.clone();
            bestMPs     = mCurrentFrame.mvpMapPoints;
            if (nIn >= 30) break;
        }
    }

    if (bestInliers < 10 || bestTcw.empty()) return false;

    mCurrentFrame.SetPose(bestTcw);
    mCurrentFrame.mvpMapPoints = bestMPs;
    fill(mCurrentFrame.mvbOutlier.begin(), mCurrentFrame.mvbOutlier.end(), false);
    Optimizer::PoseOptimization(&mCurrentFrame);

    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++) {
        if (mCurrentFrame.mvpMapPoints[i]) {
            if (mCurrentFrame.mvbOutlier[i]) {
                mCurrentFrame.mvpMapPoints[i]->mbTrackInView   = false;
                mCurrentFrame.mvpMapPoints[i]->mnLastFrameSeen = mCurrentFrame.mnId;
                mCurrentFrame.mvpMapPoints[i] = nullptr;
                mCurrentFrame.mvbOutlier[i]   = false;
            } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }
    std::cout << "[MultiSeed] bestInliers=" << bestInliers
              << " nmatchesMap=" << nmatchesMap << "\n";
    return nmatchesMap >= 10;
}

// ===========================================================================
// QuickCovisibilityRecovery  — V1.1 addition (ported from v0.1)
// ===========================================================================
// Cheap recovery attempt (~2-5ms) tried before falling back to full
// Relocalization() (~50-100ms). Most LOST episodes are brief (occlusion,
// motion blur for a couple frames) and a covisible KF with the *candidate
// KF's own pose* as seed is usually enough — no need for a full BoW DB
// query + PnP RANSAC.
//
// V1.4 FIX: the original version seeded every candidate with
// mCurrentFrame.SetPose(mLastFrame.mTcw) — a FIXED pose across the whole
// loop. Only the point set (pKF) varied; the assumed camera pose never did.
// After MapPreservingReset() mLastFrame.mTcw is just a placeholder copied
// from mpReferenceKF (see MapPreservingReset), not the camera's real pose,
// so with only a 25px search window this almost always missed — making the
// function nearly a no-op exactly when it was needed most (right after a
// Soft Reset). So each candidate was seeded with its OWN pose
// (pKF->GetPose()) instead, matching the same pattern used in
// TrackReferenceKeyFrameMultiSeed().
//
// V1.7 FIX: that change regressed ordinary mid-track LOST recovery (e.g.
// brief occlusion/blur while panning). During normal tracking mLastFrame.mTcw
// IS the real camera pose from one frame ago — always closer to the current
// true pose than a candidate KF's static pose, especially while rotating
// (a KF can be many degrees away in viewing angle). Seeding with pKF->GetPose()
// unconditionally made the 25px projection search miss far more often while
// panning, pushing recovery down to the heavier LK/async-relocalization paths
// (or outright LOST) more frequently than before.
//
// Fix: use mLastFrame.mTcw as the seed whenever it's trustworthy (i.e. we are
// NOT in the post-soft-reset window, where mLastFrame is only a placeholder —
// see MapPreservingReset()). Only fall back to each candidate's own pose
// (pKF->GetPose()) during mbPostResetBoost, which is exactly the situation
// the V1.4 fix was written for.
// ===========================================================================
bool Tracking::QuickCovisibilityRecovery()
{
    if (!mpReferenceKF)
        return false;

    // mLastFrame.mTcw is only a real pose estimate during ordinary tracking.
    // Right after MapPreservingReset() it's a placeholder (copied from the
    // seed KF, see MapPreservingReset()), so it carries no information about
    // where the camera actually is — in that case seed from each candidate's
    // own pose instead (V1.4 behaviour).
    const bool bLastFrameSeedTrustworthy =
        !mLastFrame.mTcw.empty() && !mbPostResetBoost;

    // V1.8: when rotation history indicates a fast pan was in progress just
    // before going LOST, the rotation-extrapolated pose is a better seed than
    // mLastFrame.mTcw (which is already one frame behind and may project
    // MapPoints 30-80px away from their actual positions). Use the extrapolated
    // pose as primary seed and widen the search radius to 50px to tolerate the
    // extra uncertainty; fall back to the normal 25px window with mLastFrame
    // if the wider search still fails.
    cv::Mat rotExtrapSeed;
    bool bUseRotExtrap = false;
    if (bLastFrameSeedTrustworthy && !mRotVelHistory.empty())
    {
        // Only extrapolate when the velocity history shows meaningful rotation
        // (>= kLargeRotThreshDeg per frame) — pure translation doesn't benefit.
        cv::Mat Rv = mRotVelHistory[0].rowRange(0,3).colRange(0,3);
        double tr  = Rv.at<float>(0,0)+Rv.at<float>(1,1)+Rv.at<float>(2,2);
        double rotDeg = std::acos(std::max(-1.0,std::min(1.0,(tr-1.0)/2.0)))*180.0/M_PI;
        if (rotDeg >= kLargeRotThreshDeg)
        {
            rotExtrapSeed = PredictPoseWithRotationalVelocity();
            bUseRotExtrap = !rotExtrapSeed.empty();
            if (bUseRotExtrap)
                std::cout << "[QuickCovis] Fast-pan detected (" << rotDeg
                          << " deg/frame) -> using rot-extrapolated seed, th=50\n";
        }
    }

    std::vector<KeyFrame*> vpQR;
    vpQR.push_back(mpReferenceKF);
    auto covis = mpReferenceKF->GetBestCovisibilityKeyFrames(5);
    for (auto* k : covis)
        if (k && !k->isBad()) vpQR.push_back(k);

    ORBmatcher matcher(0.9, true);

    // ── Pass 1: rotation-extrapolated seed with wide radius (fast-pan only) ──
    if (bUseRotExtrap)
    {
        for (auto* pKF : vpQR)
        {
            if (!pKF || pKF->isBad()) continue;

            mCurrentFrame.SetPose(rotExtrapSeed);
            std::set<MapPoint*> sFound;
            int nm = matcher.SearchByProjection(mCurrentFrame, pKF, sFound, 50, 100);
            if (nm < 12) continue;

            Optimizer::PoseOptimization(&mCurrentFrame);
            int nIn = 0;
            for (int i = 0; i < mCurrentFrame.N; i++)
                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nIn++;

            if (nIn >= 10)
            {
                std::cout << "[QuickCovis] Rot-extrap seed OK (" << nIn
                          << " inliers, KF " << pKF->mnId << ")\n";
                mpReferenceKF = pKF;
                mCurrentFrame.mpReferenceKF = pKF;
                mbLastRecoveryWasQuickCovis = true;  // ← THÊM
                return true;
            }
        }
        // Pass 1 failed — fall through to normal seed below.
        std::cout << "[QuickCovis] Rot-extrap pass failed, trying normal seed.\n";
    }

    // ── Pass 2: normal seed (mLastFrame or per-KF pose), radius 25px ─────────
    for (auto* pKF : vpQR)
    {
        if (!pKF || pKF->isBad()) continue;

        // V1.7: prefer the real last-known pose (closer to truth during
        // ordinary tracking, including fast pans) over a candidate KF's
        // static pose; only use the KF's own pose post-soft-reset, when
        // mLastFrame is just a placeholder.
        mCurrentFrame.SetPose(bLastFrameSeedTrustworthy ? mLastFrame.mTcw
                                                         : pKF->GetPose());
        std::set<MapPoint*> sFound;
        int nm = matcher.SearchByProjection(mCurrentFrame, pKF, sFound, 25, 100);
        if (nm < 12) continue;

        Optimizer::PoseOptimization(&mCurrentFrame);
        int nIn = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
            if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                nIn++;

        if (nIn >= 10)
        {
            std::cout << "[QuickCovis] Normal seed OK (" << nIn << " inliers, KF "
                      << pKF->mnId << ")\n";
            mpReferenceKF = pKF;
            mCurrentFrame.mpReferenceKF = pKF;
            mbLastRecoveryWasQuickCovis = true;  // ← THÊM
            return true;
        }
    }
    return false;
}

// ===========================================================================
// RelocalizationByProjectionAllKFs  — V1.4 addition
// ===========================================================================
// BoW-independent relocalization. DetectRelocalizationCandidates() in
// Relocalization() relies on BoW shared-word score against the KF database;
// right after MapPreservingReset() the camera is often still inside the
// preserved map but viewing it from an angle/overlap that scores poorly,
// so Relocalization() can fail repeatedly even though the camera position
// IS in the map (visible as KF cluster + matching MapPoints in the viewer).
//
// This function instead tries every candidate KF (reference KF + its
// covisibility neighbours, falling back to all map KFs if needed), seeds
// mCurrentFrame with that KF's OWN pose, and does a wide-radius
// SearchByProjection of that KF's MapPoints — no descriptor BoW matching,
// no DB query. Cheaper than full Relocalization() and tried before it in
// the LOST recovery chain.
// ===========================================================================
bool Tracking::RelocalizationByProjectionAllKFs()
{
    std::vector<KeyFrame*> vpCands;
    if (mpReferenceKF && !mpReferenceKF->isBad())
    {
        vpCands.push_back(mpReferenceKF);
        for (auto* kf : mpReferenceKF->GetBestCovisibilityKeyFrames(kMaxProjRelocKF))
            if (kf && !kf->isBad()) vpCands.push_back(kf);
    }
    for (auto* kf : mvpLocalKeyFrames)
    {
        if ((int)vpCands.size() >= kMaxProjRelocKF) break;
        if (kf && !kf->isBad() &&
            std::find(vpCands.begin(), vpCands.end(), kf) == vpCands.end())
            vpCands.push_back(kf);
    }
    if (vpCands.empty())
    {
        for (auto* kf : mpMap->GetAllKeyFrames())
        {
            if ((int)vpCands.size() >= kMaxProjRelocKF) break;
            if (kf && !kf->isBad()) vpCands.push_back(kf);
        }
    }
    if (vpCands.empty())
    {
        std::cout << "[ProjReloc] No candidate KFs.\n";
        return false;
    }

    const int th = 30; // pose seed = exact KF pose, doesn't need to be as
                        // narrow as a motion-model search window
    int bestInliers = 0;
    cv::Mat bestTcw;
    vector<MapPoint*> bestMPs;
    KeyFrame* bestKF = nullptr;

    for (auto* pKF : vpCands)
    {
        if (!pKF || pKF->isBad()) continue;

        mCurrentFrame.SetPose(pKF->GetPose());
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);
        fill(mCurrentFrame.mvbOutlier.begin(),   mCurrentFrame.mvbOutlier.end(),   false);

        std::set<MapPoint*> sFound; // empty = search all of pKF's MapPoints
        ORBmatcher matcher(0.9, true);
        int nm = matcher.SearchByProjection(mCurrentFrame, pKF, sFound, th, 100);
        if (nm < 15) continue;

        Optimizer::PoseOptimization(&mCurrentFrame);

        int nIn = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
            if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                nIn++;

        if (nIn > bestInliers)
        {
            bestInliers = nIn;
            bestTcw     = mCurrentFrame.mTcw.clone();
            bestMPs     = mCurrentFrame.mvpMapPoints;
            bestKF      = pKF;
            if (nIn >= 50) break; // good enough, stop early
        }
    }

    if (bestInliers < mProjRelocMinInliers || bestTcw.empty())
    {
        std::cout << "[ProjReloc] Failed, best=" << bestInliers
                   << "/" << mProjRelocMinInliers << "\n";
        return false;
    }

    mCurrentFrame.SetPose(bestTcw);
    mCurrentFrame.mvpMapPoints = bestMPs;
    fill(mCurrentFrame.mvbOutlier.begin(), mCurrentFrame.mvbOutlier.end(), false);
    Optimizer::PoseOptimization(&mCurrentFrame);

    int nFinal = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (!mCurrentFrame.mvpMapPoints[i]) continue;
        if (mCurrentFrame.mvbOutlier[i]) {
            mCurrentFrame.mvpMapPoints[i] = nullptr;
            mCurrentFrame.mvbOutlier[i]   = false;
        } else {
            nFinal++;
        }
    }

    std::cout << "[ProjReloc] OK via KF " << (bestKF ? bestKF->mnId : -1)
              << " inliers=" << bestInliers << " final=" << nFinal << "\n";

    mnLastRelocFrameId = mCurrentFrame.mnId;
    if (bestKF) {
        mpReferenceKF = bestKF;
        mCurrentFrame.mpReferenceKF = bestKF;
    }
    return nFinal >= mProjRelocMinInliers / 2;
}

// ===========================================================================
// RelocalizationByProjectionAllKFsBruteForce  — V1.5 addition
// ===========================================================================
// Escalation step for when the normal recovery chain is stuck after a Soft
// Reset (MapPreservingReset). RelocalizationByProjectionAllKFs() limits its
// candidate KFs to mpReferenceKF + its covisibility neighbours (falling back
// to mvpLocalKeyFrames only if that's empty). If the seed KF picked by the
// soft reset turns out to be a bad starting point — e.g. the camera has
// since moved to a part of the map that isn't covisible with it — that
// function keeps trying the SAME small, wrong candidate set every frame and
// never succeeds, even though the camera is clearly inside the map (visible
// in the point cloud / KF cluster in the viewer).
//
// This function ignores mpReferenceKF and covisibility completely and walks
// every keyframe currently in the map ("rơi về match line tìm lại keypoint"),
// trying each as a pose seed for a wide-radius SearchByProjection. It is
// strictly more expensive than RelocalizationByProjectionAllKFs() (O(all KFs)
// instead of O(~20 covisible KFs)) so it is only called as an escalation,
// not on every LOST frame.
// ===========================================================================
bool Tracking::RelocalizationByProjectionAllKFsBruteForce()
{
    std::vector<KeyFrame*> vpAllKFs = mpMap->GetAllKeyFrames();
    if (vpAllKFs.empty())
    {
        std::cout << "[BruteForceReloc] Map has no KeyFrames.\n";
        return false;
    }

    const int th = 30;
    int bestInliers = 0;
    cv::Mat bestTcw;
    vector<MapPoint*> bestMPs;
    KeyFrame* bestKF = nullptr;

    ORBmatcher matcher(0.9, true);
    for (auto* pKF : vpAllKFs)
    {
        if (!pKF || pKF->isBad()) continue;

        mCurrentFrame.SetPose(pKF->GetPose());
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);
        fill(mCurrentFrame.mvbOutlier.begin(),   mCurrentFrame.mvbOutlier.end(),   false);

        std::set<MapPoint*> sFound; // empty = search all of pKF's MapPoints
        int nm = matcher.SearchByProjection(mCurrentFrame, pKF, sFound, th, 100);
        if (nm < 15) continue;

        Optimizer::PoseOptimization(&mCurrentFrame);

        int nIn = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
            if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                nIn++;

        if (nIn > bestInliers)
        {
            bestInliers = nIn;
            bestTcw     = mCurrentFrame.mTcw.clone();
            bestMPs     = mCurrentFrame.mvpMapPoints;
            bestKF      = pKF;
            if (nIn >= 50) break; // good enough, stop early
        }
    }

    // Lower bar than the covisibility-limited version: this is a last-resort
    // escalation, and TrackLocalMap() right after will still reject a truly
    // bad pose via its own inlier threshold — so we just need enough to give
    // the recovery a fighting chance, not a final confirmation.
    const int minInliers = std::max(15, mProjRelocMinInliers / 2);
    if (bestInliers < minInliers || bestTcw.empty())
    {
        std::cout << "[BruteForceReloc] Failed across " << vpAllKFs.size()
                   << " KFs, best=" << bestInliers << "/" << minInliers << "\n";
        return false;
    }

    mCurrentFrame.SetPose(bestTcw);
    mCurrentFrame.mvpMapPoints = bestMPs;
    fill(mCurrentFrame.mvbOutlier.begin(), mCurrentFrame.mvbOutlier.end(), false);
    Optimizer::PoseOptimization(&mCurrentFrame);

    int nFinal = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (!mCurrentFrame.mvpMapPoints[i]) continue;
        if (mCurrentFrame.mvbOutlier[i]) {
            mCurrentFrame.mvpMapPoints[i] = nullptr;
            mCurrentFrame.mvbOutlier[i]   = false;
        } else {
            nFinal++;
        }
    }

    std::cout << "[BruteForceReloc] OK via KF " << (bestKF ? bestKF->mnId : -1)
              << " inliers=" << bestInliers << " final=" << nFinal
              << " (scanned " << vpAllKFs.size() << " KFs)\n";

    mnLastRelocFrameId = mCurrentFrame.mnId;
    if (bestKF) {
        // Re-seed reference KF AND reset the boost/escalation counters — this
        // is effectively a fresh, correctly-anchored starting point for the
        // post-soft-reset recovery.
        mpReferenceKF = bestKF;
        mCurrentFrame.mpReferenceKF = bestKF;
        mnPostResetLostFrames = 0;
    }
    return nFinal >= minInliers / 2;
}

// ===========================================================================
// QuickRotationRecovery  — V1.8
// ===========================================================================
// Called in the LOST branch AFTER QuickCovisibilityRecovery and LK both fail.
// Uses PredictPoseWithRotationalVelocity() to extrapolate the camera pose
// through the current pan, then does a wide SearchByProjection over local map
// points — same logic as TrackByProjectionFastPan() but:
//   (a) callable from the LOST branch (mState != OK), and
//   (b) explicitly rebuilds local map from covisibility KFs when
//       mvpLocalMapPoints is empty/stale (see Issue 4 fix in
//       TrackByProjectionFastPan below).
//
// Only attempted when mRotVelHistory is non-empty — i.e. the camera was
// rotating just before going LOST (the fast-pan LOST scenario). Using a
// stale 0-history velocity here would be no better than QuickCovisibility.
// Radius 50px to cope with extra uncertainty from one missed frame.
// Returns true if ≥10 inlier MapPoints are confirmed after PoseOptimization.
// ===========================================================================
bool Tracking::QuickRotationRecovery()
{
    if (mRotVelHistory.empty())
        return false;

    // Only worthwhile when the velocity history shows actual rotation.
    {
        cv::Mat Rv = mRotVelHistory[0].rowRange(0,3).colRange(0,3);
        double tr  = Rv.at<float>(0,0)+Rv.at<float>(1,1)+Rv.at<float>(2,2);
        double rotDeg = std::acos(std::max(-1.0,std::min(1.0,(tr-1.0)/2.0)))*180.0/M_PI;
        if (rotDeg < kLargeRotThreshDeg)
        {
            std::cout << "[QuickRotRecov] Rotation too small (" << rotDeg
                      << " deg), skipping.\n";
            return false;
        }
    }

    cv::Mat predPose = PredictPoseWithRotationalVelocity();
    if (predPose.empty())
    {
        if (mLastFrame.mTcw.empty()) return false;
        predPose = mLastFrame.mTcw.clone();
    }
    mCurrentFrame.SetPose(predPose);

    // Rebuild local map from covisibility KFs if stale/empty (Issue 4 fix).
    // In the LOST branch UpdateLocalMap() was NOT called this frame, so
    // mvpLocalMapPoints may be from the last OK frame (possibly many frames
    // ago, i.e. already partially stale). Re-derive cheaply from the best
    // covisible KFs around mpReferenceKF.
    std::vector<MapPoint*> vpLocalMPs = mvpLocalMapPoints; // start with what we have
    if (vpLocalMPs.empty() && mpReferenceKF)
    {
        std::cout << "[QuickRotRecov] mvpLocalMapPoints empty, rebuilding from covis.\n";
        std::unordered_set<MapPoint*> seen;
        auto addKFMPs = [&](KeyFrame* kf) {
            if (!kf || kf->isBad()) return;
            for (auto* mp : kf->GetMapPointMatches())
                if (mp && !mp->isBad() && seen.insert(mp).second)
                    vpLocalMPs.push_back(mp);
        };
        addKFMPs(mpReferenceKF);
        for (auto* kf : mpReferenceKF->GetBestCovisibilityKeyFrames(10))
            addKFMPs(kf);
        std::cout << "[QuickRotRecov] Rebuilt " << vpLocalMPs.size() << " local MPs.\n";
    }

    if (vpLocalMPs.empty())
    {
        std::cout << "[QuickRotRecov] No local MapPoints available.\n";
        return false;
    }

    // Wide projection search — 50px radius for LOST uncertainty.
    ORBmatcher matcher(0.9, true);
    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
         static_cast<MapPoint*>(nullptr));
    fill(mCurrentFrame.mvbOutlier.begin(), mCurrentFrame.mvbOutlier.end(), false);

    int nmatches = matcher.SearchByProjection(mCurrentFrame, vpLocalMPs, 50);
    if (nmatches < 10)
    {
        // Try even wider as last resort.
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
             static_cast<MapPoint*>(nullptr));
        nmatches = matcher.SearchByProjection(mCurrentFrame, vpLocalMPs, 75);
    }

    std::cout << "[QuickRotRecov] Projection matches: " << nmatches << "\n";
    if (nmatches < 10) return false;

    Optimizer::PoseOptimization(&mCurrentFrame);

    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->mbTrackInView   = false;
                mCurrentFrame.mvpMapPoints[i]->mnLastFrameSeen = mCurrentFrame.mnId;
                mCurrentFrame.mvpMapPoints[i] = nullptr;
                mCurrentFrame.mvbOutlier[i]   = false;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }

    std::cout << "[QuickRotRecov] Inlier map after PoseOpt: " << nmatchesMap << "\n";
    if (nmatchesMap >= 10)
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }
    return false;
}


// ===========================================================================
// Compares the reference KF's current median scene depth against the
// snapshot taken right before tracking was lost. A ratio far from 1.0
// indicates the recovered pose (LK bridge / MultiSeed / Relocalization)
// has drifted in scale relative to the existing map.
// ===========================================================================
double Tracking::EstimateScaleFromDepth()
{
    if (mPreLostMedianDepth <= 0 || !mpReferenceKF)
        return -1.0;

    float cur = mpReferenceKF->ComputeSceneMedianDepth(2);
    if (cur <= 0)
        return -1.0;

    double ratio = cur / mPreLostMedianDepth;
    if (ratio < 0.1 || ratio > 10.0)
        return -1.0; // way too extreme to be a real scale jump, discard

    std::cout << "[Scale/Depth] pre=" << mPreLostMedianDepth
              << " cur=" << cur << " ratio=" << ratio << "\n";
    return ratio;
}

// ===========================================================================
// EstimateScaleFromObjects  — V1.1 addition (ported from v0.1)
// ===========================================================================
// Secondary scale source: compares detection bbox area against the
// projected bbox of the matching map object's ellipsoid. Useful when no
// pre-lost depth snapshot is available (e.g. lost very early, map still
// thin) but the scene contains tracked objects.
// ===========================================================================
double Tracking::EstimateScaleFromObjects(const cv::Mat& Tcw)
{
    auto objects = mpMap->GetAllMapObjects();
    if (objects.empty() || current_frame_good_detections_.empty())
        return -1.0;

    Matrix34d Rt = cvToEigenMatrix<double, float, 3, 4>(Tcw);
    Matrix34d P  = K_ * Rt;

    std::vector<double> ratios;
    for (const auto& det : current_frame_good_detections_)
    {
        float det_area = (det->bbox[2]-det->bbox[0]) * (det->bbox[3]-det->bbox[1]);
        if (det_area < 100) continue;

        double best_iou = 0.3, best_ratio = -1.0;
        for (auto* obj : objects)
        {
            if (obj->GetTrack()->GetCategoryId() != det->category_id) continue;
            if (obj->GetTrack()->IsBad()) continue;

            BBox2 proj_bb = obj->GetEllipsoid().project(P).ComputeBbox();
            float proj_area = (proj_bb[2]-proj_bb[0]) * (proj_bb[3]-proj_bb[1]);
            if (proj_area < 10) continue;

            double iou = bboxes_iou(det->bbox, proj_bb);
            if (iou > best_iou) {
                best_iou = iou;
                best_ratio = std::sqrt(det_area / proj_area); // area ratio ~ scale^2
            }
        }
        if (best_ratio > 0) ratios.push_back(best_ratio);
    }

    if (ratios.empty()) return -1.0;
    std::sort(ratios.begin(), ratios.end());
    double median = ratios[ratios.size() / 2];
    std::cout << "[Scale/Objects] " << ratios.size()
              << " pairs, median ratio=" << median << "\n";
    return median;
}

// ===========================================================================
// ApplyScaleCorrection  — V1.1 addition (ported from v0.1)
// ===========================================================================
void Tracking::ApplyScaleCorrection(double scale_ratio)
{
    if (scale_ratio < 0.2 || scale_ratio > 5.0)
    {
        std::cout << "[Scale] Ratio " << scale_ratio << " out of range [0.2, 5.0], skip.\n";
        return;
    }
    if (std::abs(scale_ratio - 1.0) < 0.05)
    {
        std::cout << "[Scale] Ratio " << scale_ratio << " negligible (<5%), skip.\n";
        return;
    }

    cv::Mat Tcw = mCurrentFrame.mTcw.clone();
    Tcw.rowRange(0,3).col(3) *= (float)scale_ratio;
    mCurrentFrame.SetPose(Tcw);

    // Velocity and LK seeds are no longer valid after a scale jump.
    mVelocity = cv::Mat();
    mLastGoodVelocity.release();
    mRotVelHistory.clear();
    mPrevGray.release();
    mvPrevPts.clear();
    mvPrevMPs.clear();
    mnLKFallbackFrames = 0;

    // Snapshot consumed.
    mPreLostMedianDepth = -1.0f;
    mPreLostTcw.release();

    std::cout << "[Scale] Correction applied: " << scale_ratio << "\n";
}

// ===========================================================================
// UpdateLastFrame  — unchanged from V0
// ===========================================================================
void Tracking::UpdateLastFrame()
{
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    if (!pRef || mlRelativeFramePoses.empty())
        return;
    cv::Mat Tlr = mlRelativeFramePoses.back();
    mLastFrame.SetPose(Tlr * pRef->GetPose());

    if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || !mbOnlyTracking)
        return;

    vector<pair<float,int>> vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for (int i = 0; i < mLastFrame.N; i++) {
        float z = mLastFrame.mvDepth[i];
        if (z > 0) vDepthIdx.push_back(make_pair(z, i));
    }
    if (vDepthIdx.empty()) return;
    sort(vDepthIdx.begin(), vDepthIdx.end());

    int nPoints = 0;
    for (size_t j = 0; j < vDepthIdx.size(); j++) {
        int i = vDepthIdx[j].second;
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        bool bCreateNew = (!pMP || pMP->Observations() < 1);
        if (bCreateNew) {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);
            mLastFrame.mvpMapPoints[i] = pNewMP;
            mlpTemporalPoints.push_back(pNewMP);
        }
        nPoints++;
        if (vDepthIdx[j].first > mThDepth && nPoints > 100) break;
    }
}

// ===========================================================================
// TrackWithMotionModel  — V1: adaptive search radius under occlusion
// ===========================================================================
bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9, true);
    UpdateLastFrame();

    if (mVelocity.empty() || mLastFrame.mTcw.empty())
        return false;

    // V1.3: when rotating fast, use averaged rotation velocity as seed —
    // the single-frame constant-velocity pose drifts behind by ~1 frame.
    double rotDeg = ComputeVelocityRotDeg();
    cv::Mat predPose = mVelocity * mLastFrame.mTcw; // default
    if (rotDeg > kLargeRotThreshDeg)
    {
        cv::Mat rotExtrap = PredictPoseWithRotationalVelocity();
        if (!rotExtrap.empty())
        {
            predPose = rotExtrap;
            std::cout << "[TrackMM] Using rot-extrapolated seed (rot="
                      << rotDeg << " deg)\n";
        }
    }
    mCurrentFrame.SetPose(predPose);
    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
         static_cast<MapPoint*>(NULL));

    int th = (mSensor != System::STEREO) ? 30 : 15;

    // V1: widen search when person covers large portion of frame
    if (mbUseMask) {
        const cv::Mat& emask = mCurrentFrame.GetExcludeMask();
        if (!emask.empty()) {
            int nz = cv::countNonZero(emask);
            float kept = float(nz) / (emask.rows * emask.cols);
            if (kept < 0.6f)
                th = std::min(th * 2, 100);
        }
    }

    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th,
                                               mSensor == System::MONOCULAR);
    if (nmatches < 20) {
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
             static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2*th,
                                               mSensor == System::MONOCULAR);
    }
    if (nmatches < 20) return false;

    Optimizer::PoseOptimization(&mCurrentFrame);

    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++) {
        if (mCurrentFrame.mvpMapPoints[i]) {
            if (mCurrentFrame.mvbOutlier[i]) {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                mCurrentFrame.mvpMapPoints[i] = nullptr;
                mCurrentFrame.mvbOutlier[i]   = false;
                pMP->mbTrackInView   = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }
    if (mbOnlyTracking) { mbVO = nmatchesMap < 10; return nmatches > 20; }
    return nmatchesMap >= 10;
}

// ===========================================================================
// TrackLocalMap  — V1: correct mask filter + flexible inlier threshold
// ===========================================================================
bool Tracking::TrackLocalMap()
{
    UpdateLocalMap();
    SearchLocalPoints();

    // ── V1: mask filter — protect ceiling MPs, penalise person MPs ───────────
    if (mbUseMask)
    {
        const cv::Mat& emask = mCurrentFrame.GetExcludeMask();
        if (!emask.empty())
        {
            // Find topmost person bbox → define safe ceiling zone
            float min_person_top = float(emask.rows);
            std::vector<cv::Rect> person_bboxes;
            person_bboxes.reserve(current_frame_good_detections_.size());
            for (const auto& det : current_frame_good_detections_) {
                if (det->category_id != PERSON_CLASS_ID) continue;
                if (det->bbox[1] < min_person_top) min_person_top = det->bbox[1];
                person_bboxes.emplace_back(
                    cvRound(det->bbox[0]), cvRound(det->bbox[1]), 
                    cvRound(det->bbox[2] - det->bbox[0]), 
                    cvRound(det->bbox[3] - det->bbox[1])
                );
            }
            const float safe_ceiling_y = min_person_top - 20.0f;

            int removed = 0;
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                const cv::KeyPoint& kp = mCurrentFrame.mvKeysUn[i];
                int cx = cvRound(kp.pt.x), cy = cvRound(kp.pt.y);
                if (cx < 0 || cy < 0 || cx >= emask.cols || cy >= emask.rows) continue;
                if (emask.at<uchar>(cy, cx) != 0) continue; // not in person region

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if (!pMP || pMP->isBad()) continue;

                // Protect well-established MPs
                if (pMP->Observations() >= 5) continue;

                // Protect ceiling MPs above all person bboxes
                if (float(cy) < safe_ceiling_y && pMP->Observations() >= 2) continue;

                // Check if actually inside any person bbox (not just erode overlap)
                if (pMP->Observations() >= 3) {
                    bool in_bbox = false;
                    cv::Point2i pt(cx, cy);
                    for (const auto& rect : person_bboxes) {
                        if (rect.contains(pt)) {
                            in_bbox = true; break;
                        }
                    }
                    if (!in_bbox) continue;
                }

                mCurrentFrame.mvbOutlier[i] = true;
                removed++;
            }
            if (removed > 0)
                std::cout << "[TrackLocalMap] Removed " << removed
                          << " person-region MPs\n";
        }
    }

    Optimizer::PoseOptimization(&mCurrentFrame);

    // ── Count inliers ─────────────────────────────────────────────────────────
    int nStable = 0;
    mnMatchesInliers = 0;
    for (int i = 0; i < mCurrentFrame.N; i++) {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if (!pMP) continue;
        if (!mCurrentFrame.mvbOutlier[i]) {
            pMP->IncreaseFound();
            if (!mbOnlyTracking) {
                if (pMP->Observations() > 0) {
                    mnMatchesInliers++;
                    if (pMP->Observations() >= 3) nStable++;
                }
            } else {
                mnMatchesInliers++;
            }
        } else if (mSensor == System::STEREO) {
            mCurrentFrame.mvpMapPoints[i] = nullptr;
        }
    }

    // ── V1: flexible inlier threshold ────────────────────────────────────────
    int thresh = 30; // V0 default

    if (mbUseMask && nStable < 15) {
        float personArea = 0, frameArea = float(mImGray.cols * mImGray.rows);
        for (const auto& det : current_frame_good_detections_) {
            if (det->category_id != PERSON_CLASS_ID) continue;
            personArea += (det->bbox[2]-det->bbox[0]) * (det->bbox[3]-det->bbox[1]);
        }
        if (personArea / frameArea > 0.25f) {
            thresh = 10; // hard floor — never go below this
            std::cout << "[TrackLocalMap] Heavy occlusion: thresh=" << thresh
                      << " stableInliers=" << nStable << "\n";
        }
    }

    // V1.9: sau QuickCovisibilityRecovery (cheap bridge), chỉ cần
    // mnMatchesInliers >= thresh (có thể = 10 khi occlusion). Điều kiện
    // "≥50 inliers ngay sau reloc" chỉ áp dụng cho full Relocalization()
    // thật sự (PnP RANSAC + nhiều KF candidate) — không phải QuickCovis.
    if (!mbLastRecoveryWasQuickCovis &&
        mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames &&
        mnMatchesInliers < 50)
        return false;
    mbLastRecoveryWasQuickCovis = false; // consumed
    if (mnMatchesInliers < thresh)
        return false;
    return true;
}

// ===========================================================================
// SearchLocalPoints  — V1: adaptive th under occlusion / post-LK
// ===========================================================================
void Tracking::SearchLocalPoints()
{
    for (auto* pMP : mCurrentFrame.mvpMapPoints) {
        if (pMP) {
            if (pMP->isBad()) pMP = nullptr;
            else {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView   = false;
            }
        }
    }

    int nToMatch = 0;
    for (auto* pMP : mvpLocalMapPoints) {
        if (pMP->mnLastFrameSeen == mCurrentFrame.mnId) continue;
        if (pMP->isBad()) continue;
        if (mCurrentFrame.isInFrustum(pMP, 0.5)) {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if (nToMatch > 0)
    {
        ORBmatcher matcher(0.8);
        int th = 3;
        if (mSensor == System::RGBD) th = 3;
        if (mCurrentFrame.mnId < mnLastRelocFrameId + 2) th = 5;

        // V1: widen when heavily occluded
        if (mbUseMask) {
            const cv::Mat& emask = mCurrentFrame.GetExcludeMask();
            if (!emask.empty()) {
                int nz = cv::countNonZero(emask);
                float kept = float(nz) / (emask.rows * emask.cols);
                if (kept < 0.6f) {
                    th = std::min(th * 3, 15);
                    std::cout << "[SearchLocalPoints] Occlusion: th=" << th << "\n";
                }
            }
        }

        // V1: widen after LK fallback (local map may be stale)
        if (mnLKFallbackFrames > 0) {
            th = std::max(th, 10);
            std::cout << "[SearchLocalPoints] Post-LK: th=" << th << "\n";
        }

        // V1.2: widen during inertia coasting — pose chỉ là ngoại suy, projection
        // của local map points có thể lệch vài px so với keypoint thật.
        if (mbInertiaMode) {
            th = std::max(th, 10);
            std::cout << "[SearchLocalPoints] Inertia: th=" << th << "\n";
        }

        matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
    }
}

// ===========================================================================
// UpdateLocalMap / UpdateLocalPoints / UpdateLocalKeyFrames — unchanged from V0
// ===========================================================================
void Tracking::UpdateLocalMap()
{
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();
    for (auto* pKF : mvpLocalKeyFrames) {
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();
        for (auto* pMP : vpMPs) {
            if (!pMP) continue;
            if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId) continue;
            if (!pMP->isBad()) {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalKeyFrames()
{
    map<KeyFrame*, int> keyframeCounter;
    for (int i = 0; i < mCurrentFrame.N; i++) {
        if (mCurrentFrame.mvpMapPoints[i]) {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if (!pMP->isBad()) {
                const map<KeyFrame*, size_t> obs = pMP->GetObservations();
                for (auto it = obs.begin(); it != obs.end(); it++)
                    keyframeCounter[it->first]++;
            } else {
                mCurrentFrame.mvpMapPoints[i] = nullptr;
            }
        }
    }
    if (keyframeCounter.empty()) return;

    int max = 0;
    KeyFrame* pKFmax = nullptr;
    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    for (auto it = keyframeCounter.begin(); it != keyframeCounter.end(); it++) {
        KeyFrame* pKF = it->first;
        if (pKF->isBad()) continue;
        if (it->second > max) { max = it->second; pKFmax = pKF; }
        mvpLocalKeyFrames.push_back(pKF);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    for (auto* pKF : mvpLocalKeyFrames) {
        if (mvpLocalKeyFrames.size() > 80) break;
        for (auto* pNeigh : pKF->GetBestCovisibilityKeyFrames(10)) {
            if (pNeigh && !pNeigh->isBad() &&
                pNeigh->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                mvpLocalKeyFrames.push_back(pNeigh);
                pNeigh->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                break;
            }
        }
        for (auto* pChild : pKF->GetChilds()) {
            if (pChild && !pChild->isBad() &&
                pChild->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                mvpLocalKeyFrames.push_back(pChild);
                pChild->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                break;
            }
        }
        KeyFrame* pParent = pKF->GetParent();
        if (pParent && pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
            mvpLocalKeyFrames.push_back(pParent);
            pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
            break;
        }
    }

    if (pKFmax) {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

// ===========================================================================
// NeedNewKeyFrame  — V1: motion guard + velocity sanity
// ===========================================================================
bool Tracking::NeedNewKeyFrame()
{
    if (mbOnlyTracking) return false;
    if (mbInertiaMode)  return false;
    if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) return false;

    // V1.3: post-reset boost — hạ ngưỡng trong kSoftResetBoostFrames frame
    // đầu sau khi relocalize thành công từ soft reset. Mục đích: densify map
    // nhanh (tạo KF sớm hơn) để tracking ổn định trở lại.
    // Đếm frame tracking thành công liên tiếp; khi đủ kSoftResetBoostFrames
    // thì tắt boost để tránh spam KF vô thời hạn.
    const bool bBoost = mbPostResetBoost;
    if (bBoost)
    {
        mnPostResetBoostFrames++;
        if (mnPostResetBoostFrames >= kSoftResetBoostFrames)
        {
            mbPostResetBoost = false;
            std::cout << "[NKF] Post-reset boost expired after "
                      << mnPostResetBoostFrames << " frames.\n";
        }
    }

    // V1.1: absolute KF inlier floor (ported from v0.1). Without this, a KF
    // can be inserted purely because mnMatchesInliers > 15 (see c2 below) even
    // when the scene is heavily occluded — letting low-quality KFs poison the
    // map. Floor is relaxed (but never removed) when a person covers a large
    // part of the frame.
    {
        int kfFloor = mKFMinInliersNoOcclusion;
        if (mbUseMask)
        {
            for (const auto& det : current_frame_good_detections_)
            {
                if (det->category_id != PERSON_CLASS_ID) continue;
                float area = (det->bbox[2]-det->bbox[0]) * (det->bbox[3]-det->bbox[1]);
                float frameArea = float((mImGray.empty() ? 480 : mImGray.rows) *
                                         (mImGray.empty() ? 640 : mImGray.cols));
                if (area / frameArea > 0.25f) { kfFloor = mKFMinInliersOcclusion; break; }
            }
        }
        if (mnMatchesInliers < kfFloor)
            return false;
    }

    const int nKFs = mpMap->KeyFramesInMap();
    // V1.3: boost mode bỏ qua cooldown mMaxFrames — ta muốn tạo KF ngay
    // sau khi relocalize xong để rebuild local map nhanh. Cooldown bình thường
    // ngăn KF spam sau relocalize nhưng với soft reset thì ta CẦN KF nhanh.
    if (!bBoost && mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
        return false;

    int nMinObs  = (nKFs <= 2) ? 2 : 3;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    int nTrackedClose = 0, nNonTrackedClose = 0;
    if (mSensor != System::MONOCULAR) {
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth) {
                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }
    bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

    float thRefRatio = 0.75f;
    if (nKFs < 2) thRefRatio = 0.4f;
    if (mSensor == System::MONOCULAR) {
        thRefRatio = 0.9f;
        // V1: easier KF insertion when rotating
        double rot = ComputeVelocityRotDeg();
        if (rot > 5.0) {
            thRefRatio = std::min(thRefRatio, 0.75f);
            std::cout << "[NKF] rot=" << rot << " deg -> thRefRatio=" << thRefRatio << "\n";
        }
        // V1.3: post-reset boost — insert KF aggressively to rebuild local map
        if (bBoost) {
            thRefRatio = std::min(thRefRatio, 0.65f);
            std::cout << "[NKF] Boost frame " << mnPostResetBoostFrames
                      << " -> thRefRatio=" << thRefRatio << "\n";
        }
    }

    const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
    const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
    const bool c1c = (mSensor != System::MONOCULAR &&
                      (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose));
    const bool c2  = ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose)
                       && mnMatchesInliers > 15);

    // V1: motion guard — don't spam KF when camera is still
    bool bEnoughMotion = true;
    if (mSensor == System::MONOCULAR && !mVelocity.empty()) {
        double rot   = ComputeVelocityRotDeg();
        double trans = cv::norm(mVelocity.rowRange(0,3).col(3));
        bEnoughMotion = (rot > 2.0 || trans > 0.003);
    }

    if ((c1a || c1b || c1c) && c2 && bEnoughMotion)
    {
        if (bLocalMappingIdle) return true;
        else {
            mpLocalMapper->InterruptBA();
            if (mSensor != System::MONOCULAR)
                return mpLocalMapper->KeyframesInQueue() < 3;
            return false;
        }
    }
    return false;
}

// ===========================================================================
// CreateNewKeyFrame  — unchanged from V0
// ===========================================================================
void Tracking::CreateNewKeyFrame()
{
    if (!mpLocalMapper->SetNotStop(true)) return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if (mSensor != System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();
        vector<pair<float,int>> vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for (int i = 0; i < mCurrentFrame.N; i++) {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0) vDepthIdx.push_back(make_pair(z, i));
        }
        if (!vDepthIdx.empty()) {
            sort(vDepthIdx.begin(), vDepthIdx.end());
            int nPoints = 0;
            for (size_t j = 0; j < vDepthIdx.size(); j++) {
                int i = vDepthIdx[j].second;
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                bool bCreateNew = (!pMP || pMP->Observations() < 1);
                if (pMP && pMP->Observations() < 1)
                    mCurrentFrame.mvpMapPoints[i] = nullptr;
                if (bCreateNew) {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D, pKF, mpMap);
                    pNewMP->AddObservation(pKF, i);
                    pKF->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);
                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                }
                nPoints++;
                if (vDepthIdx[j].first > mThDepth && nPoints > 100) break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);
    mpLocalMapper->SetNotStop(false);
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame   = pKF;
}

// ===========================================================================
// TrackByProjectionFastPan  — V1.3
// ===========================================================================
// Called when rotation > kLargeRotThreshDeg AND BoW-based methods fail.
// Uses PredictPoseWithRotationalVelocity() as seed, then does a wide
// SearchByProjection over all local map points (no descriptor matching bias
// toward a specific KF).  On a fast pan this frequently finds 20-40 inliers
// that MultiSeed misses because the KF's BoW descriptor pool is at a very
// different viewing angle.
// ===========================================================================
bool Tracking::TrackByProjectionFastPan()
{
    // Build seed pose
    cv::Mat predPose = PredictPoseWithRotationalVelocity();
    if (predPose.empty())
    {
        // No history yet — fall back to last-frame pose as seed
        if (mLastFrame.mTcw.empty()) return false;
        predPose = mLastFrame.mTcw.clone();
    }
    mCurrentFrame.SetPose(predPose);

    // Collect local map points (normally already filled by UpdateLocalMap
    // before this call). V1.9 fix: TrackByProjectionFastPan() is called from
    // the OK branch (large-rotation path), where UpdateLocalMap() only runs
    // AFTER TrackLocalMap() succeeds — not before this function — so on the
    // very first large-rotation frame after init/relocalize, or whenever the
    // previous frame failed before reaching TrackLocalMap(), mvpLocalMapPoints
    // can still be empty/stale. Rather than failing outright (same gap as
    // Issue 4 in QuickRotationRecovery), rebuild cheaply from mpReferenceKF's
    // covisibility neighbourhood before giving up.
    std::vector<MapPoint*> vpLocalMPs = mvpLocalMapPoints;
    if (vpLocalMPs.empty() && mpReferenceKF)
    {
        std::cout << "[FastPan] mvpLocalMapPoints empty, rebuilding from covis.\n";
        std::unordered_set<MapPoint*> seen;
        auto addKFMPs = [&](KeyFrame* kf) {
            if (!kf || kf->isBad()) return;
            for (auto* mp : kf->GetMapPointMatches())
                if (mp && !mp->isBad() && seen.insert(mp).second)
                    vpLocalMPs.push_back(mp);
        };
        addKFMPs(mpReferenceKF);
        for (auto* kf : mpReferenceKF->GetBestCovisibilityKeyFrames(10))
            addKFMPs(kf);
        std::cout << "[FastPan] Rebuilt " << vpLocalMPs.size() << " local MPs.\n";
    }
    if (vpLocalMPs.empty()) return false;

    // Wide projection search — th=15 captures up to ~15px reprojection error
    // which corresponds to ~15° pan at typical focal lengths.
    ORBmatcher matcher(0.9, true);
    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
         static_cast<MapPoint*>(NULL));

    int nmatches = matcher.SearchByProjection(mCurrentFrame, vpLocalMPs, 15);
    if (nmatches < 10)
    {
        // Try even wider
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
             static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame, vpLocalMPs, 25);
    }

    std::cout << "[FastPan] Projection matches: " << nmatches << "\n";
    if (nmatches < 10) return false;

    Optimizer::PoseOptimization(&mCurrentFrame);

    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->mbTrackInView   = false;
                mCurrentFrame.mvpMapPoints[i]->mnLastFrameSeen = mCurrentFrame.mnId;
                mCurrentFrame.mvpMapPoints[i] = nullptr;
                mCurrentFrame.mvbOutlier[i]   = false;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }
    std::cout << "[FastPan] InlierMap after PoseOpt: " << nmatchesMap << "\n";
    return nmatchesMap >= 10;
}

// ===========================================================================
// TrackWithOpticalFlow  — V1 LK bridge (MapPoints only, no AI depth)
// ===========================================================================
bool Tracking::TrackWithOpticalFlow()
{
    if (mPrevGray.empty() || mvPrevPts.empty() || mvPrevMPs.empty()) {
        std::cout << "[LK] No prev data, skip.\n";
        return false;
    }

    // ── 1. Forward LK track ──────────────────────────────────────────────────
    std::vector<cv::Point2f> curr_pts;
    std::vector<uchar> status_fwd;
    std::vector<float> err_fwd;
    cv::calcOpticalFlowPyrLK(mPrevGray, mImGray, mvPrevPts, curr_pts,
                              status_fwd, err_fwd,
                              cv::Size(kLKWinSize, kLKWinSize), kLKPyrLevels,
                              cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,30,0.01));

    // ── 2. Backward check ────────────────────────────────────────────────────
    std::vector<cv::Point2f> back_pts;
    std::vector<uchar> status_bwd;
    std::vector<float> err_bwd;
    cv::calcOpticalFlowPyrLK(mImGray, mPrevGray, curr_pts, back_pts,
                              status_bwd, err_bwd,
                              cv::Size(kLKWinSize, kLKWinSize), kLKPyrLevels,
                              cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,30,0.01));

    // ── 3. Filter valid correspondences ──────────────────────────────────────
    const float FB_THRESH  = 2.0f;
    const float ERR_THRESH = 30.0f;

    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;
    std::vector<MapPoint*>   valid_mps;

    for (size_t i = 0; i < mvPrevPts.size(); ++i)
    {
        if (!status_fwd[i] || !status_bwd[i]) continue;
        if (cv::norm(mvPrevPts[i] - back_pts[i]) > FB_THRESH) continue;
        if (err_fwd[i] > ERR_THRESH) continue;

        const cv::Point2f& cp = curr_pts[i];
        if (cp.x < 0 || cp.y < 0 || cp.x >= mImGray.cols || cp.y >= mImGray.rows) continue;

        MapPoint* pMP = mvPrevMPs[i];
        if (!pMP || pMP->isBad()) continue;

        cv::Mat wpos = pMP->GetWorldPos();
        pts_3d.emplace_back(wpos.at<float>(0), wpos.at<float>(1), wpos.at<float>(2));
        pts_2d.push_back(cp);
        valid_mps.push_back(pMP);
    }

    std::cout << "[LK] Valid pts after FB check: " << pts_3d.size()
              << "/" << mvPrevPts.size() << "\n";

    if ((int)pts_3d.size() < 8) {
        std::cout << "[LK] Too few pts.\n";
        return false;
    }

    // ── 4. PnP RANSAC ────────────────────────────────────────────────────────
    cv::Mat rvec, tvec;
    std::vector<int> inliers_idx;
    bool use_guess = false;
    if (!mLastFrame.mTcw.empty()) {
        cv::Rodrigues(mLastFrame.mTcw.rowRange(0,3).colRange(0,3), rvec);
        tvec = mLastFrame.mTcw.rowRange(0,3).col(3).clone();
        use_guess = true;
    }

    bool pnp_ok = cv::solvePnPRansac(pts_3d, pts_2d, mK, mDistCoef,
                                       rvec, tvec, use_guess,
                                       100, 4.0f, 0.99, inliers_idx,
                                       cv::SOLVEPNP_ITERATIVE);

    if (!pnp_ok || (int)inliers_idx.size() < 6) {
        std::cout << "[LK] PnP failed (" << inliers_idx.size() << " inliers).\n";
        return false;
    }
    std::cout << "[LK] PnP inliers: " << inliers_idx.size() << "\n";

    // ── 5. Set pose ──────────────────────────────────────────────────────────
    cv::Mat R_mat;
    cv::Rodrigues(rvec, R_mat);
    R_mat.convertTo(R_mat, CV_32F);
    tvec.convertTo(tvec, CV_32F);

    cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
    R_mat.copyTo(Tcw.rowRange(0,3).colRange(0,3));
    tvec.copyTo(Tcw.rowRange(0,3).col(3));
    mCurrentFrame.SetPose(Tcw);

    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);
    fill(mCurrentFrame.mvbOutlier.begin(),   mCurrentFrame.mvbOutlier.end(),   false);

    // Assign MapPoints to nearest keypoints
    for (int idx : inliers_idx) {
        const cv::Point2f& p = pts_2d[idx];
        float best_d = 4.0f; int best_k = -1;
        for (int k = 0; k < mCurrentFrame.N; ++k) {
            float d = std::hypot(mCurrentFrame.mvKeysUn[k].pt.x - p.x,
                                 mCurrentFrame.mvKeysUn[k].pt.y - p.y);
            if (d < best_d) { best_d = d; best_k = k; }
        }
        if (best_k >= 0) mCurrentFrame.mvpMapPoints[best_k] = valid_mps[idx];
    }

    // ── 6. Refine with PoseOptimization ──────────────────────────────────────
    Optimizer::PoseOptimization(&mCurrentFrame);

    int n_final = 0;
    for (int i = 0; i < mCurrentFrame.N; ++i)
        if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
            n_final++;

    std::cout << "[LK] Final inliers after PoseOpt: " << n_final << "\n";
    if (n_final < 6) return false;

    // ── 7. Update LK seeds ───────────────────────────────────────────────────
    mImGray.copyTo(mPrevGray);
    mvPrevPts.clear(); mvPrevMPs.clear();
    for (int i = 0; i < mCurrentFrame.N; ++i) {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if (pMP && !mCurrentFrame.mvbOutlier[i] && !pMP->isBad()) {
            mvPrevPts.push_back(mCurrentFrame.mvKeysUn[i].pt);
            mvPrevMPs.push_back(pMP);
        }
    }

    mnLKFallbackFrames++;
    return true;
}

// ===========================================================================
// ===========================================================================
// IsStaticAnchorClass  — V1.3
// ===========================================================================
// Returns true only for COCO classes that are:
//   (a) not alive (no people, animals)
//   (b) not self-propelled (no vehicles in motion)
//   (c) not easily carried away by hand (no laptops, bottles, bags, etc.)
//
// COCO class IDs that are EXCLUDED (dynamic / potentially dynamic):
//   0          person
//   1–8        bicycle, car, motorcycle, airplane, bus, train, truck, boat
//   14–23      bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
//   24–28      backpack, umbrella, handbag, tie, suitcase
//   32–41      skis, snowboard, sports ball, kite, baseball bat, baseball glove,
//              skateboard, surfboard, tennis racket, bottle
//   42–46      wine glass, cup, fork, knife, spoon
//   47–55      bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog,
//              pizza, donut, cake
//   63–67      laptop, mouse, remote, keyboard, cell phone
//   73–74      book, clock
//   76–79      scissors, teddy bear, hair drier, toothbrush
//
// COCO class IDs that ARE kept as static anchors:
//   9–13       traffic light, fire hydrant, stop sign, parking meter, bench
//   56–62      chair, couch, potted plant, bed, dining table, toilet, tv
//   68–72      microwave, oven, toaster, sink, refrigerator
//   75         vase
//
// Special cases:
//   chair (56)        — kept; large enough to be triangulated reliably.
//                       Can be pushed but unlikely to move between sessions.
//   potted plant (58) — kept; usually large floor plants in lobbies.
//   bench (13)        — kept; outdoor/indoor fixed seating.
//   parked vehicles (1–8) — excluded. We cannot distinguish parked vs moving
//                            from class_id alone, so we conservatively exclude.
// ===========================================================================
bool IsStaticAnchorClass(int class_id)
{
    // Whitelist approach: only IDs known to be structurally fixed are allowed.
    // Everything not on this list is treated as potentially dynamic.
    static const std::unordered_set<int> kStaticAnchorClasses = {
        // Infrastructure / outdoor furniture
        9,   // traffic light   (pole-mounted, fixed)
        10,  // fire hydrant    (ground-mounted, fixed)
        11,  // stop sign       (pole-mounted, fixed)
        12,  // parking meter   (pole-mounted, fixed)
        13,  // bench           (fixed or very heavy)
        // Indoor furniture — large, rarely moved
        56,  // chair           (can be moved but heavy enough for short sessions)
        57,  // couch / sofa    (very heavy, essentially fixed)
        58,  // potted plant    (large floor plants in lobbies)
        59,  // bed             (fixed)
        60,  // dining table    (fixed in typical building lobbies)
        61,  // toilet          (plumbing-fixed)
        62,  // tv / monitor    (wall-mounted or fixed stand)
        // Kitchen / lab fixtures (large, plumbing/wiring attached)
        68,  // microwave       (counter-fixed)
        69,  // oven            (built-in)
        70,  // toaster         (counter item — borderline, kept for coverage)
        71,  // sink            (plumbing-fixed)
        72,  // refrigerator    (very heavy, fixed)
        // Decor
        75,  // vase            (large floor vases in lobbies are fixed)
    };
    return kStaticAnchorClasses.count(class_id) > 0;
}

// ===========================================================================
// RelocalizationFromObjects  — V1.3: static anchor filter
// ===========================================================================
bool Tracking::RelocalizationFromObjects(bool use_points)
{
    // ── Build detection list — static anchor classes only ────────────────────
    std::vector<BBox2, Eigen::aligned_allocator<BBox2>> bboxes;
    std::vector<size_t> bboxes_categories;
    int n_det_skipped = 0;
    for (auto det : current_frame_good_detections_) {
        if (det->score <= 0.1) continue;
        if (!IsStaticAnchorClass(det->category_id)) { n_det_skipped++; continue; }
        bboxes.push_back(det->bbox);
        bboxes_categories.push_back(det->category_id);
    }

    // ── Build map object list — static anchor classes + minimum observations ─
    // Require obs > 20 to avoid using freshly-triangulated noisy ellipsoids.
    std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>> ellipsoids;
    std::vector<size_t> ellipsoids_categories;
    int n_obj_skipped = 0;
    for (auto* obj : mpMap->GetAllMapObjects()) {
        int cid = static_cast<int>(obj->GetTrack()->GetCategoryId());
        if (!IsStaticAnchorClass(cid)) { n_obj_skipped++; continue; }
        if (obj->GetTrack()->IsBad()) { n_obj_skipped++; continue; }
        if (obj->GetTrack()->GetNbObservations() < 20) { n_obj_skipped++; continue; }
        ellipsoids.push_back(obj->GetEllipsoid());
        ellipsoids_categories.push_back(static_cast<size_t>(cid));
    }

    std::cout << "[RelocFromObjects] anchors: " << bboxes.size() << " det / "
              << ellipsoids.size() << " map  (skipped det=" << n_det_skipped
              << " obj=" << n_obj_skipped << ")\n";

    // Need at least 1 detection and 1 map anchor to attempt pose estimation.
    // solveP3P_ransac internally requires ≥3 pairs for a full solution;
    // returning false here quickly avoids spinning with empty inputs.
    if (bboxes.empty() || ellipsoids.empty()) {
        std::cout << "[RelocFromObjects] Not enough static anchors, skip.\n";
        return false;
    }

    auto [best_index, poses, scores, used_pairs, used_inliers] =
        ORB_SLAM2::solveP3P_ransac(ellipsoids, ellipsoids_categories,
                                   bboxes, bboxes_categories, K_, 0.2);
    if (best_index == -1 || poses.empty() || scores[best_index] > bboxes.size())
        return false;

    std::vector<int> indices(poses.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&scores](int a, int b){ return scores[a] < scores[b]; });

    cv::Mat best_Rt_objects;
    for (int i = 0; i < (int)indices.size() && scores[indices[i]] < bboxes.size(); ++i)
    {
        int bi = indices[i];
        const auto& pose_est = poses[bi];
        Eigen::Matrix3d o_est = pose_est.block<3,3>(0,0);
        Eigen::Vector3d p_est = pose_est.col(3);
        Matrix34d Rt_est;
        Rt_est << o_est.transpose(), -o_est.transpose() * p_est;
        cv::Mat Rt(4, 4, CV_32F, 0.0);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 4; c++)
                Rt.at<float>(r, c) = Rt_est(r, c);
        Rt.at<float>(3, 3) = 1.0;
        mCurrentFrame.SetPose(Rt);

        std::vector<Ellipse, Eigen::aligned_allocator<Ellipse>> ellipses_in;
        std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>> ellipsoids_in;
        for (auto x : used_inliers[bi]) {
            ellipses_in.push_back(Ellipse::FromBbox(bboxes[x.first]));
            ellipsoids_in.push_back(ellipsoids[x.second]);
        }
        Rt = OptimizePoseFromObjects(ellipses_in, ellipsoids_in, Rt, K_);
        if (i == 0) best_Rt_objects = Rt;

        ORBmatcher matcher(0.9, true);
        int nb_matches = matcher.SearchByProjectionAllPoints(
            mCurrentFrame, mpMap->GetAllMapPoints(), 10, 100);
        std::cout << "[RelocFromObjects] " << nb_matches << " ORB matches after object pose.\n";

        if (nb_matches > 10) {
            if (use_points) Optimizer::PoseOptimization(&mCurrentFrame);
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
            return true;
        }
    }
    mCurrentFrame.SetPose(best_Rt_objects);
    mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
    return true;
}

// ===========================================================================
// Relocalization  — V1: min match lowered to 10 (from 15) for low-texture scenes
// ===========================================================================
bool Tracking::Relocalization()
{
    mLastProcessedState = mState;
    mCurrentFrame.ComputeBoW();

    vector<KeyFrame*> vpCandidateKFs =
        mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);
    if (vpCandidateKFs.empty()) return false;

    // V1.4: relax the good-inlier threshold during the post-Soft-Reset boost
    // window. The camera is known to still be inside the preserved map (see
    // MapPreservingReset) — TrackLocalMap() right after this will still reject
    // a genuinely bad pose, so a lower bar here just gives more recovery
    // attempts a chance to succeed instead of being discarded outright.
    const int minGood = mbPostResetBoost
        ? std::max(15, mRelocMinGoodInliers / 2)
        : mRelocMinGoodInliers;

    const int nKFs = vpCandidateKFs.size();
    ORBmatcher matcher(0.75, true);
    vector<PnPsolver*>          vpPnPsolvers(nKFs);
    vector<vector<MapPoint*>>   vvpMapPointMatches(nKFs);
    vector<bool>                vbDiscarded(nKFs, false);
    int nCandidates = 0;

    for (int i = 0; i < nKFs; i++) {
        KeyFrame* pKF = vpCandidateKFs[i];
        if (pKF->isBad()) { vbDiscarded[i] = true; continue; }
        int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
        if (nmatches < 10) { // V1: 10 instead of V0's 15
            vbDiscarded[i] = true; continue;
        }
        PnPsolver* pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
        pSolver->SetRansacParameters(0.99, 8, 500, 4, 0.5, 5.991);
        vpPnPsolvers[i] = pSolver;
        nCandidates++;
    }

    bool bMatch = false;
    ORBmatcher matcher2(0.9, true);

    while (nCandidates > 0 && !bMatch)
    {
        for (int i = 0; i < nKFs; i++)
        {
            if (vbDiscarded[i]) continue;
            vector<bool> vbInliers; int nInliers; bool bNoMore;
            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);
            if (bNoMore) { vbDiscarded[i] = true; nCandidates--; }
            if (Tcw.empty()) continue;

            Tcw.copyTo(mCurrentFrame.mTcw);
            set<MapPoint*> sFound;
            for (int j = 0; j < (int)vbInliers.size(); j++) {
                if (vbInliers[j]) {
                    mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                    sFound.insert(vvpMapPointMatches[i][j]);
                } else mCurrentFrame.mvpMapPoints[j] = nullptr;
            }

            int nGood = Optimizer::PoseOptimization(&mCurrentFrame);
            if (nGood < 10) continue;

            for (int io = 0; io < mCurrentFrame.N; io++)
                if (mCurrentFrame.mvbOutlier[io])
                    mCurrentFrame.mvpMapPoints[io] = nullptr;

            // V1.1/V1.4: configurable + boost-relaxed instead of hard-coded 50
            if (nGood < minGood) {
                int nadditional = matcher2.SearchByProjection(
                    mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);
                if (nadditional + nGood >= minGood) {
                    nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                    if (nGood > (int)(minGood * 0.6) && nGood < minGood) {
                        sFound.clear();
                        for (int ip = 0; ip < mCurrentFrame.N; ip++)
                            if (mCurrentFrame.mvpMapPoints[ip])
                                sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                        nadditional = matcher2.SearchByProjection(
                            mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);
                        if (nGood + nadditional >= minGood) {
                            nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                            for (int io = 0; io < mCurrentFrame.N; io++)
                                if (mCurrentFrame.mvbOutlier[io])
                                    mCurrentFrame.mvpMapPoints[io] = nullptr;
                        }
                    }
                }
            }
            if (nGood >= minGood) { bMatch = true; break; }
        }
    }

    if (!bMatch) return false;
    mnLastRelocFrameId = mCurrentFrame.mnId;
    return true;
}

// ===========================================================================
// MapPreservingReset  — V1.3
// ===========================================================================
// Drops only tracking state; keeps map + KF database intact so the next
// Track() call can relocalize back into the existing map.
// If the map has object ellipsoids (OA-SLAM) RelocalizationFromObjects() will
// attempt to recover pose from bbox↔ellipsoid matching, which works even
// when the camera has spun 180° and BoW fails completely.
// ===========================================================================
bool Tracking::MapPreservingReset()
{
    const int nKFs = mpMap->KeyFramesInMap();
    if (nKFs < 5)
    {
        std::cout << "[SoftReset] Map too small (" << nKFs
                  << " KFs), recommending hard reset.\n";
        return false;
    }

    std::cout << "[SoftReset] Preserving map (" << nKFs << " KFs, "
              << mpMap->MapPointsInMap() << " MPs, "
              << mpMap->GetNumberMapObjects() << " objects).\n";

    // \u2500\u2500 1. Drop motion state \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    mVelocity.release();
    mLastGoodVelocity.release();
    mRotVelHistory.clear();
    mbInertiaMode   = false;
    mnInertiaFrames = 0;
    mnConsecutiveLostFrames = 0;

    // Drop LK bridge state
    mPrevGray.release();
    mvPrevPts.clear();
    mvPrevMPs.clear();
    mnLKFallbackFrames = 0;

    // Drop scale-guard snapshot
    mPreLostMedianDepth = -1.0f;
    mPreLostTcw.release();
    mnPreLostInliers    = 0;

    // Drop monocular initializer
    if (mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    // Seed mLastFrame tu KF gan nhat trong map.
    // QuickCovisibilityRecovery va Relocalization dung mLastFrame.mTcw lam
    // seed. Neu mLastFrame stale, projection search se miss het.
    // Dung mpReferenceKF lam placeholder de recovery co diem khoi dau hop ly.
    KeyFrame* pSeedKF = nullptr;
    if (mpReferenceKF && !mpReferenceKF->isBad())
    {
        pSeedKF = mpReferenceKF;
    }
    else
    {
        int bestObs = 0;
        for (auto* kf : mpMap->GetAllKeyFrames())
        {
            if (!kf || kf->isBad()) continue;
            int obs = static_cast<int>(kf->GetMapPointMatches().size());
            if (obs > bestObs) { bestObs = obs; pSeedKF = kf; }
        }
    }
    if (pSeedKF)
    {
        mLastFrame.SetPose(pSeedKF->GetPose());
        mLastFrame.mpReferenceKF = pSeedKF;
        mpReferenceKF = pSeedKF;
        std::cout << "[SoftReset] Seeded mLastFrame from KF "
                  << pSeedKF->mnId << "\n";
    }

    // Activate post-reset KF boost: NeedNewKeyFrame() se ha nguong tam thoi
    // de rebuild local map nhanh hon sau khi relocalize thanh cong.
    mbPostResetBoost       = true;
    mnPostResetBoostFrames = 0;

    // V1.5: reset stale-recovery watchdog — start counting LOST frames fresh
    // from this soft reset.
    mnPostResetLostFrames = 0;

    // Bo qua cooldown mnLastRelocFrameId + mMaxFrames: muon tao KF ngay sau
    // relocalize de densify map.
    mnLastRelocFrameId = 0;

    // State -> LOST -> Track() vao recovery pipeline ngay lap tuc.
    mState = LOST;

    std::cout << "[SoftReset] Done. Next frame will attempt relocalization.\n";
    return true;
}

// ===========================================================================
// PredictPoseWithRotationalVelocity  — V1.3
// ===========================================================================
// Averages the rotation component of the last N good velocities (via log-map
// averaging on SO(3)) and composes it with the last frame's pose to produce
// a better initial seed for TrackWithMotionModel / TrackByProjectionFastPan
// when the camera is in a fast pan.
//
// On a fast pan the single-frame constant-velocity model is already one step
// behind by the time we process the current frame (processing lag ~1 frame at
// 15fps).  Averaging recent velocities gives a smoother rotation estimate and
// reduces the chance that the projected window misses all MapPoints.
// ===========================================================================
cv::Mat Tracking::PredictPoseWithRotationalVelocity() const
{
    if (mRotVelHistory.empty() || mLastFrame.mTcw.empty())
        return cv::Mat();

    // Use the most recent good velocity directly if we only have 1 sample
    const cv::Mat& vel = mRotVelHistory[0];
    if (mRotVelHistory.size() == 1)
        return vel * mLastFrame.mTcw;

    // Average rotations via Rodrigues (simple linear average on rotation vector)
    // This is an approximation but good enough for small inter-frame angles.
    cv::Mat rvec_sum = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat t_sum    = cv::Mat::zeros(3, 1, CV_64F);
    int n = static_cast<int>(mRotVelHistory.size());

    for (const auto& v : mRotVelHistory)
    {
        cv::Mat R = v.rowRange(0,3).colRange(0,3);
        cv::Mat t = v.rowRange(0,3).col(3);
        cv::Mat rvec;
        cv::Rodrigues(R, rvec);
        rvec.convertTo(rvec, CV_64F);
        t.convertTo(t, CV_64F);
        rvec_sum += rvec;
        t_sum    += t;
    }
    rvec_sum /= n;
    t_sum    /= n;

    cv::Mat R_avg;
    cv::Rodrigues(rvec_sum, R_avg);
    R_avg.convertTo(R_avg, CV_32F);
    t_sum.convertTo(t_sum, CV_32F);

    cv::Mat vel_avg = cv::Mat::eye(4, 4, CV_32F);
    R_avg.copyTo(vel_avg.rowRange(0,3).colRange(0,3));
    t_sum.copyTo(vel_avg.rowRange(0,3).col(3));

    return vel_avg * mLastFrame.mTcw;
}

// ===========================================================================
// Reset  — V1: also reset LK state
// ===========================================================================
void Tracking::Reset()
{
    cout << "System Reseting\n";
    if (mpViewer) {
        mpViewer->RequestStop();
        while (!mpViewer->isStopped()) usleep(3000);
    }

    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done\n";

    if (local_object_mapper_) {
        cout << "Reseting Local Object Mapper...";
        local_object_mapper_->RequestReset();
        cout << " done\n";
    }

    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done\n";

    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done\n";

    objectTracks_.clear();
    mpMap->clear();
    KeyFrame::nNextId = 0;
    Frame::nNextId    = 0;
    mState            = NO_IMAGES_YET;
    mVelocity         = cv::Mat();
    mLastFrame        = Frame();
    mCurrentFrame     = Frame();

    // V1: reset consecutive lost counter
    mnConsecutiveLostFrames = 0;

    // V1.3: reset post-reset boost
    mbPostResetBoost       = false;
    mnPostResetBoostFrames = 0;

    // V1: reset LK bridge state
    mPrevGray.release();
    mvPrevPts.clear();
    mvPrevMPs.clear();
    mnLKFallbackFrames = 0;

    // V1.3: reset rotation velocity history
    mRotVelHistory.clear();

    // V1.1: reset pre-lost snapshot state (stale after a hard reset)
    mPreLostMedianDepth = -1.0f;
    mPreLostTcw.release();
    mnPreLostInliers = 0;

    // V1.2: reset inertia state
    mbInertiaMode = false;
    mnInertiaFrames = 0;
    mLastGoodVelocity.release();

    if (mpInitializer) {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if (mpViewer) mpViewer->Release();
}

// ===========================================================================
// ChangeCalibration  — unchanged from V0
// ===========================================================================
void Tracking::ChangeCalibration(const string& strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"], fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"], cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0,0) = fx; K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx; K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0) { DistCoef.resize(5); DistCoef.at<float>(4) = k3; }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];
    Frame::mbInitialComputations = true;
}

// ===========================================================================
// InformOnlyTracking  — unchanged from V0
// ===========================================================================
void Tracking::InformOnlyTracking(const bool& flag)
{
    mbOnlyTracking = flag;
}

} // namespace ORB_SLAM2