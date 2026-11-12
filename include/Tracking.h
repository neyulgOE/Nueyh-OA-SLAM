/**
 * This file is part of OA-SLAM.
 * Copyright (C) 2022 Matthieu Zins <matthieu.zins@inria.fr>
 * OA-SLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

/**
 * This file is part of ORB-SLAM2.
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es>
 * ORB-SLAM2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Viewer.h"
#include "FrameDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"
#include "Utils.h"
#include "ObjectTrack.h"

#include <atomic>
#include <mutex>
#include <thread>
#include <Eigen/Dense>
#include <condition_variable>

namespace ORB_SLAM2
{

// PERSON_CLASS_ID: YOLO/COCO class 0 = person
const int PERSON_CLASS_ID = 0;

// Returns true for COCO class IDs that represent structurally fixed objects
// (furniture, infrastructure) which can safely serve as relocalization anchors.
// Dynamic / moveable objects (people, animals, vehicles, handheld items) return false.
// Defined in Tracking.cc; free function usable from any translation unit.
bool IsStaticAnchorClass(int class_id);

class Viewer;
class ARViewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LocalObjectMapping;
class LoopClosing;
class System;
class Detection;
class Ellipsoid;

class Tracking
{
public:
    Tracking(System* pSys, ORBVocabulary* pVoc,
             FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer,
             Map* pMap, KeyFrameDatabase* pKFDB,
             const string& strSettingPath, const int sensor);

    // V1.6: stops + joins the async recovery thread cleanly.
    ~Tracking();

    // Grab functions — preprocess + call Track()
    cv::Mat GrabImageStereo(const cv::Mat& imRectLeft,
                            const cv::Mat& imRectRight,
                            const double& timestamp);

    cv::Mat GrabImageStereo_object(const cv::Mat& imRectLeft,
                                   const cv::Mat& imRectRight,
                                   const double& timestamp,
                                   const std::vector<Detection::Ptr>& detections,
                                   bool force_relocalize);

    cv::Mat GrabImageRGBD(const cv::Mat& imRGB,
                          const cv::Mat& imD,
                          const double& timestamp);

    cv::Mat GrabImageMonocular(const cv::Mat& im,
                               const double& timestamp,
                               const std::vector<Detection::Ptr>& detections,
                               bool force_relocalize);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLocalObjectMapper(LocalObjectMapping* pLocalObjectMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);
    void SetARViewer(ARViewer* pARViewer);
    
    void SetMap(Map* pMap) { mpMap = pMap; }

    void ChangeCalibration(const string& strSettingPath);
    void InformOnlyTracking(const bool& flag);

    const System* GetSystem() const { return mpSystem; }

    void Reset();

    // ── V1.3: Map-preserving soft reset ──────────────────────────────────────
    // Drops only the tracking state (velocity, LK seeds, initializer) and sets
    // mState = LOST so the next Track() call immediately enters the recovery
    // pipeline (QuickCovisibilityRecovery → LK → RelocalizationFromObjects →
    // Relocalization) WITHOUT wiping the map or the KF database.
    // If the map contains static-anchor object ellipsoids the object-based
    // relocalizer can find the camera pose even after a hard spin.
    // Returns true if the map is large enough to make soft-reset meaningful
    // (≥5 KFs); returns false (caller should fall back to hard Reset()) if
    // the map is too small to relocalize into.
    bool MapPreservingReset();

    // ── Object track accessors ───────────────────────────────────────────────
    const std::list<ObjectTrack::Ptr>& GetObjectTracks() { return objectTracks_; }
    unsigned int GetCurrentFrameIdx() const { return current_frame_idx_; }
    Eigen::Matrix3d GetK() const { return K_; }
    void RemoveTrack(std::shared_ptr<ObjectTrack> track);
    std::vector<Detection::Ptr> GetCurrentFrameDetections()     { return current_frame_detections_; }
    std::vector<Detection::Ptr> GetCurrentFrameGoodDetections() { return current_frame_good_detections_; }
    double GetCurrentMeanDepth() const { return current_mean_depth_; }

    // Toggle mask at runtime (e.g. from viewer)
    std::atomic<bool> mbUseMask{true};

public:
    // ── Tracking state ───────────────────────────────────────────────────────
    enum eTrackingState {
        SYSTEM_NOT_READY             = -1,
        NO_IMAGES_YET                =  0,
        NOT_INITIALIZED              =  1,
        OK                           =  2,
        LOST                         =  3,
        FORCE_RELOC_POINTS           =  4,
        FORCE_RELOC_OBJECTS          =  5,
        FORCE_RELOC_OBJECTS_AND_POINTS = 6,
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    int mSensor;

    Frame   mCurrentFrame;
    cv::Mat mImGray;
    cv::Mat im_rgb_;

    // Monocular initialisation
    std::vector<int>        mvIniLastMatches;
    std::vector<int>        mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    // Trajectory storage
    list<cv::Mat>    mlRelativeFramePoses;
    list<KeyFrame*>  mlpReferences;
    list<double>     mlFrameTimes;
    list<bool>       mlbLost;

    bool mbOnlyTracking;
    bool mbVO;

    // ── Thread pointers ──────────────────────────────────────────────────────
    LocalMapping*       mpLocalMapper;
    LoopClosing*        mpLoopClosing;
    LocalObjectMapping* local_object_mapper_;

    // ── ORB extractors ───────────────────────────────────────────────────────
    ORBextractor* mpORBextractorLeft;
    ORBextractor* mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    // ── V1.2: low-texture extractor ──────────────────────────────────────────
    // Cùng pyramid/nFeatures với mpORBextractorLeft, nhưng FAST threshold thấp
    // hơn ~một nửa. Trần nhà / tường trắng với threshold gốc cho ra gần 0
    // keypoint; threshold thấp cho phép FAST bắt corner yếu hơn để PnP/BA
    // không bị đói điểm ngay lúc cần nhất (giữa góc cua, mono only).
    ORBextractor* mpORBextractorLowTexture = nullptr;
    static constexpr double kLowTextureExtractorThresh = 50.0;

    // ── BoW ─────────────────────────────────────────────────────────────────
    ORBVocabulary*    mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    Initializer* mpInitializer;

    // ── Local map ────────────────────────────────────────────────────────────
    KeyFrame*                mpReferenceKF;
    std::vector<KeyFrame*>   mvpLocalKeyFrames;
    std::vector<MapPoint*>   mvpLocalMapPoints;

    // ── System / drawers ─────────────────────────────────────────────────────
    System*      mpSystem;
    Viewer*      mpViewer;
    ARViewer*    mpARViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer*   mpMapDrawer;
    Map*         mpMap;

    // ── Calibration ──────────────────────────────────────────────────────────
    cv::Mat         mK;
    Eigen::Matrix3d K_;
    cv::Mat         mDistCoef;
    float           mbf;

    // ── KF insertion parameters ──────────────────────────────────────────────
    int mMinFrames;
    int mMaxFrames;

    float mThDepth;
    float mDepthMapFactor;

    int mnMatchesInliers;

    // ── Last frame / KF / reloc info ─────────────────────────────────────────
    KeyFrame*    mpLastKeyFrame;
    Frame        mLastFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;

    long unsigned int mId_NewMapOrigin = 0;
    cv::Mat mVelocity;
    bool    mbRGB;

    list<MapPoint*> mlpTemporalPoints;

    // ── Object tracking ──────────────────────────────────────────────────────
    std::list<ObjectTrack::Ptr>    objectTracks_;
    size_t                         current_frame_idx_  = 0;
    bool                           createdNewKeyFrame_ = false;
    std::vector<Detection::Ptr>    current_frame_detections_;
    std::vector<Detection::Ptr>    current_frame_good_detections_;
    double                         current_mean_depth_ = 0.0;

    // ── V1: consecutive lost counter ─────────────────────────────────────────
    // Used to decide when to truly reset (only after map is tiny AND lost long)
    int mnConsecutiveLostFrames = 0;

    // ── V1: texture score ────────────────────────────────────────────────────
    // Variance of Laplacian: <50 = white wall / low texture → stronger CLAHE
    double current_texture_score_ = 1000.0;

    // ── V1.2: zonal texture score ─────────────────────────────────────────────
    // Chia frame thành lưới NxM, tính variance-of-Laplacian riêng từng cell
    // bằng cách tái sử dụng map Laplacian đã tính cho current_texture_score_
    // (không thêm pass xử lý ảnh nào). Không gán nhãn "trần/nền" — chỉ biết
    // "% diện tích frame đang low-texture", bất kể nó là gì và camera đang
    // hướng thế nào (không cần IMU, không cần segmentation model).
    static constexpr int kTextureGridRows = 3;
    static constexpr int kTextureGridCols = 4;
    // % diện tích frame có texture < kLowTextureExtractorThresh, frame gần nhất
    float mLowTextureFraction = 0.0f;

    // ── V1: LK optical flow bridge ───────────────────────────────────────────
    // Fallback when ORB matching fails (occlusion, fast pan, motion blur).
    // Uses MapPoints only — no AI depth.
    cv::Mat                  mPrevGray;
    std::vector<cv::Point2f> mvPrevPts;
    std::vector<MapPoint*>   mvPrevMPs;
    int                      mnLKFallbackFrames = 0;

    // ── V1: large-rotation multi-seed recovery ───────────────────────────────
    // Threshold (degrees/frame) to activate multi-seed search.
    // V1.3: lowered from 5.0 → 3.0 so we catch moderate pans earlier
    // (at 15fps a 45°/s pan = 3°/frame — was slipping through at 5°).
    static constexpr double kLargeRotThreshDeg = 3.0;
    // Max candidate KFs to try
    static constexpr int    kMaxMultiSeedKF    = 25;

    // ── V1.3: fast-pan LK tuning ─────────────────────────────────────────────
    // Pyramid levels for calcOpticalFlowPyrLK.  5 levels instead of 4 doubles
    // the effective search range at the coarsest level (from 41*2^3=328px to
    // 41*2^4=656px half-side), capturing large inter-frame displacements during
    // a fast pan without inflating computational cost significantly.
    static constexpr int kLKPyrLevels = 5;
    // Window size for LK.  41 already generous; keep it.
    static constexpr int kLKWinSize   = 41;

    // ── V1.1: pre-lost snapshot for scale-guard / scale-recovery ─────────────
    // Updated whenever tracking is OK and decently confident; used to detect
    // scale jumps after LK bridge / MultiSeed / Relocalization save a track.
    float   mPreLostMedianDepth = -1.0f;
    cv::Mat mPreLostTcw;
    long unsigned int mPreLostFrameId = 0;
    int     mnPreLostInliers    = 0;

    // ── V1.2: inertia mode ────────────────────────────────────────────────────
    bool    mbInertiaMode   = false;
    int     mnInertiaFrames = 0;
    cv::Mat mLastGoodVelocity;
    static constexpr int kMaxInertiaFrames = 10;

    // ── V1.3: rotation-aware velocity prediction ──────────────────────────────
    // When the camera is in a fast pan, the constant-velocity model predicts a
    // translation that often puts the projected MapPoints outside the search
    // window in TrackWithMotionModel().  Storing the last N velocity matrices
    // lets us extrapolate a "rotational velocity" via SLERP and produce a better
    // initial pose for the next frame.
    //   mRotVelHistory[0] = most recent, ..., [kRotVelHistLen-1] = oldest.
    // Only updated when TrackLocalMap succeeds AND velocity sanity passes.
    static constexpr int kRotVelHistLen = 4;
    std::vector<cv::Mat> mRotVelHistory;  // ring buffer of last good velocities

    // Compute a rotation-extrapolated pose from mLastGoodVelocity.
    // Returns an empty Mat if not enough history.
    cv::Mat PredictPoseWithRotationalVelocity() const;

    // ── V1.1: configurable quality thresholds (read from settings file) ─────
    // Defaults match the original V1 hard-coded behaviour.
    int mRelocMinGoodInliers     = 30;
    int mKFMinInliersNoOcclusion = 15;
    int mKFMinInliersOcclusion   = 10;

    // ── V1.4: BoW-independent projection relocalization ──────────────────────
    // Minimum inliers (after PoseOptimization) required to accept a pose found
    // by RelocalizationByProjectionAllKFs(). Configurable via settings file.
    int mProjRelocMinInliers = 15;

    // ── V1.3: post-reset KF-insertion boost ───────────────────────────────────
    // Set to true by MapPreservingReset(). While active, NeedNewKeyFrame()
    // skips the mMaxFrames cooldown and uses a lower thRefRatio so the map
    // is densified quickly after the camera relocalized back into it.
    // Cleared after kSoftResetBoostFrames frames of successful tracking.
    bool mbPostResetBoost        = false;
    int  mnPostResetBoostFrames  = 0;
    static constexpr int kSoftResetBoostFrames = 60; // ~2s at 30fps

    // ── V1.5: stale-recovery escalation watchdog ─────────────────────────────
    // Problem: after MapPreservingReset() (Smart Reset), mState=LOST and the
    // normal recovery chain (QuickCovisibilityRecovery -> LK -> ProjReloc ->
    // BoW Relocalization) keeps retrying with the SAME narrow candidate set
    // (mpReferenceKF + its covisibility neighbours / current mvpLocalKeyFrames).
    // If the seed KF chosen by the soft reset is wrong (camera moved far from
    // it, or it sits in a disconnected covisibility cluster), every attempt
    // fails the same way forever -> frame counter freezes, map stops growing,
    // viewer looks "stuck" even though the map itself is intact and the
    // camera IS visibly inside it (as seen in the point cloud).
    //
    // Fix: count consecutive LOST frames specifically while mbPostResetBoost
    // is active (i.e. we know we're in a post-soft-reset recovery attempt).
    // Once it exceeds kStaleRecoveryEscalateFrames, force one full brute-force
    // pass: RelocalizationByProjectionAllKFsBruteForce() over EVERY keyframe
    // in the map (bypassing mpReferenceKF's covisibility graph entirely) —
    // this is the "rơi về match line tìm lại keypoint" behaviour requested.
    // If that still fails after kStaleRecoveryGiveUpFrames, give up softly and
    // fall back to hard Reset() automatically (matches the observed fact that
    // hard reset is reliable while soft reset can get stuck).
    int mnPostResetLostFrames = 0;
    static constexpr int kStaleRecoveryEscalateFrames = 15;  // ~0.5s at 30fps
    static constexpr int kStaleRecoveryGiveUpFrames   = 90;  // ~3s at 30fps

    // Brute-force projection relocalization over ALL keyframes in the map,
    // ignoring mpReferenceKF / covisibility entirely. Used only as an
    // escalation step when the normal recovery chain is stuck post-soft-reset.
    bool RelocalizationByProjectionAllKFsBruteForce();

    // ── V1.6: async LOST-recovery thread + throttle ──────────────────────────
    // Problem: when LOST, Track() ran the WHOLE recovery chain (LK -> ProjReloc
    // -> RelocFromObjects -> full BoW Relocalization, each candidate KF doing
    // its own PoseOptimization) synchronously, every single frame, inside the
    // main thread that also owns mpMap->mMutexMapUpdate. Each step is individually
    // fast, but the chain as a whole is heavy, so the main thread (and therefore
    // frame display / viewer) stalls for the full duration every frame while
    // LOST — this is the "đơ" (freeze) the user is seeing. It is NOT a frame-rate
    // throttle and NOT a sign the system is "trying harder" — it's just doing
    // the same expensive, mostly-failing work over and over with no backoff.
    //
    // Fix: move the expensive part of recovery (ProjReloc + RelocFromObjects +
    // full BoW Relocalization) to a dedicated background thread that:
    //   1. Operates on its OWN copy of the current Frame (via Frame's copy
    //      constructor) — never touches mCurrentFrame directly, so there is
    //      no race with the main thread's per-frame work.
    //   2. Is fed a lightweight snapshot (gray image + detections + frame id)
    //      ONLY when it is idle (mbRecoveryThreadBusy == false) — this is the
    //      throttle: if a recovery attempt is still running, new snapshots are
    //      simply dropped rather than queued, so the thread never falls behind
    //      and never piles up backlog.
    //   3. Publishes its result (success/fail + recovered pose + map point
    //      matches) through mRecoveryResultMutex; the MAIN thread is the only
    //      one that ever applies a recovered pose to mCurrentFrame/mState, so
    //      Track()'s existing single-threaded invariants on Frame/Map state
    //      stay intact.
    //
    // While the background attempt is in flight (or between attempts), the
    // main thread keeps running its existing CHEAP per-frame fallback only:
    // QuickCovisibilityRecovery() (already cheap) -> TrackWithOpticalFlow()
    // (cheap) -> TryInertiaRecovery() (cheap). None of these touch the
    // background thread's Frame copy. The viewer therefore always gets a
    // frame at full rate; only the eventual relocalization success is delayed
    // by however long the background attempt actually needs.
    std::thread mRecoveryThread;
    std::atomic<bool> mbRecoveryThreadBusy{false};
    std::atomic<bool> mbStopRecoveryThread{false};
    std::atomic<bool> mbRecoveryHasWork{false}; // snapshot ready, thread should wake
    std::condition_variable mRecoveryCondVar;

    std::mutex            mRecoverySnapshotMutex;
    cv::Mat                mRecoverySnapshotGray;
    std::vector<Detection::Ptr> mRecoverySnapshotGoodDetections;
    size_t                 mRecoverySnapshotFrameId = 0;

    std::mutex mRecoveryResultMutex;
    bool        mbRecoveryResultReady    = false; // main thread hasn't consumed it yet
    bool        mbRecoveryResultSuccess  = false;
    cv::Mat     mRecoveryResultTcw;
    std::vector<MapPoint*> mRecoveryResultMapPoints; // indexed like Frame::mvpMapPoints
    size_t mRecoveryResultFrameId = 0;         // which frame this pose belongs to
    KeyFrame*   mRecoveryResultRefKF = nullptr;

    // V1.6: minimum wall-clock gap between two background recovery attempts,
    // even if the thread finishes early — avoids spinning the background
    // thread itself into a hot loop of fail/retry on a genuinely hard scene.
    static constexpr int kRecoveryThreadMinIntervalMs = 150;

    // Launches the persistent background worker (called once from the ctor).
    void StartRecoveryThread();
    // Background worker loop — waits for work, runs heavy reloc chain on its
    // OWN Frame copy, publishes result, throttles, repeats until stopped.
    void RecoveryThreadLoop();
    // Called from Track() (main thread) once per LOST frame: pushes a fresh
    // snapshot to the background thread IF AND ONLY IF it is currently idle.
    // Never blocks; returns immediately either way.
    void OfferRecoverySnapshotIfIdle();
    // Called from Track() (main thread): non-blocking check for a published
    // result; if present and still relevant (frame id not stale), applies it
    // to mCurrentFrame/mState. Returns true if a result was applied this call.
    bool ConsumePublishedRecoveryResultIfAny();

    // ── Core tracking functions ───────────────────────────────────────────────
    void Track();

    void StereoInitialization();
    void MonocularInitialization();
    void CreateInitialMapMonocular();

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();

    // V1 additions
    bool TrackWithOpticalFlow();              // LK bridge (MapPoints only)
    bool TrackReferenceKeyFrameMultiSeed();  // multi-seed for large rotation
    double ComputeVelocityRotDeg() const;    // helper: rotation deg from velocity

    // ── V1.3: fast-pan projection search (no BoW) ────────────────────────────
    // When rotation is large, BoW descriptor matching fails because the viewing
    // angle changed too much. This searches by projecting all local KF map
    // points into the current frame using the rotation-extrapolated pose.
    // Much faster than full Relocalization() and doesn't require the area to
    // be in the BoW DB yet (useful for rooms the camera has only seen once).
    // Returns true if pose is recovered with ≥ 10 inliers.
    bool TrackByProjectionFastPan();

    // ── V1.2: inertia recovery ─────────────────────────────────────────────
    // Ngoại suy mLastGoodVelocity lên pose của mLastFrame. true nếu đang
    // thực sự quay (rotation đáng kể) và chưa vượt cap; false thì caller
    // rơi về xử lý LOST bình thường.
    bool TryInertiaRecovery();

    // ── V1.1 additions (ported from v0.1, fixes scale drift / slow recovery) ─
    // Quick, cheap recovery attempt before falling back to full Relocalization().
    // Tries refKF + covisible KFs with the candidate KF's own pose as seed
    // (NOT mLastFrame.mTcw — see fix V1.4: a fixed seed across all candidates
    // made this function nearly useless after a Soft Reset, since the only
    // thing that varied was the point set, not the assumed camera pose).
    bool QuickCovisibilityRecovery();

    // ── V1.8: rotation-seeded recovery for fast-pan LOST ─────────────────────
    // Called in the LOST branch AFTER QuickCovisibilityRecovery and LK both
    // fail. Uses PredictPoseWithRotationalVelocity() (mRotVelHistory) to
    // extrapolate the camera pose through the current rotation, then does a
    // wide SearchByProjection over local map points — identical logic to
    // TrackByProjectionFastPan() but callable from the LOST branch where
    // mState != OK. Only attempted when mRotVelHistory is non-empty (i.e. the
    // camera was rotating just before going LOST, which is exactly the fast-pan
    // scenario). Radius 50px to cope with the extra uncertainty from one missed
    // frame. Returns true if ≥10 inlier MapPoints are confirmed after PoseOpt.
    bool QuickRotationRecovery();

    // ── V1.4: BoW-independent projection relocalization ──────────────────────
    // Seeds pose = pKF->GetPose() for each candidate KF (covisibility-based,
    // same pattern as TrackReferenceKeyFrameMultiSeed), then projects that
    // KF's MapPoints into the current frame with a wide search radius.
    // Unlike Relocalization(), this does NOT depend on DetectRelocalizationCandidates()
    // (BoW shared-word score) — useful right after MapPreservingReset() when
    // the camera is still inside the old map but BoW scoring returns few/no
    // candidates (e.g. very different viewing angle / partial overlap).
    // Tried before the full BoW-based Relocalization() in the LOST recovery
    // chain since it is cheaper and does not require DB-wide candidates.
    bool RelocalizationByProjectionAllKFs();
    static constexpr int kMaxProjRelocKF = 20;

    // Estimate scale ratio (current / pre-lost) from KF median depth.
    // Returns -1.0 if not enough data or ratio out of sane range.
    double EstimateScaleFromDepth();

    // Estimate scale ratio from detection bbox vs projected object ellipsoid.
    // Returns -1.0 if no reliable object match.
    double EstimateScaleFromObjects(const cv::Mat& Tcw);

    // Apply scale correction to current pose; clears velocity/LK seeds since
    // they are no longer valid after a scale jump.
    void ApplyScaleCorrection(double scale_ratio);

    // Read configurable quality thresholds from the settings file
    // (defaults below match the original V1 hard-coded behaviour).
    void LoadQualityThresholds(const string& strSettingPath);

    bool Relocalization();
    bool RelocalizationFromObjects(bool use_points);

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

private:
    // Build person-exclusion mask from current_frame_good_detections_.
    // kernel_size: erode kernel (5 for V1 — smaller than V0's 13 to protect ceiling).
    cv::Mat BuildPersonMask(const cv::Size& sz, int kernel_size = 5) const;
    // V1.9: true khi pose frame hiện tại đến từ QuickCovisibilityRecovery() / LK / QuickRotation
    // TrackLocalMap() dùng để không áp đặt ngưỡng 50 inliers lên local recovery.
    bool mbLastRecoveryWasQuickCovis = false;
    bool mbLastRecoveryWasLK = false;
    bool mbLastRecoveryWasQuickRot = false;
};

} // namespace ORB_SLAM2

#endif // TRACKING_H