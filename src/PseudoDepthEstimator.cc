/**
 * PseudoDepthEstimator.cc
 * =======================
 * Implementation của PseudoDepthEstimator.
 *
 * Compile note:
 *   Cần link: -lonnxruntime
 *   Cần include: onnxruntime/core/providers/cuda/cuda_provider_factory.h (nếu dùng GPU)
 *
 * DepthAnything v2 input/output spec:
 *   Input:  "image"  shape [1, 3, 518, 518]  float32  (ImageNet normalized)
 *   Output: "depth"  shape [1, 1, 518, 518]  float32  (relative, unnormalized)
 *   Note: output là inverse depth hoặc direct depth tùy version export.
 *         Model từ hf.co/onnx-community/depth-anything-v2-small output DIRECT depth (lớn = xa).
 */

#include "PseudoDepthEstimator.h"

// ── Guard: nếu ONNX Runtime chưa được cài, toàn bộ file này là stub ─────────
// Để build không bị lỗi linker khi chưa có thư viện.
// Khi đã cài xong: xóa dòng #define NO_ONNXRUNTIME hoặc để CMake tự set.
#ifndef NO_ONNXRUNTIME

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/cuda/cuda_provider_factory.h>

#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cassert>

namespace ORB_SLAM2
{

// ── ImageNet normalization constants ────────────────────────────────────────
static const float kMean[3] = {0.485f, 0.456f, 0.406f};
static const float kStd[3]  = {0.229f, 0.224f, 0.225f};

// ============================================================================
// Constructor / Destructor
// ============================================================================

PseudoDepthEstimator::PseudoDepthEstimator(const Config& cfg)
    : cfg_(cfg)
{
    if (!cfg_.enabled) {
        std::cout << "[PseudoDepth] Disabled by config.\n";
        return;
    }

    try {
        // Tạo Ort::Env
        auto* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "PseudoDepth");
        ort_env_ = env;

        // Session options
        auto* opts = new Ort::SessionOptions();
        opts->SetIntraOpNumThreads(2);
        opts->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        ort_opts_ = opts;

        // GPU EP
        if (cfg_.use_gpu) {
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = cfg_.gpu_device_id;
            cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cuda_opts.gpu_mem_limit = (size_t)(2LL * 1024 * 1024 * 1024); // 2GB
            opts->AppendExecutionProvider_CUDA(cuda_opts);
            std::cout << "[PseudoDepth] Using CUDA GPU device " << cfg_.gpu_device_id << "\n";
        }

        // Load model
        auto* sess = new Ort::Session(*env, cfg_.model_path.c_str(), *opts);
        ort_sess_ = sess;

        session_ready_ = true;
        std::cout << "[PseudoDepth] Model loaded: " << cfg_.model_path << "\n";

        // Print input/output info
        Ort::AllocatorWithDefaultOptions allocator;
        size_t n_in  = sess->GetInputCount();
        size_t n_out = sess->GetOutputCount();
        for (size_t i = 0; i < n_in; ++i) {
            auto name = sess->GetInputNameAllocated(i, allocator);
            auto info = sess->GetInputTypeInfo(i);
            auto shape = info.GetTensorTypeAndShapeInfo().GetShape();
            std::cout << "[PseudoDepth]  Input[" << i << "] \"" << name.get() << "\" shape=[";
            for (auto d : shape) std::cout << d << ",";
            std::cout << "]\n";
        }
        for (size_t i = 0; i < n_out; ++i) {
            auto name = sess->GetOutputNameAllocated(i, allocator);
            std::cout << "[PseudoDepth]  Output[" << i << "] \"" << name.get() << "\"\n";
        }

        // Khởi động background inference thread
        inference_thread_ = std::thread(&PseudoDepthEstimator::InferenceThread, this);

    } catch (const Ort::Exception& e) {
        std::cerr << "[PseudoDepth] ONNX Runtime error: " << e.what() << "\n";
        session_ready_ = false;
    }
}

PseudoDepthEstimator::~PseudoDepthEstimator()
{
    stop_flag_ = true;
    input_cv_.notify_all();
    if (inference_thread_.joinable())
        inference_thread_.join();

    delete static_cast<Ort::Session*>(ort_sess_);
    delete static_cast<Ort::SessionOptions*>(ort_opts_);
    delete static_cast<Ort::Env*>(ort_env_);
}

// ============================================================================
// PushFrame — gọi từ main thread
// ============================================================================

void PseudoDepthEstimator::PushFrame(const cv::Mat& rgb_frame, uint64_t frame_id)
{
    if (!IsEnabled()) return;

    {
        std::lock_guard<std::mutex> lock(input_mutex_);
        // Luôn drop frame cũ, chỉ giữ frame mới nhất
        // Lý do: tracking chạy ở 15fps, inference ~30-50ms → có thể lag 1 frame
        // Không sao nếu depth của frame N-1 được dùng cho frame N (scene thay đổi chậm)
        pending_input_.rgb = rgb_frame.clone();
        pending_input_.frame_id = frame_id;
        has_input_ = true;
    }
    input_cv_.notify_one();
}

// ============================================================================
// GetLatestDepth — gọi từ main thread
// ============================================================================

bool PseudoDepthEstimator::GetLatestDepth(cv::Mat& depth_out, uint64_t& frame_id_out)
{
    std::lock_guard<std::mutex> lock(output_mutex_);
    if (!has_output_) return false;
    depth_out = latest_depth_.clone();
    frame_id_out = latest_frame_id_;
    return true;
}

// ============================================================================
// InferenceThread — background thread
// ============================================================================

void PseudoDepthEstimator::InferenceThread()
{
    while (!stop_flag_)
    {
        // Đợi input
        InputFrame input;
        {
            std::unique_lock<std::mutex> lock(input_mutex_);
            input_cv_.wait(lock, [this]{ return has_input_ || stop_flag_; });
            if (stop_flag_) break;
            input = std::move(pending_input_);
            has_input_ = false;
        }

        // Chạy inference
        cv::Mat depth = RunInference(input.rgb);
        if (depth.empty()) continue;

        // Lưu kết quả
        {
            std::lock_guard<std::mutex> lock(output_mutex_);
            latest_depth_ = depth;
            latest_frame_id_ = input.frame_id;
            has_output_ = true;
        }
    }
}

// ============================================================================
// RunInference — chạy 1 frame qua network
// ============================================================================

cv::Mat PseudoDepthEstimator::RunInference(const cv::Mat& rgb)
{
    auto* sess = static_cast<Ort::Session*>(ort_sess_);
    if (!sess) return {};

    int orig_w = rgb.cols;
    int orig_h = rgb.rows;

    // Preprocess
    std::vector<float> input_data = Preprocess(rgb, orig_w, orig_h);

    // Tạo tensor
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 4> input_shape = {1, 3, cfg_.input_size, cfg_.input_size};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size()
    );

    // Run
    const char* input_names[]  = {"image"};
    const char* output_names[] = {"depth"};

    auto output_tensors = sess->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1
    );

    // Lấy output
    float* depth_data = output_tensors[0].GetTensorMutableData<float>();
    auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    // shape = [1, 1, H, W] hoặc [1, H, W]
    int out_h = (int)shape[shape.size()-2];
    int out_w = (int)shape[shape.size()-1];

    // Wrap vào cv::Mat
    cv::Mat depth_raw(out_h, out_w, CV_32F, depth_data);
    cv::Mat depth_float = depth_raw.clone();

    // Resize về kích thước gốc
    cv::Mat depth_resized;
    cv::resize(depth_float, depth_resized, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);

    // DepthAnything v2 output là relative depth (lớn = xa).
    // Normalize về [0, 1] để dễ xử lý
    double mn, mx;
    cv::minMaxLoc(depth_resized, &mn, &mx);
    if (mx - mn < 1e-6) return {};
    cv::Mat depth_norm;
    depth_resized.convertTo(depth_norm, CV_32F, 1.0/(mx-mn), -mn/(mx-mn));

    return depth_norm;  // [0,1], lớn = xa hơn
}

// ============================================================================
// Preprocess
// ============================================================================

std::vector<float> PseudoDepthEstimator::Preprocess(const cv::Mat& rgb, int& orig_w, int& orig_h)
{
    orig_w = rgb.cols;
    orig_h = rgb.rows;

    // Resize về input_size × input_size
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(cfg_.input_size, cfg_.input_size), 0, 0, cv::INTER_LINEAR);

    // Convert BGR→RGB nếu cần (OpenCV default là BGR)
    cv::Mat rgb_mat;
    if (rgb.channels() == 3)
        cv::cvtColor(resized, rgb_mat, cv::COLOR_BGR2RGB);
    else if (rgb.channels() == 1)
        cv::cvtColor(resized, rgb_mat, cv::COLOR_GRAY2RGB);
    else
        rgb_mat = resized;

    rgb_mat.convertTo(rgb_mat, CV_32F, 1.0f / 255.0f);

    // CHW layout + ImageNet normalization
    int H = cfg_.input_size;
    int W = cfg_.input_size;
    std::vector<float> blob(3 * H * W);

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float val = rgb_mat.at<cv::Vec3f>(h, w)[c];
                blob[c * H * W + h * W + w] = (val - kMean[c]) / kStd[c];
            }
        }
    }
    return blob;
}

// ============================================================================
// AlignScaleAndOffset — STATIC, gọi từ Tracking
// ============================================================================
// Bài toán: tìm (scale, offset) sao cho:
//   depth_aligned[i] = scale * depth_relative[i] + offset ≈ depth_3d[i]
//
// Giải bằng least-squares:
//   [d_rel_1  1] [scale ]   [d_3d_1]
//   [d_rel_2  1] [offset] = [d_3d_2]
//   [   ...     ]           [  ...  ]
//
// Dùng RANSAC nhẹ để loại outlier (điểm triangulate sai, depth network sai cục bộ).
// ============================================================================

bool PseudoDepthEstimator::AlignScaleAndOffset(
    const cv::Mat& depth_relative,
    const std::vector<cv::Point2f>& keypoints_2d,
    const std::vector<float>& depths_3d,
    cv::Mat& depth_aligned,
    int min_pairs,
    float max_depth_3d)
{
    assert(keypoints_2d.size() == depths_3d.size());
    if ((int)keypoints_2d.size() < min_pairs) return false;

    // Thu thập các cặp (d_rel, d_3d) hợp lệ
    struct Pair { float d_rel, d_3d; };
    std::vector<Pair> pairs;
    pairs.reserve(keypoints_2d.size());

    for (size_t i = 0; i < keypoints_2d.size(); ++i)
    {
        float d3 = depths_3d[i];
        if (d3 <= 0.05f || d3 > max_depth_3d) continue;  // quá gần/xa = unreliable

        int px = cvRound(keypoints_2d[i].x);
        int py = cvRound(keypoints_2d[i].y);
        if (px < 0 || py < 0 || px >= depth_relative.cols || py >= depth_relative.rows) continue;

        float d_rel = depth_relative.at<float>(py, px);
        if (d_rel < 1e-4f) continue;  // vùng tối / invalid

        pairs.push_back({d_rel, d3});
    }

    if ((int)pairs.size() < min_pairs) {
        std::cout << "[PseudoDepth] AlignScale: too few valid pairs " << pairs.size() << "\n";
        return false;
    }

    // ── RANSAC để loại outlier ───────────────────────────────────────────────
    // Mỗi iteration: chọn 2 điểm → giải scale/offset → đếm inlier
    const int N = (int)pairs.size();
    const int MAX_ITER = 50;
    const float INLIER_THRESH = 0.15f;  // 15% của d_3d là ngưỡng inlier

    float best_scale = 1.0f, best_offset = 0.0f;
    int best_inliers = 0;

    std::srand(42);
    for (int iter = 0; iter < MAX_ITER && best_inliers < N * 0.9f; ++iter)
    {
        // Chọn 2 điểm ngẫu nhiên
        int i1 = std::rand() % N;
        int i2 = std::rand() % N;
        if (i1 == i2) continue;

        float dr1 = pairs[i1].d_rel, d1 = pairs[i1].d_3d;
        float dr2 = pairs[i2].d_rel, d2 = pairs[i2].d_3d;

        float denom = dr1 - dr2;
        if (std::abs(denom) < 1e-5f) continue;

        float scale  = (d1 - d2) / denom;
        float offset = d1 - scale * dr1;

        if (scale <= 0) continue;  // scale âm = vô nghĩa

        // Đếm inlier
        int inliers = 0;
        for (const auto& p : pairs) {
            float pred = scale * p.d_rel + offset;
            float err  = std::abs(pred - p.d_3d) / (p.d_3d + 1e-3f);
            if (err < INLIER_THRESH) ++inliers;
        }

        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_scale   = scale;
            best_offset  = offset;
        }
    }

    // Refine: least-squares trên toàn bộ inlier set
    {
        double sum_dr2 = 0, sum_dr = 0, sum_dr_d3 = 0, sum_d3 = 0;
        int n = 0;
        for (const auto& p : pairs) {
            float pred = best_scale * p.d_rel + best_offset;
            float err  = std::abs(pred - p.d_3d) / (p.d_3d + 1e-3f);
            if (err < INLIER_THRESH * 2) {  // 2x threshold để refine mượt hơn
                sum_dr2    += p.d_rel * p.d_rel;
                sum_dr     += p.d_rel;
                sum_dr_d3  += p.d_rel * p.d_3d;
                sum_d3     += p.d_3d;
                ++n;
            }
        }
        if (n >= min_pairs) {
            // [sum_dr2  sum_dr ] [scale ]   [sum_dr_d3]
            // [sum_dr   n      ] [offset] = [sum_d3   ]
            double det = sum_dr2 * n - sum_dr * sum_dr;
            if (std::abs(det) > 1e-8) {
                best_scale  = (float)((n * sum_dr_d3 - sum_dr * sum_d3) / det);
                best_offset = (float)((sum_dr2 * sum_d3 - sum_dr * sum_dr_d3) / det);
            }
        }
    }

    std::cout << "[PseudoDepth] AlignScale: scale=" << best_scale
              << " offset=" << best_offset
              << " inliers=" << best_inliers << "/" << N << "\n";

    if (best_scale <= 0) return false;

    // Áp dụng alignment
    depth_aligned = depth_relative * best_scale + best_offset;

    // Clamp: depth âm hoặc quá xa là vô nghĩa
    cv::threshold(depth_aligned, depth_aligned, 0.01f, 0, cv::THRESH_TOZERO);
    // Không clamp max vì scene có thể có điểm rất xa (hành lang dài)

    return true;
}

} // namespace ORB_SLAM2

#else // NO_ONNXRUNTIME — stub implementations khi chưa cài ONNX Runtime

#include <iostream>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{

PseudoDepthEstimator::PseudoDepthEstimator(const Config& cfg) : cfg_(cfg)
{
    cfg_.enabled = false;
    session_ready_ = false;
    std::cout << "[PseudoDepth] Built WITHOUT ONNX Runtime (stub mode). "
              << "Install onnxruntime and rebuild to enable.\n";
}

PseudoDepthEstimator::~PseudoDepthEstimator()
{
    stop_flag_ = true;
    input_cv_.notify_all();
    if (inference_thread_.joinable())
        inference_thread_.join();
}

void PseudoDepthEstimator::PushFrame(const cv::Mat&, uint64_t) {}

bool PseudoDepthEstimator::GetLatestDepth(cv::Mat&, uint64_t&) { return false; }

bool PseudoDepthEstimator::AlignScaleAndOffset(
    const cv::Mat&,
    const std::vector<cv::Point2f>&,
    const std::vector<float>&,
    cv::Mat&,
    int, float)
{
    return false;
}

void PseudoDepthEstimator::InferenceThread() {}
cv::Mat PseudoDepthEstimator::RunInference(const cv::Mat&) { return {}; }
std::vector<float> PseudoDepthEstimator::Preprocess(const cv::Mat&, int&, int&) { return {}; }

} // namespace ORB_SLAM2

#endif // NO_ONNXRUNTIME