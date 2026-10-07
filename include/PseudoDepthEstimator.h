#pragma once

/**
 * PseudoDepthEstimator
 * ====================
 * Chạy DepthAnything v2 Small (ONNX) để sinh relative depth map từ RGB frame.
 * Thread-safe: inference chạy trong background thread, kết quả lấy qua GetLatestDepth().
 *
 * Pipeline:
 *   RGB frame → resize 518×518 → normalize → ONNX inference → resize về kích thước gốc
 *   → AlignScale() với MapPoints hiện có → depth map CV_32F (đơn vị tương đối, đã scale)
 *
 * Cài đặt ONNX Runtime C++:
 *   wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-gpu-1.17.1.tgz
 *   tar xzf onnxruntime-linux-x64-gpu-1.17.1.tgz
 *   # Trong CMakeLists.txt:
 *   # include_directories(onnxruntime-linux-x64-gpu-1.17.1/include)
 *   # link_directories(onnxruntime-linux-x64-gpu-1.17.1/lib)
 *   # target_link_libraries(... onnxruntime)
 *
 * Model download:
 *   https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx
 */

#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <vector>

// Forward declare để tránh include onnxruntime header ở đây
// (tránh làm nặng compile của toàn project)
namespace Ort { class Session; class Env; class SessionOptions; }

namespace ORB_SLAM2
{

class PseudoDepthEstimator
{
public:
    // ── Cấu hình ────────────────────────────────────────────────────────────
    struct Config {
        std::string model_path;         // đường dẫn file .onnx
        int input_size = 518;           // DepthAnything v2 dùng 518×518
        bool use_gpu = true;            // dùng CUDA EP nếu có
        int gpu_device_id = 0;
        float depth_scale_clamp = 10.0f; // clamp depth (sau align) để tránh outlier
        bool enabled = true;            // tắt/bật runtime
    };

    explicit PseudoDepthEstimator(const Config& cfg);
    ~PseudoDepthEstimator();

    // Không copy
    PseudoDepthEstimator(const PseudoDepthEstimator&) = delete;
    PseudoDepthEstimator& operator=(const PseudoDepthEstimator&) = delete;

    // ── API chính ────────────────────────────────────────────────────────────

    /**
     * PushFrame: đưa frame RGB mới vào queue để inference.
     * Non-blocking. Nếu queue đã có frame cũ chưa process → thay thế (drop frame cũ).
     * Gọi từ main tracking thread, sau khi có frame mới.
     */
    void PushFrame(const cv::Mat& rgb_frame, uint64_t frame_id);

    /**
     * GetLatestDepth: lấy depth map mới nhất đã inference xong.
     * Trả về false nếu chưa có kết quả (frame đầu tiên, hoặc network chậm).
     * depth_out: CV_32F, cùng size với frame gốc, unit = relative (chưa align).
     * frame_id_out: frame_id tương ứng với depth này.
     */
    bool GetLatestDepth(cv::Mat& depth_out, uint64_t& frame_id_out);

    /**
     * AlignScaleAndOffset: fit depth map relative về absolute scale.
     *
     * Lý thuyết:
     *   depth_absolute = scale * depth_relative + offset
     *   Trong đó scale, offset được tìm bằng least-squares từ các MapPoint
     *   đang track được trong frame (chúng có depth absolute thực từ 3D pose).
     *
     * Input:
     *   depth_relative: output của network (CV_32F)
     *   keypoints_2d:   tọa độ 2D của các MapPoint trong frame (undistorted)
     *   depths_3d:      depth (z trong camera frame) tương ứng từ pose
     *   min_pairs:      số cặp tối thiểu để tin kết quả alignment
     *
     * Output:
     *   depth_aligned: CV_32F, unit = meter (approximate)
     *   Trả về false nếu không đủ pairs hoặc alignment thất bại.
     */
    static bool AlignScaleAndOffset(
        const cv::Mat& depth_relative,
        const std::vector<cv::Point2f>& keypoints_2d,
        const std::vector<float>& depths_3d,
        cv::Mat& depth_aligned,
        int min_pairs = 8,
        float max_depth_3d = 20.0f   // loại điểm quá xa (unreliable triangulation)
    );

    bool IsEnabled() const { return cfg_.enabled && session_ready_; }
    void SetEnabled(bool v) { cfg_.enabled = v; }

private:
    void InferenceThread();

    // Chạy inference 1 frame, trả về depth map (CV_32F, normalized 0-1)
    cv::Mat RunInference(const cv::Mat& rgb);

    // Preprocess: resize, normalize theo ImageNet mean/std
    std::vector<float> Preprocess(const cv::Mat& rgb, int& orig_w, int& orig_h);

    Config cfg_;
    bool session_ready_ = false;

    // ONNX Runtime objects — dùng void* để tránh include header ở đây
    void* ort_env_   = nullptr;  // Ort::Env*
    void* ort_sess_  = nullptr;  // Ort::Session*
    void* ort_opts_  = nullptr;  // Ort::SessionOptions*

    // Background thread
    std::thread inference_thread_;
    std::atomic<bool> stop_flag_{false};

    // Input queue (capacity=1: luôn lấy frame mới nhất)
    struct InputFrame {
        cv::Mat rgb;
        uint64_t frame_id;
    };
    std::mutex input_mutex_;
    std::condition_variable input_cv_;
    bool has_input_ = false;
    InputFrame pending_input_;

    // Output (kết quả mới nhất)
    std::mutex output_mutex_;
    cv::Mat latest_depth_;
    uint64_t latest_frame_id_ = 0;
    bool has_output_ = false;
};

} // namespace ORB_SLAM2
