import cv2
import numpy as np
import os

# --- CẤU HÌNH ---
CHECKERBOARD = (9, 6) 
SQUARE_SIZE = 0.035 
SAVE_DIR = "/home/huyen1905/OA-SLAM/d455_calib_images"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []
cap = cv2.VideoCapture(6, cv2.CAP_V4L2)

print(f"--- HƯỚNG DẪN ---")
print(f"Ảnh sẽ được lưu tại: {os.path.abspath(SAVE_DIR)}")
print("Nhấn 's' để chụp (cần >10 ảnh), nhấn 'q' để kết thúc và xem kết quả.")

count = 0
gray = None # Định nghĩa biến gray trước vòng lặp

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    display_frame = frame.copy()
    if ret:
        cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners, ret)
        
    cv2.imshow('Calibration', display_frame)
    k = cv2.waitKey(1)
    
    if k == ord('s') and ret:
        filename = os.path.join(SAVE_DIR, f"calib_{count}.jpg")
        cv2.imwrite(filename, frame)
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        print(f"Đã lưu ảnh: {filename}")
        count += 1
        
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Tính toán và IN KẾT QUẢ RA TERMINAL
if len(imgpoints) > 10:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    print("\n" + "="*40)
    print("      KẾT QUẢ HIỆU CHUẨN CAMERA")
    print("="*40)
    print(f"Camera Matrix (K) - Dùng cho file .yaml:")
    print(f"fx: {mtx[0,0]:.4f}, fy: {mtx[1,1]:.4f}")
    print(f"cx: {mtx[0,2]:.4f}, cy: {mtx[1,2]:.4f}")
    print("\nDistortion Coefficients (D):")
    print(f"k1: {dist[0][0]:.6f}, k2: {dist[0][1]:.6f}")
    print(f"p1: {dist[0][2]:.6f}, p2: {dist[0][3]:.6f}")
    print(f"k3: {dist[0][4] if len(dist[0]) > 4 else 0.0:.6f}")
    print("="*40 + "\n")
else:
    print(f"\nLỗi: Chỉ chụp được {len(imgpoints)} ảnh. Cần ít nhất 11 ảnh để tính toán!")
