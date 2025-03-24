import cv2
import numpy as np
from baseballcv.functions import LoadTools
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import colorsys
import os
from datetime import datetime
from scipy.optimize import curve_fit

def create_color_gradient(n, start_color=(255, 255, 255), end_color=(255, 255, 255)):
    """創建白色軌跡點"""
    return [(255, 255, 255)] * n  # 直接返回純白色列表

def select_video():
    """打開檔案選擇對話框"""
    root = tk.Tk()
    root.withdraw()  # 隱藏主視窗
    file_path = filedialog.askopenfilename(
        title="選擇視訊檔案",
        filetypes=[
            ("視訊檔案", "*.mp4 *.avi *.mov"),
            ("所有檔案", "*.*")
        ]
    )
    return file_path

def get_output_path():
    """生成輸出檔案路徑"""
    # 獲取當前時間作為檔案名稱的一部分
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"baseball_tracking_{current_time}.mp4"
    
    # 設定儲存路徑為yolo_baseball_track資料夾
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo_baseball_track_video")
    
    # 確保yolo_baseball_track資料夾存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return os.path.join(output_dir, default_filename)

def get_reference_points(frame, model):
    """獲取投手板和本壘板的位置"""
    results = model(frame, conf=0.5)
    rubber_pos = None
    home_pos = None
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 獲取類別
            cls = int(box.cls[0])
            # 獲取邊界框座標
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 計算中心點
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 根據類別分配位置
            if cls == 0:  # 投手板
                rubber_pos = (center_x, center_y)
            elif cls == 1:  # 本壘板
                home_pos = (center_x, center_y)
            # 忽略其他類別（如捕手手套）
    
    return rubber_pos, home_pos

def detect_pitcher_release(frame, model):
    """檢測投手投球動作"""
    results = model(frame, conf=0.5)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 獲取類別
            cls = int(box.cls[0])
            # 獲取置信度
            conf = float(box.conf[0])
            # 獲取邊界框座標
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 確保座標在有效範圍內
            x1 = max(0, min(x1, frame.shape[1]))
            y1 = max(0, min(y1, frame.shape[0]))
            x2 = max(0, min(x2, frame.shape[1]))
            y2 = max(0, min(y2, frame.shape[0]))
            
            # 如果是投手且置信度大於閾值
            if cls == 0 and conf > 0.7:  # 假設類別0是投手
                # 確保區域有效
                if x2 > x1 and y2 > y1:
                    # 計算投手區域
                    pitcher_area = frame[y1:y2, x1:x2]
                    if pitcher_area.size > 0:
                        # 計算投手區域的運動
                        gray = cv2.cvtColor(pitcher_area, cv2.COLOR_BGR2GRAY)
                        if hasattr(detect_pitcher_release, 'prev_gray'):
                            # 確保prev_gray的尺寸與當前gray相同
                            if gray.shape == detect_pitcher_release.prev_gray.shape:
                                # 計算幀差
                                diff = cv2.absdiff(gray, detect_pitcher_release.prev_gray)
                                # 計算運動量
                                motion = np.mean(diff)
                                # 如果運動量超過閾值，認為投手投球
                                if motion > 30:  # 可以調整這個閾值
                                    return True
                        detect_pitcher_release.prev_gray = gray
    return False

def is_ball_in_pitcher_area(ball_pos, pitcher_box):
    """檢查球是否在投手區域內"""
    if pitcher_box is None:
        return False
    x1, y1, x2, y2 = pitcher_box
    ball_x, ball_y = ball_pos
    return x1 <= ball_x <= x2 and y1 <= ball_y <= y2

def fit_parabola(x, y):
    """使用拋物線擬合軌跡點"""
    def parabola(x, a, b, c):
        return a * x**2 + b * x + c
    
    try:
        # 確保資料點足夠
        if len(x) < 3:
            return None, None, None
        
        # 擬合拋物線
        popt, _ = curve_fit(parabola, x, y)
        return popt
    except:
        return None, None, None

def filter_trajectory_points(points, threshold=30):
    """篩選軌跡點，排除異常點"""
    if len(points) < 3:
        return points
    
    # 提取x和y座標
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    
    # 擬合拋物線
    popt = fit_parabola(x, y)
    if popt is None:
        return points
    
    a, b, c = popt
    
    # 計算每個點到擬合拋物線的距離
    distances = []
    for i in range(len(points)):
        x_i = points[i][0]
        y_i = points[i][1]
        y_fit = a * x_i**2 + b * x_i + c
        distance = abs(y_i - y_fit)
        distances.append(distance)
    
    # 計算距離的均值和標準差
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    # 篩選點
    filtered_points = []
    for i in range(len(points)):
        if distances[i] <= mean_dist + threshold * std_dist:
            filtered_points.append(points[i])
    
    return filtered_points

def track_baseball(video_path, output_path):
    # 載入模型
    load_tools = LoadTools()
    ball_model = load_tools.load_model("ball_trackingv4")
    pitcher_model = load_tools.load_model("phc_detector")
    ball_model = YOLO(ball_model)
    pitcher_model = YOLO(pitcher_model)
    
    # 開啟視訊檔案
    cap = cv2.VideoCapture(video_path)
    
    # 獲取視訊屬性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 檢測頻率控制
    detection_interval = 2  # 每隔幾幀進行一次完整檢測
    frame_skip = 0
    
    # 計算縮放比例（保持寬高比）
    scale_factor = 0.4  # 縮放到原始大小的40%
    scaled_width = int(width * scale_factor)
    scaled_height = int(height * scale_factor)
    
    # 建立視訊寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 用於儲存軌跡的點
    trajectory_points = []
    max_points = 60
    colors = create_color_gradient(max_points)  # 使用純白色
    
    # 建立軌跡圖層
    trajectory_layer = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 幀計數器和追蹤狀態
    frame_count = 0
    tracking_started = False
    release_detected = False
    ball_detected = False
    
    # 儲存投手區域
    pitcher_box = None
    
    # 用於儲存上一幀的球位置
    last_ball_pos = None
    last_ball_size = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame_skip += 1
        
        # 縮放影像以加速處理
        scaled_frame = cv2.resize(frame, (scaled_width, scaled_height))
        
        # 每隔一定幀數才進行完整檢測
        if frame_skip >= detection_interval:
            frame_skip = 0
            
            # 檢測投手投球動作
            if not release_detected:
                if detect_pitcher_release(scaled_frame, pitcher_model):
                    release_detected = True
                    print("檢測到投球動作")
            
            # 只獲取投手區域，不顯示檢測框
            results = pitcher_model(scaled_frame, conf=0.5)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # 獲取類別和置信度
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # 只處理投手（類別1）
                    if cls == 1:  # 投手
                        # 獲取邊界框座標
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # 將座標轉換回原始大小
                        x1 = int(x1 / scale_factor)
                        y1 = int(y1 / scale_factor)
                        x2 = int(x2 / scale_factor)
                        y2 = int(y2 / scale_factor)
                        
                        # 確保座標在有效範圍內
                        x1 = max(0, min(x1, frame.shape[1]))
                        y1 = max(0, min(y1, frame.shape[0]))
                        x2 = max(0, min(x2, frame.shape[1]))
                        y2 = max(0, min(y2, frame.shape[0]))
                        
                        # 儲存投手區域
                        pitcher_box = (x1, y1, x2, y2)
        
        # 檢測球
        ball_detected = False
        ball_size = None  # 儲存球的大小
        current_ball_pos = None
        
        # 使用縮放後的影像進行球體檢測
        results = ball_model(scaled_frame, conf=0.5)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 將座標轉換回原始大小
                center_x = int((x1 + x2) / 2 / scale_factor)
                center_y = int((y1 + y2) / 2 / scale_factor)
                
                # 計算球的大小
                ball_size = int(min((x2 - x1) / 2, (y2 - y1) / 2) / scale_factor)
                current_ball_pos = (center_x, center_y)
                
                ball_detected = True
                
                # 檢查球是否在投手區域內
                if not is_ball_in_pitcher_area((center_x, center_y), pitcher_box):
                    if not tracking_started:
                        tracking_started = True
                        trajectory_points = []  # 清空之前的軌跡點
                        print("球離開投手區域，開始追蹤")
                    
                    # 只有當球的位置發生顯著變化時才添加新點
                    if last_ball_pos is None or (
                        abs(center_x - last_ball_pos[0]) > 5 or 
                        abs(center_y - last_ball_pos[1]) > 5
                    ):
                        trajectory_points.append((center_x, center_y))
        
        # 如果當前幀沒有檢測到球，使用上一幀的位置和大小
        if not ball_detected and last_ball_pos is not None:
            current_ball_pos = last_ball_pos
            ball_size = last_ball_size
        
        # 更新上一幀的資訊
        last_ball_pos = current_ball_pos
        last_ball_size = ball_size
        
        # 保持軌跡點數量在合理範圍內
        if len(trajectory_points) > max_points:
            trajectory_points.pop(0)
        
        # 篩選軌跡點
        if len(trajectory_points) >= 3:
            trajectory_points = filter_trajectory_points(trajectory_points)
        
        # 更新軌跡圖層
        trajectory_layer = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 繪製連續的軌跡線條
        if len(trajectory_points) >= 2:
            # 將所有點轉換為numpy數組以便繪製
            points = np.array(trajectory_points, np.int32)
            
            # 使用高斯平滑處理點
            if len(points) >= 3:
                # 對x和y座標分別進行平滑
                x_coords = points[:, 0]
                y_coords = points[:, 1]
                
                # 使用高斯平滑
                kernel_size = 5  # 增加平滑度
                x_smooth = cv2.GaussianBlur(x_coords.astype(float), (kernel_size, 1), 0)
                y_smooth = cv2.GaussianBlur(y_coords.astype(float), (kernel_size, 1), 0)
                
                # 重新組合平滑後的點
                points = np.column_stack((x_smooth, y_smooth)).astype(np.int32)
            
            # 繪製發光效果
            for i in range(len(points)-1):
                pt1 = points[i]
                pt2 = points[i+1]
                
                # 計算漸變顏色（從紅色到橙色到黃色）
                ratio = i / len(points)
                if ratio < 0.5:
                    # 從紅色到橙色
                    r = 255
                    g = int(255 * (ratio * 2))
                    b = 0
                else:
                    # 從橙色到黃色
                    r = 255
                    g = 255
                    b = int(255 * ((ratio - 0.5) * 2))
                
                color = (b, g, r)
                
                # 繪製多層發光效果
                # 最外層發光（更寬且更亮）
                cv2.line(trajectory_layer, pt1, pt2, color, 15)
                
                # 中間層發光（加強黃色）
                cv2.line(trajectory_layer, pt1, pt2, (0, 255, 255), 10)
                
                # 內層核心（更細的白色）
                cv2.line(trajectory_layer, pt1, pt2, (255, 255, 255), 3)
                
                # 最後一個點的特殊效果
                if i == len(points)-2:
                    # 增強火花效果
                    for _ in range(5):
                        spark_x = pt2[0] + np.random.randint(-15, 16)
                        spark_y = pt2[1] + np.random.randint(-15, 16)
                        cv2.circle(trajectory_layer, (spark_x, spark_y), 2, (255, 255, 255), -1)
        
        # 將軌跡圖層與原始幀合併
        frame = cv2.addWeighted(frame, 1, trajectory_layer, 0.8, 0)
        
        # 添加資訊顯示
        progress = (frame_count / total_frames) * 100
        info_text = f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 顯示球的檢測狀態（使用紅色和綠色）
        if ball_detected:
            ball_status = "Ball Detected"
            ball_color = (0, 255, 0)  # 綠色
        else:
            ball_status = "No Ball Detected"
            ball_color = (0, 0, 255)  # 紅色
        cv2.putText(frame, ball_status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, ball_color, 2)
        
        # 顯示追蹤狀態（使用紅色和綠色）
        if not tracking_started:
            status_text = "Waiting for ball to leave pitcher area"
            status_color = (0, 0, 255)  # 紅色
        else:
            status_text = "Tracking ball"
            status_color = (0, 255, 0)  # 綠色
        cv2.putText(frame, status_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # 寫入幀
        out.write(frame)
        
        # 顯示縮小的視窗
        display_scale = 0.6  # 顯示窗口縮放為原始大小的60%
        display_width = int(width * display_scale)
        display_height = int(height * display_scale)
        display_frame = cv2.resize(frame, (display_width, display_height))
        cv2.imshow('Baseball Tracking', display_frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 釋放資源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 選擇輸入視訊
    video_path = select_video()
    if not video_path:
        print("未選擇視訊檔案，程式退出")
        exit()
    
    # 生成輸出檔案路徑
    output_path = get_output_path()
    
    # 執行追蹤程式
    track_baseball(video_path, output_path) 