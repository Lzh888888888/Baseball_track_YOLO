# 棒球軌跡追蹤系統
(本專案使用Cursor製作)  
這是一個基於YOLOv8的棒球軌跡追蹤系統，可以即時偵測和追蹤棒球的運動軌跡，並產生視覺化的軌跡效果。

## 功能特點

- 即時棒球偵測和追蹤
- 自動產生火球效果軌跡
- 支援影片檔案處理
- 即時顯示處理進度
- 自動儲存處理結果

## 系統需求

- Python 3.8+
- OpenCV
- YOLOv8
- NumPy

## 安裝步驟

1. 複製專案：
```bash
git clone https://github.com/Lzh888888888/Baseball_track_YOLO.git
cd Baseball_track_YOLO
```

2. 安裝相依套件：
```bash
pip install -r requirements.txt
```

3. 下載YOLOv8模型：
- 將訓練好的模型檔案放在專案目錄中
- 確保模型檔案名稱為 `best.pt`

> 本專案使用的YOLOv8模型來自於 [BaseballCV](https://github.com/dylandru/BaseballCV) 專案。

## 使用方法

1. 執行程式：
```bash
python baseball_tracking.py
```

2. 選擇影片檔案：
- 在彈出的檔案選擇對話框中選擇要處理的影片檔案
- 支援常見影片格式（mp4, avi等）
- (第一次執行會下載YOLO模型，需稍等)

3. 查看處理結果：
- 程式會即時顯示處理進度
- 處理完成後，影片會自動儲存在 `yolo_baseball_track` 目錄中

## 操作說明

- 按 'q' 鍵退出程式
- 等待處理完成，影片會自動儲存

## 注意事項

- 確保影片畫面清晰，光線充足
- 建議使用高幀率影片以獲得更好的追蹤效果
- 處理大檔案時可能需要較長時間

## 輸出說明

處理後的影片將包含：
- 棒球偵測框
- 火球效果軌跡
- 處理進度顯示
- 狀態資訊顯示

## 技術細節

- 使用YOLOv8進行棒球偵測
- 採用高斯平滑處理軌跡
- 實現多層發光效果
- 支援影片縮放處理

## 演示效果

以下演示影片使用本工具進行棒球軌跡追蹤：

### 程式處理效果
#### 原始影片追蹤效果
https://github.com/user-attachments/assets/40857fdd-85b9-4f70-8b62-2d9d789ccd72

> **備註**：
> - 演示影片來源：[YouTube](https://www.youtube.com/watch?v=0vXIiNiDhLk&t=44s&ab_channel=%E6%84%9B%E7%88%BE%E9%81%94%E9%AB%94%E8%82%B2%E5%AE%B6%E6%97%8FELTASports)

## 授權說明

本專案採用 MIT 授權條款 