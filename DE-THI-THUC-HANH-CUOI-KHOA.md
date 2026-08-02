# Đề thi Thực hành Cuối khóa — AI Primer (Tổng hợp 8 Buổi)

| Thuộc tính | Giá trị |
|---|---|
| Môn học | **AI Primer (Nhập môn Trí tuệ Nhân tạo)** |
| Thời lượng làm bài | **90 phút** |
| Hình thức thi | **Thực hành trên máy tính (Jupyter Notebook `.ipynb`)** |
| Thang điểm | **20.0 điểm** |
| Đơn vị đào tạo | Aptech — Học kỳ 3 |
| Thư mục lưu trữ | `de-thi-cuoi-khoa/` |

---

## 📋 Hướng dẫn Tổng quan & Quy định Làm bài

1. **Yêu cầu Nộp bài:**
   - Học viên tạo file Jupyter Notebook đặt tên theo cú pháp: `[HoTen]_[MSSV]_BaiLamCuoiKhoa.ipynb`.
   - Tất cả các cell code phải được **thực thi thành công 100%** và lưu sẵn kết quả hiển thị (*output stream* & *đồ thị matplotlib*).
   - Khởi tạo ngẫu nhiên phải được cố định `SEED = 42` ở đầu notebook để bảo đảm tính tái lập kết quả (*reproducibility*).

2. **Cấu trúc Thang điểm (Tổng 20.0 điểm):**
   - **Phần A: AI, Machine Learning & Deep Learning (6.0 điểm)** — Bao gồm Câu 1, Câu 2, Câu 3.
   - **Phần B: Tác tử AI & Thuật toán Tìm kiếm (6.0 điểm)** — Bao gồm Câu 4, Câu 5.
   - **Phần C: Biểu diễn Tri thức, Logic & Đánh giá Mô hình (8.0 điểm)** — Bao gồm Câu 6, Câu 7.

---

## 📝 NỘI DUNG ĐỀ THI VÀ BIỂU ĐIỂM CHI TIẾT (THANG ĐIỂM 20)

### PHẦN A: AI, MACHINE LEARNING & DEEP LEARNING (6.0 ĐIỂM)

#### **Câu 1 (2.0 điểm) — Quy trình ML 5 bước & Chống Data Leakage (Buổi 01 & 02)**
Cho bộ dữ liệu Phân loại Rượu `load_wine` từ `sklearn.datasets`.
- **Yêu cầu thực hành:**
  1. *(0.5 điểm)* Chia tập dữ liệu thành **70% Train** và **30% Test** có phân tầng nhãn `stratify=y` với `random_state=42`.
  2. *(1.0 điểm)* Xây dựng một `Pipeline` gồm 2 bước:
     - Bước 1: `StandardScaler()`
     - Bước 2: Mô hình `LogisticRegression(random_state=42)`
     
     Huấn luyện Pipeline trên tập Train và dự đoán trên tập Test.
  3. *(0.5 điểm)* Báo cáo chỉ số **Accuracy** và in Ma trận nhầm lẫn **Confusion Matrix**. Viết 2–3 câu giải thích ngắn gọn: *Vì sao việc bọc `StandardScaler` vào trong `Pipeline` giúp ngăn chặn hiện tượng Rò rỉ Dữ liệu (Data Leakage)?*

#### **Câu 2 (2.0 điểm) — Thí nghiệm Perceptron vs MLP trên Cổng XOR (Buổi 03)**
Cho tập dữ liệu 4 mẫu đại diện cho Cổng Logic XOR: $X = [[0,0], [0,1], [1,0], [1,1]]$, $y = [0, 1, 1, 0]$.
- **Yêu cầu thực hành:**
  1. *(0.75 điểm)* Huấn luyện mô hình Nơ-ron 1 tầng `Perceptron(random_state=42)`. Tính chỉ số Accuracy.
  2. *(0.75 điểm)* Huấn luyện mô hình Mạng Nơ-ron Đa tầng `MLPClassifier(hidden_layer_sizes=(4,), activation='tanh', solver='lbfgs', random_state=42)`. Tính chỉ số Accuracy.
  3. *(0.50 điểm)* Giải thích ngắn gọn: *Vì sao Perceptron 1 tầng thất bại (chỉ đạt 50%), trong khi Mạng Nơ-ron Đa tầng MLP lại giải quyết thành công 100% bài toán XOR?*

#### **Câu 3 (2.0 điểm) — Phân loại Chữ số Viết tay `load_digits` (Buổi 04)**
Cho bộ dữ liệu ảnh chữ số viết tay `load_digits` (1797 mẫu ảnh 8x8 pixels).
- **Yêu cầu thực hành:**
  1. *(1.0 điểm)* Chia tập Train/Test (75/25) và xây dựng Pipeline chuẩn hóa + `MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)`. Tính Accuracy trên tập Test.
  2. *(1.0 điểm)* Vẽ và hiển thị biểu đồ Ma trận nhầm lẫn 10x10 (*Confusion Matrix Heatmap*) bằng `matplotlib`.

---

### PHẦN B: TÁC TỬ AI & THUẬT TOÁN TÌM KIẾM (6.0 ĐIỂM)

#### **Câu 4 (3.0 điểm) — Bảng Đặc tả PEAS & Mô phỏng Vacuum World Agent (Buổi 05)**
- **Yêu cầu thực hành:**
  1. *(1.0 điểm)* Viết bảng **Đặc tả PEAS** (Performance, Environment, Actuators, Sensors) cho hệ thống **Robot Hút bụi Tự động trong Gia đình**.
  2. *(1.5 điểm)* Lập trình mô phỏng môi trường Vacuum World gồm 2 ô A và B (ban đầu đều Bẩn). Cho 2 loại Tác tử chạy trong 6 bước:
     - **Simple Reflex Agent:** Phản xạ đơn giản dựa trên ô hiện tại.
     - **Goal-based Agent:** Dừng hoạt động (`NoOp`) khi cả 2 ô đã sạch để tiết kiệm năng lượng.
  3. *(0.5 điểm)* Báo cáo tổng chi phí năng lượng tiêu tốn và rút ra nhận xét so sánh hiệu năng.

#### **Câu 5 (3.0 điểm) — Thuật toán Tìm kiếm A* Search trên Mê cung 2D (Buổi 06)**
Cho lưới Mê cung 6x6 (với `0` là đường đi, `1` là tường chướng ngại vật):
```python
import numpy as np

grid = np.array([
    [0, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0]
])
start = (0, 0)
goal = (5, 5)
```
- **Yêu cầu thực hành:**
  1. *(2.0 điểm)* Cài đặt thuật toán **A* Search** với hàm Heuristic Manhattan: $h(n) = |x_n - x_{\text{goal}}| + |y_n - y_{\text{goal}}|$
  2. *(1.0 điểm)* Xuất kết quả: **Chi phí đường đi ngắn nhất (Path Cost)**, **Số nút đã mở rộng (Expanded Nodes)** và vẽ trực quan đường đi trên lưới bằng `matplotlib`.

---

### PHẦN C: BIỂU DIỄN TRI THỨC, LOGIC & ĐÁNH GIÁ MÔ HÌNH (8.0 ĐIỂM)

#### **Câu 6 (4.0 điểm) — Cơ sở Tri thức & Động cơ Suy diễn Tiến Forward Chaining (Buổi 07)**
Cho Tập sự thật ban đầu $F = \{\text{"Sốt"}, \text{"Ho"}, \text{"Đau họng"}\}$ và Tập luật Sản xuất (Rules):
- Luật 1: `IF {"Sốt", "Ho"} THEN "Viêm đường hô hấp"`
- Luật 2: `IF {"Viêm đường hô hấp", "Đau họng"} THEN "Viêm họng cấp"`

- **Yêu cầu thực hành:**
  1. *(2.5 điểm)* Cài đặt Động cơ Suy diễn Tiến **Forward Chaining Engine** bằng Python.
  2. *(1.5 điểm)* Chạy thuật toán suy diễn và in ra **Vết suy luận (Trace)** từng bước (Luật nào được áp dụng, suy ra tri thức mới nào) và Tập tri thức hoàn chỉnh cuối cùng.

#### **Câu 7 (4.0 điểm) — Bảng Chân trị Logic & Tối ưu Ngưỡng (Threshold Tuning) (Buổi 08)**
- **Yêu cầu thực hành:**
  1. *(1.5 điểm)* Lập trình Python xuất **Bảng chân trị Logic Mệnh đề** cho phép kéo theo: $P \to Q \equiv \neg P \lor Q$
  2. *(1.5 điểm)* Cho mô hình Logistic Regression phân loại trên tập dữ liệu mất cân bằng (95% Âm tính, 5% Dương tính). Dự đoán xác suất `y_probs`. Tính chỉ số **Accuracy** và **Recall** tại Ngưỡng mặc định ($Threshold = 0.50$).
  3. *(1.0 điểm)* Thực hiện **Tối ưu Ngưỡng (Threshold Tuning)**: Dịch chuyển ngưỡng từ 0.50 xuống ngưỡng tối ưu (ví dụ $Threshold = 0.15$) để tối thiểu hóa Hàm chi phí phạt sai sót: $\text{Cost} = (10 \times \text{FN}) + (1 \times \text{FP})$. Vẽ đồ thị **Precision-Recall Curve** và đồ thị **Chi phí phạt theo Ngưỡng**.

---

## 🎯 Bảng Ma trận Thang điểm 20 (Bareme Điểm Chi tiết)

| Tiêu chí Đánh giá | Tiêu chuẩn Chấm điểm | Điểm tối đa |
|---|---|---:|
| **Câu 1 (Buổi 1-2)** | Chia dữ liệu 0.5đ + Pipeline LogisticRegression 1.0đ + Giải thích Data Leakage 0.5đ | **2.0 điểm** |
| **Câu 2 (Buổi 3)** | Perceptron XOR 0.75đ + MLP XOR 0.75đ + Giải thích tính phi tuyến 0.5đ | **2.0 điểm** |
| **Câu 3 (Buổi 4)** | Huấn luyện Digits MLP >95% Acc 1.0đ + Confusion Matrix 10x10 Heatmap 1.0đ | **2.0 điểm** |
| **Câu 4 (Buổi 5)** | Bảng PEAS 1.0đ + Code mô phỏng Vacuum Agent 1.5đ + So sánh chi phí pin 0.5đ | **3.0 điểm** |
| **Câu 5 (Buổi 6)** | Code A* Search & Manhattan Heuristic 2.0đ + Đường đi & Đồ thị Maze 1.0đ | **3.0 điểm** |
| **Câu 6 (Buổi 7)** | Động cơ Forward Chaining 2.5đ + Trace vết suy luận & Tập tri thức cuối 1.5đ | **4.0 điểm** |
| **Câu 7 (Buổi 8)** | Bảng chân trị Logic 1.5đ + Logistic Imbalanced 1.5đ + Threshold Tuning & PR Curve 1.0đ | **4.0 điểm** |
| **TỔNG CỘNG** | **HOÀN THÀNH XUẤT SẮC TOÀN BỘ 7 CÂU HỎI** | **20.0 ĐIỂM** |

---

## 📌 Hướng dẫn Nộp bài & Tiêu chuẩn Đánh giá Bài làm

- **Điểm Giỏi (16.0 – 20.0 điểm):** Thực thi đầy đủ 7 câu, code chạy 100% không lỗi, các đồ thị `matplotlib` hiển thị trực quan sinh động, phần diễn giải lý thuyết chính xác và thuyết phục.
- **Điểm Khá (12.0 – 15.5 điểm):** Thực hiện được 5–6 câu, code chạy đúng, đồ thị hiển thị cơ bản.
- **Điểm Trung bình (10.0 – 11.5 điểm):** Thực hiện được 3–4 câu, code có lỗi nhỏ hoặc thiếu hình vẽ trực quan.
- **Điểm Yếu (< 10.0 điểm):** Không nộp notebook thực thi hoặc code lỗi không chạy được quá nửa bài.
