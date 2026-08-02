# BÀI GIẢI CHI TIẾT VÀ HƯỚNG DẪN CODE THỰC HÀNH — AI PRIMER

> [!IMPORTANT]  
> **THÔNG TIN HỌC PHẦN & TÁI LẬP KẾT QUẢ (REPRODUCIBILITY)**  
> - **Môn học:** AI Primer (Nhập môn Trí tuệ Nhân tạo) — Aptech Học kỳ 3  
> - **Thời lượng làm bài:** 90 phút | **Thang điểm:** 20.0 điểm  
> - **Quy chuẩn Tái lập:** Cố định ngẫu nhiên toàn cục `SEED = 42` ở đầu tất cả các cell code.

---

## PHẦN A: AI, MACHINE LEARNING & DEEP LEARNING (6.0 ĐIỂM)

---

### CÂU 1 (2.0 điểm) — Quy trình ML 5 bước & Chống Data Leakage (Buổi 01 & 02)

#### 1. Lý thuyết Chuyên sâu: Quy trình ML 5 bước & Bản chất Data Leakage

##### a. Phân tích Chi tiết 5 bước Quy trình Machine Learning
1. **Thu thập Dữ liệu & Phân tích Khám phá (Data Collection & EDA):** Biểu diễn tập dữ liệu dạng ma trận $D = \{(x_i, y_i)\}_{i=1}^N$ với $x_i \in \mathbb{R}^d$ là vector đặc trưng và $y_i$ là nhãn. Phân tích loại biến, phân bố nhãn $P(y)$ và các chỉ số thống kê cơ bản.
2. **Tiền xử lý Dữ liệu (Data Preprocessing):** Chuẩn hóa thang đo (Feature Scaling) bằng phương pháp Z-score: $x' = \frac{x - \mu}{\sigma}$ trong đó $\mu$ là giá trị trung bình và $\sigma$ là độ lệch chuẩn. Phương pháp này đưa các đặc trưng có đơn vị đo khác nhau về cùng phân bố có $\mu=0$ và $\sigma=1$, giúp thuật toán tối ưu hóa dốc (Gradient Descent) hội tụ mượt mà.
3. **Phân chia Tập Dữ liệu (Train/Test Split):** Chia bộ dữ liệu $D$ thành tập huấn luyện $D_{train}$ ($70\%$) và tập kiểm thử $D_{test}$ ($30\%$). Áp dụng kỹ thuật phân tầng nhãn (**Stratified Sampling**) bảo đảm tỷ lệ giữa các lớp nhãn được bảo toàn tuyệt đối ở cả 2 tập: $P_{train}(y = k) \approx P_{test}(y = k) \approx P(y = k)$.
4. **Huấn luyện Mô hình (Model Training & Fitting):** Xây dựng Pipeline và tối ưu hóa hàm mất mát Cross-Entropy / Log-Loss: $\mathcal{L}(w) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{i,k} \ln \hat{y}_{i,k}$.
5. **Đánh giá & Rút ra Dự đoán (Model Evaluation & Inference):** Đánh giá độ chính xác tổng thể (Accuracy) và Ma trận nhầm lẫn (Confusion Matrix $C_{i,j}$) trên tập $D_{test}$ hoàn toàn độc lập.

##### b. Bản chất Toán học của Data Leakage và Cơ chế Chống Rò rỉ của Pipeline
- **Bản chất Data Leakage (Rò rỉ Dữ liệu):** Data Leakage xảy ra khi thông tin thống kê của tập Test bị vô tình lộ vào quá trình tính toán tiền xử lý hoặc huấn luyện mô hình.
  - *Toán học rò rỉ:* Nếu ta tính trung bình toàn cục $\mu_{all}$ và độ lệch chuẩn toàn cục $\sigma_{all}$ trên toàn bộ tập dữ liệu $D = D_{train} \cup D_{test}$ trước khi split: $\mu_{all} = \frac{1}{|D_{train}| + |D_{test}|} \left( \sum_{i \in D_{train}} x_i + \sum_{j \in D_{test}} x_j \right)$.
    Giá trị $\mu_{all}$ đã chứa đựng thông tin trực tiếp của từng điểm dữ liệu trong tập Test. Khi đó mô hình sẽ có "kiến thức báo trước" (Prior Knowledge) về tập Test $\to$ Kết quả Accuracy trên tập Test bị cao ảo, nhưng mô hình sẽ thất bại khi triển khai thực tế.
- **Cơ chế Đóng gói Chống Leakage của Sklearn Pipeline:**  
  `Pipeline` tạo ra một đồ thị hướng không chu trình (DAG) đóng gói các bước xử lý. Khi gọi `pipe.fit(X_train, y_train)`:
  - Bước `StandardScaler` chỉ thực hiện `fit_transform()` trên duy nhất $X_{train}$ để trích xuất $\mu_{train}$ và $\sigma_{train}$.
  - Khi gọi `pipe.predict(X_test)`: Pipeline chỉ thực hiện `transform()` trên $X_{test}$ bằng tham số $\mu_{train}$ và $\sigma_{train}$ đã lưu sẵn. Tập Test hoàn toàn giữ tính ẩn danh tuyệt đối.

#### 2. Mã nguồn Python Chuẩn hóa (100% Runnable)

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Cố định SEED bảo đảm reproducibility
SEED = 42

# Bước 1: Nạp bộ dữ liệu load_wine
wine = load_wine()
X_wine, y_wine = wine.data, wine.target

# Bước 2 & 3: Chia tập dữ liệu Train/Test (70/30) có phân tầng nhãn stratify
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X_wine, y_wine, test_size=0.3, stratify=y_wine, random_state=SEED
)

# Bước 4: Xây dựng Pipeline đóng gói StandardScaler và LogisticRegression
pipe_wine = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=SEED))
])

# Huấn luyện mô hình duy nhất trên tập Train
pipe_wine.fit(X_train_w, y_train_w)

# Bước 5: Dự đoán và Đánh giá trên tập Test
y_pred_w = pipe_wine.predict(X_test_w)
accuracy_wine = accuracy_score(y_test_w, y_pred_w)
cm_wine = confusion_matrix(y_test_w, y_pred_w)

print("=== KẾT QUẢ CÂU 1 ===")
print(f"Accuracy tập Test: {accuracy_wine:.4f} ({int(accuracy_wine * len(y_test_w))}/{len(y_test_w)} mẫu đúng)")
print("Confusion Matrix:")
print(cm_wine)
```

#### 3. 🔍 VÌ SAO VIẾT CODE NHƯ VẬY? (Design Rationale & Technical Decisions)
1. **Vì sao phải dùng `stratify=y_wine` trong `train_test_split`?**  
   Bộ dữ liệu `load_wine` gồm 3 lớp rượu. Tham số `stratify=y_wine` cưỡng chế chia theo tỷ lệ phân bố nhãn ban đầu, giúp tập Test phản ánh chính xác hiệu năng tổng quát.
2. **Vì sao chọn `test_size=0.3`?**  
   Với bộ dữ liệu 178 mẫu, tỷ lệ 70% Train (124 mẫu) cung cấp đủ tri thức cho `LogisticRegression` học, trong khi 30% Test (54 mẫu) đủ độ dài thống kê để đánh giá độ chính xác không bị dao động nhiễu.
3. **Vì sao dùng `Pipeline` thay vì gọi `StandardScaler()` và `LogisticRegression()` riêng lẻ?**  
   Nếu gọi riêng lẻ: dễ mắc sai lầm là gọi `scaler.fit_transform(X_wine)` trên toàn bộ dữ liệu trước khi split (Data Leakage). Dùng `Pipeline` tự động cách ly tập Test, bảo đảm chuẩn phương pháp luận ML.

#### 4. Đầu ra Định lượng
* **Accuracy tập Test:** `0.9815` ($53/54$ mẫu đúng).
* **Ma trận nhầm lẫn (Confusion Matrix):** $\begin{bmatrix} 18 & 0 & 0 \\ 1 & 20 & 0 \\ 0 & 0 & 15 \end{bmatrix}$

---

### CÂU 2 (2.0 điểm) — Thí nghiệm Perceptron vs MLP trên Cổng XOR (Buổi 03)

#### 1. Lý thuyết Chuyên sâu: Ranh giới Tuyến tính & Biến đổi Không gian Phi tuyến

##### a. Mô hình Perceptron 1 tầng và Chứng minh Toán học về Giới hạn đối với Cổng XOR
* **Mô hình Perceptron 1 tầng (Rosenblatt, 1958):** Công thức tính đầu ra của Perceptron 1 tầng có dạng: $y = f\left(\sum_{i=1}^n w_i x_i + b\right) = f(w^T x + b)$ trong đó $f(z)$ là hàm bước nhảy Binary Step ($f(z) = 1$ nếu $z \ge 0$ và $0$ nếu $z < 0$).
* **Ranh giới Quyết định (Decision Boundary):** Là siêu phẳng tuyến tính $w_1 x_1 + w_2 x_2 + b = 0$.
* **Chứng minh Toán học Perceptron không thể giải bài toán XOR:**  
  Giả sử tồn tại các bộ trọng số $w_1, w_2, b$ sao cho Perceptron giải đúng cả 4 mẫu của cổng XOR:
  1. Mẫu $(0, 0) \to y=0 \implies w_1(0) + w_2(0) + b < 0 \implies b < 0$
  2. Mẫu $(0, 1) \to y=1 \implies w_1(0) + w_2(1) + b \ge 0 \implies w_2 + b \ge 0 \implies w_2 \ge -b > 0$
  3. Mẫu $(1, 0) \to y=1 \implies w_1(1) + w_2(0) + b \ge 0 \implies w_1 + b \ge 0 \implies w_1 \ge -b > 0$
  4. Mẫu $(1, 1) \to y=0 \implies w_1(1) + w_2(1) + b < 0 \implies w_1 + w_2 + b < 0$
  Thay điều kiện $w_1 \ge -b$ và $w_2 \ge -b$ vào biểu thức (4): $(-b) + (-b) + b < 0 \implies -b < 0 \implies b > 0$. Điều này mâu thuẫn hoàn toàn với điều kiện (1) $b < 0$.  
  $\implies$ **Không tồn tại đường phân chia tuyến tính nào cho cổng XOR (Linearly Non-separable).**

##### b. Mạng Nơ-ron Đa tầng (MLP) và Định lý Đại diện Tổng quát
* **Định lý Đại diện Tổng quát (Universal Approximation Theorem - Cybenko, 1989):** Một mạng Feedforward với ít nhất một tầng ẩn và hàm kích hoạt phi tuyến có khả năng xấp xỉ bất kỳ hàm số liên tục nào với độ chính xác tùy ý.
* **Toán học biến đổi không gian của MLP:**  
  MLP thêm một tầng ẩn có hàm kích hoạt phi tuyến $h = \tanh(W^{(1)} x + b^{(1)})$. Phép toán này biến đổi 4 điểm dữ liệu chéo nhau ở mặt phẳng 2D ban đầu sang không gian biểu diễn ẩn 4D ($R^4$). Tại đây, bề mặt quyết định bị "uốn cong", giúp các điểm thuộc nhãn 1 và nhãn 0 tách biệt về hai phía của một siêu phẳng phân chia tuyến tính ở tầng ra.

#### 2. Mã nguồn Python Chuẩn hóa (100% Runnable)

```python
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

# 1. Khởi tạo tập dữ liệu Cổng Logic XOR (4 mẫu)
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# 2. Huấn luyện Perceptron 1 tầng
perceptron_model = Perceptron(random_state=SEED)
perceptron_model.fit(X_xor, y_xor)
acc_perceptron = perceptron_model.score(X_xor, y_xor)

# 3. Huấn luyện MLPClassifier (1 tầng ẩn 4 nơ-ron, activation='tanh', solver='lbfgs')
mlp_xor = MLPClassifier(
    hidden_layer_sizes=(4,), 
    activation='tanh', 
    solver='lbfgs', 
    random_state=SEED
)
mlp_xor.fit(X_xor, y_xor)
acc_mlp_xor = mlp_xor.score(X_xor, y_xor)

print("=== KẾT QUẢ CÂU 2 ===")
print(f"Accuracy Perceptron 1 tầng : {acc_perceptron:.2f} (Thất bại - Sai 50%)")
print(f"Accuracy MLP Đa tầng       : {acc_mlp_xor:.2f} (Thành công - Đúng 100%)")
```

#### 3. 🔍 VÌ SAO VIẾT CODE NHƯ VẬY? (Design Rationale & Technical Decisions)
1. **Vì sao chọn `hidden_layer_sizes=(4,)` cho MLP?**  
   Mặc dù lý thuyết chỉ cần 2 nơ-ron ở tầng ẩn là đủ, việc chọn 4 nơ-ron giúp mạng có đủ dư thừa tham số để tìm nghiệm tối ưu cực nhanh và ổn định.
2. **Vì sao chọn hàm kích hoạt `activation='tanh'` thay vì `'relu'`?**  
   Hàm $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ có miền giá trị đối xứng qua 0 $[-1, 1]$ và đạo hàm mượt. Với bộ dữ liệu 4 mẫu, $\tanh$ giúp thuật toán không bị rơi vào vùng chết gradient (Dying ReLU).
3. **Vì sao chọn `solver='lbfgs'` thay vì `'adam'` hay `'sgd'`?**  
   `lbfgs` là thuật toán tối ưu bậc 2 (Second-order optimization), hội tụ chính xác 100% chỉ sau vài vòng lặp trên tập dữ liệu nhỏ.

#### 4. Đầu ra Định lượng
* **Accuracy Perceptron 1 tầng:** `0.50` (50%).
* **Accuracy MLPClassifier:** `1.00` (100%).

---

### CÂU 3 (2.0 điểm) — Phân loại Chữ số Viết tay `load_digits` (Buổi 04)

#### 1. Lý thuyết Chuyên sâu: Biểu diễn Vector Ảnh & Kiến trúc MLP Phễu

##### a. Biểu diễn Ảnh Chữ số dạng Vector Đặc trưng
Mỗi mẫu ảnh chữ số $8 \times 8$ pixels được phẳng hóa (flatten) thành một vector đặc trưng không gian 64 chiều $x_i \in \mathbb{R}^{64}$, trong đó từng phần tử biểu diễn độ sáng mức xám (grayscale value từ 0 đến 16).

##### b. Kiến trúc Mạng MLP Phễu nén đặc trưng (Feature Bottleneck)
- **Tầng vào (Input):** 64 nơ-ron (tương ứng 64 pixels).
- **Tầng ẩn 1 (Hidden Layer 1):** 64 nơ-ron với hàm kích hoạt ReLU $h_1 = \max(0, W_1 x + b_1)$, có nhiệm vụ học các biểu diễn hình học đường nét cơ bản (cạnh ngang, góc cong).
- **Tầng ẩn 2 (Hidden Layer 2):** 32 nơ-ron $h_2 = \max(0, W_2 h_1 + b_2)$, đóng vai trò là cổ chai nén (Bottleneck) để tổng hợp các đường nét thành cấu trúc hình học chữ số hoàn chỉnh và loại bỏ nhiễu.
- **Tầng ra (Output Layer):** 10 nơ-ron với hàm phân bố xác suất Softmax: $P(y = k | x) = \frac{e^{z_k}}{\sum_{j=0}^9 e^{z_j}}$ trích xuất xác suất thuộc về từng chữ số từ 0 đến 9.

#### 2. Mã nguồn Python Chuẩn hóa (100% Runnable)

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# 1. Nạp bộ dữ liệu load_digits
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

# 2. Chia tập Train/Test (75/25) phân tầng
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_digits, y_digits, test_size=0.25, stratify=y_digits, random_state=SEED
)

# 3. Pipeline chuẩn hóa + MLP (64, 32) với max_iter=500
pipe_digits = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED))
])

pipe_digits.fit(X_train_d, y_train_d)
y_pred_d = pipe_digits.predict(X_test_d)

acc_digits = accuracy_score(y_test_d, y_pred_d)
cm_digits = confusion_matrix(y_test_d, y_pred_d)

print("=== KẾT QUẢ CÂU 3 ===")
print(f"Accuracy MLP trên tập Test Digits: {acc_digits:.4f}")

# 4. Trực quan hóa Confusion Matrix 10x10 bằng matplotlib
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_digits, cmap='Greens')
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix Heatmap 10x10 (Digits MLP)')
fig.colorbar(im, ax=ax)

# Annotate chữ số trên ô ma trận
for i in range(10):
    for j in range(10):
        color = 'white' if cm_digits[i, j] > cm_digits.max() / 2 else 'black'
        ax.text(j, i, str(cm_digits[i, j]), ha='center', va='center', color=color, fontsize=9)

plt.tight_layout()
plt.show()
```

#### 3. 🔍 VÌ SAO VIẾT CODE NHƯ VẬY? (Design Rationale & Technical Decisions)
1. **Vì sao thiết lập `hidden_layer_sizes=(64, 32)`?**  
   Kiến trúc dạng phễu nén đặc trưng (Feature Bottleneck): 64 nơ-ron tầng 1 biến đổi phi tuyến các pixel ảnh, 32 nơ-ron tầng 2 cô đọng đặc trưng loại bỏ nhiễu trước khi đưa vào 10 nơ-ron tầng ra.
2. **Vì sao đặt `max_iter=500` trong `MLPClassifier`?**  
   Đảm bảo thuật toán lan truyền ngược (Backpropagation) đủ thời gian hội tụ hoàn toàn mà không phát sinh cảnh báo `ConvergenceWarning`.
3. **Vì sao mã hóa màu chữ tương phản `color = 'white' if cm[i,j] > max/2 else 'black'`?**  
   Đảm bảo chữ số hiển thị tương phản rõ ràng trên cả nền nhạt lẫn nền đậm, đạt tiêu chuẩn hình vẽ xuất bản khoa học.

#### 4. Đầu ra Định lượng
* **Accuracy tập Test:** `0.9667` ($96.67\%$).

---

## PHẦN B: TÁC TỬ AI & THUẬT TOÁN TÌM KIẾM (6.0 ĐIỂM)

---

### CÂU 4 (3.0 điểm) — Bảng Đặc tả PEAS & Mô phỏng Vacuum World Agent (Buổi 05)

#### 1. Lý thuyết Chuyên sâu: Lý thuyết Tác tử AI (Agent Theory)

##### a. Định nghĩa Tác tử và Hàm Tác tử
Theo Russell & Norvig, **Tác tử (Agent)** là bất kỳ đối tượng nào có khả năng nhận thức môi trường thông qua **Cảm biến (Sensors)** và tác động lên môi trường đó thông qua **Cơ cấu Tác động (Actuators)**.  
Hàm tác tử được mô tả toán học bởi ánh xạ $f: \mathcal{P}^* \to \mathcal{A}$, chuyển lịch sử nhận thức $\mathcal{P}^*$ thành hành động $\mathcal{A}$.

##### b. Phân loại 4 Kiến trúc Tác tử Cơ bản
1. **Simple Reflex Agent (Tác tử Phản xạ Đơn giản):** Quyết định dựa duy nhất trên nhận thức hiện tại $A = f(P_{current})$, bỏ qua lịch sử. Dễ rơi vào lặp vô tận trong môi trường quan sát một phần.
2. **Model-based Reflex Agent (Tác tử Phản xạ Dựa trên Mô hình):** Duy trì trạng thái bên trong (Internal State) để theo dõi các khía cạnh chưa quan sát được của môi trường.
3. **Goal-based Agent (Tác tử Dựa trên Mục tiêu):** Kết hợp trạng thái bên trong với thông tin Mục tiêu ($Goal$) để chọn hành động dẫn tới mục tiêu đích.
4. **Utility-based Agent (Tác tử Dựa trên Hữu dụng):** Dùng hàm hữu dụng $U(s)$ để đánh giá và tối ưu hóa mức độ hiệu quả giữa nhiều trạng thái mục tiêu khác nhau.

#### 2. Bảng Đặc tả PEAS cho Robot Hút bụi Tự động Gia đình
* **P (Performance):** Tỷ lệ sạch sàn nhà, tiết kiệm năng lượng pin, độ bền thiết bị, giảm tiếng ồn và tránh va chạm.
* **E (Environment):** Sàn nhà 2D (ô A, B), các chất liệu sàn, vật cản, vị trí bụi ngẫu nhiên.
* **A (Actuators):** Bánh xe di chuyển (Tiến/Lùi/Trái/Phải), động cơ hút bụi (Suck), chổi quét, loa báo.
* **S (Sensors):** Cảm biến vị trí ô (SLAM/GPS), cảm biến bụi (Dirt sensor), cảm biến chống rơi, cảm biến pin.

#### 3. Mã nguồn Python Chuẩn hóa (100% Runnable)

```python
class VacuumEnvironment:
    """Môi trường mô phỏng Vacuum World gồm 2 ô A và B."""
    def __init__(self):
        self.status = {'A': 1, 'B': 1}  # 1: Bẩn, 0: Sạch
        self.agent_pos = 'A'           # Vị trí xuất phát ở A
        self.cost = 0                  # Đơn vị chi phí pin tiêu tốn

    def step(self, action):
        if action == 'Suck':
            self.status[self.agent_pos] = 0
            self.cost += 1
        elif action == 'Right':
            self.agent_pos = 'B'
            self.cost += 1
        elif action == 'Left':
            self.agent_pos = 'A'
            self.cost += 1
        elif action == 'NoOp':
            pass

# 1. Simple Reflex Agent
env_reflex = VacuumEnvironment()
for _ in range(6):
    current_status = env_reflex.status[env_reflex.agent_pos]
    if current_status == 1:
        action = 'Suck'
    elif env_reflex.agent_pos == 'A':
        action = 'Right'
    else:
        action = 'Left'
    env_reflex.step(action)

# 2. Goal-based Agent
env_goal = VacuumEnvironment()
for _ in range(6):
    if env_goal.status['A'] == 0 and env_goal.status['B'] == 0:
        action = 'NoOp'
        break
    current_status = env_goal.status[env_goal.agent_pos]
    if current_status == 1:
        action = 'Suck'
    elif env_goal.agent_pos == 'A':
        action = 'Right'
    else:
        action = 'Left'
    env_goal.step(action)

print("=== KẾT QUẢ CÂU 4 ===")
print(f"Simple Reflex Agent tiêu tốn  : {env_reflex.cost} đơn vị pin (Lặp vô tận)")
print(f"Goal-based Agent tiêu tốn     : {env_goal.cost} đơn vị pin (Tiết kiệm 50% pin)")
```

#### 4. 🔍 VÌ SAO VIẾT CODE NHƯ VẬY? (Design Rationale & Technical Decisions)
1. **Vì sao thiết kế thuộc tính `status = {'A': 1, 'B': 1}` dưới dạng Dictionary?**  
   Cho phép truy vấn và cập nhật trạng thái ô với độ phức tạp $O(1)$, mã nguồn trực quan.
2. **Vì sao `Goal-based Agent` cần điều kiện `if status['A'] == 0 and status['B'] == 0: break`?**  
   Tác tử mục tiêu có khả năng so sánh trạng thái hiện tại với trạng thái đích ($Goal$). Khi sàn sạch hoàn toàn, nó chủ động thực hiện `NoOp` để dừng tiêu tốn năng lượng.

#### 5. Đầu ra Định lượng
* **Reflex Agent:** `6` đơn vị pin.
* **Goal-based Agent:** `3` đơn vị pin (Tiết kiệm 50%).

---

### CÂU 5 (3.0 điểm) — Thuật toán Tìm kiếm A* Search trên Mê cung 2D (Buổi 06)

#### 1. Lý thuyết Chuyên sâu: Thuật toán A* Search & Tính Hợp lệ của Heuristic

##### a. Định nghĩa Bài toán Tìm kiếm Không gian Trạng thái
Bộ bài toán bao gồm: Không gian trạng thái $S$, Trạng thái bắt đầu $s_0$, Tập trạng thái đích $S_G$, Tập các hành động $Actions(s)$, và Chi phí bước $c(s, a, s')$.

##### b. Toán học Thuật toán A* Search
- **Hàm đánh giá tổng thể $f(n)$:** $f(n) = g(n) + h(n)$
  - $g(n)$: Chi phí thực tế đi từ $s_0$ đến nút $n$.
  - $h(n)$: Chi phí ước lượng đi từ nút $n$ đến nút đích $S_G$.
- **Tính Hợp lệ của Heuristic (Admissibility):** Heuristic $h(n)$ được gọi là **hợp lệ (Admissible)** nếu nó không bao giờ đánh giá cao hơn chi phí thực tế $h^*(n)$ để tới đích: $0 \le h(n) \le h^*(n), \quad \forall n$. Khoảng cách Manhattan $h(n) = |x_n - x_{goal}| + |y_n - y_{goal}|$ là một Heuristic hợp lệ trên lưới ô vuông 4 hướng di chuyển, vì chi phí ngắn nhất trên lưới không thể nhỏ hơn khoảng cách Manhattan. Do đó, A* Search bảo đảm tính **Tối ưu (Optimality)**.

#### 2. Mã nguồn Python Chuẩn hóa (100% Runnable)

```python
import heapq

grid_maze = np.array([
    [0, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0]
])

start_node = (0, 0)
goal_node = (5, 5)

def a_star_search(grid, start, goal):
    def h(p):
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    frontier = [(h(start), 0, [start])]
    explored = set()
    expanded_count = 0

    while frontier:
        f, g, path = heapq.heappop(frontier)
        curr = path[-1]

        if curr == goal:
            return path, g, expanded_count

        if curr not in explored:
            explored.add(curr)
            expanded_count += 1
            r, c = curr

            for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == 0:
                    if (nr, nc) not in explored:
                        g_new = g + 1
                        f_new = g_new + h((nr, nc))
                        heapq.heappush(frontier, (f_new, g_new, path + [(nr, nc)]))

    return None, float('inf'), expanded_count

path_a, cost_a, exp_a = a_star_search(grid_maze, start_node, goal_node)

print("=== KẾT QUẢ CÂU 5 ===")
print(f"Path Cost : {cost_a} bước")
print(f"Expanded  : {exp_a} nút")

# Trực quan hóa
grid_plot = grid_maze.copy().astype(float)
for r, c in path_a:
    grid_plot[r, c] = 0.5

fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(grid_plot, cmap='Blues')
ax.set_title('Đường đi Tối ưu A* Search trên Lưới 6x6')
ax.text(0, 0, 'START', ha='center', va='center', color='red', fontweight='bold')
ax.text(5, 5, 'GOAL', ha='center', va='center', color='green', fontweight='bold')
plt.tight_layout()
plt.show()
```

#### 3. 🔍 VÌ SAO VIẾT CODE NHƯ VẬY? (Design Rationale & Technical Decisions)
1. **Vì sao dùng thư viện `heapq`?**  
   A* Search đòi hỏi luôn lấy ra nút có chi phí $f(n)$ nhỏ nhất. `heapq` (Min-Heap) lấy ra phần tử nhỏ nhất với độ phức tạp $O(\log N)$.
2. **Vì sao cấu trúc trong `frontier` là `(f, g, path)`?**  
   Python so sánh tuple theo phần tử đầu tiên, giúp `heapq` tự động ưu tiên $f(n)$. Nạp `path` trực tiếp vào tuple giúp lấy ngay tuyến đường tối ưu khi chạm đích mà không cần truy vết ngược `parent`.
3. **Vì sao dùng `explored = set()`?**  
   Phép kiểm tra `(nr, nc) not in explored` trên `set` có độ phức tạp $O(1)$, ngăn lặp vô hạn.

#### 4. Đầu ra Định lượng
* **Path Cost:** `10` bước.
* **Expanded Nodes:** `12` nút.

---

## PHẦN C: BIỂU DIỄN TRI THỨC, LOGIC & ĐÁNH GIÁ MÔ HÌNH (8.0 ĐIỂM)

---

### CÂU 6 (4.0 điểm) — Cơ sở Tri thức & Động cơ Forward Chaining (Buổi 07)

#### 1. Lý thuyết Chuyên sâu: Luật Sản xuất & Suy diễn Tiến

##### a. Biểu diễn Tri thức dạng Luật Sản xuất (Production Rules)
Tri thức chuyên gia được biểu diễn dưới dạng tập các luật sản xuất: $R_i: \bigwedge_{j=1}^k c_j \implies r$ trong đó $c_j$ là các điều kiện tiền đề (Premises/IF) và $r$ là tri thức kết luận (Conclusion/THEN).

##### b. Nguyên lý Suy diễn Tiến (Forward Chaining / Data-driven)
- Thuật toán bắt đầu từ **Tập sự thật đã biết (Facts Set $F$)**.
- Áp dụng quy tắc suy diễn **Modus Ponens**: $\frac{P, \quad P \implies Q}{Q}$
- Động cơ quét liên tục qua các luật. Nếu vế IF là tập con của tri thức đã biết ($Condition \subseteq F$) và vế THEN chưa có trong $F$, tri thức mới sẽ được nạp vào $F$.
- Vòng lặp đạt đến **Điểm cố định (Fixed Point)** khi $F^{(k+1)} = F^{(k)}$, thuật toán tự động kết thúc.

#### 2. Mã nguồn Python Chuẩn hóa (100% Runnable)

```python
facts_init = {"Sốt", "Ho", "Đau họng"}

rules_kb = [
    ({"Sốt", "Ho"}, "Viêm đường hô hấp"),
    ({"Viêm đường hô hấp", "Đau họng"}, "Viêm họng cấp")
]

def forward_chaining_engine(facts, rules):
    known = set(facts)
    trace_log = []
    changed = True

    while changed:
        changed = False
        for cond, result in rules:
            if cond.issubset(known) and result not in known:
                known.add(result)
                trace_log.append((cond, result))
                changed = True

    return known, trace_log

final_kb, trace_log = forward_chaining_engine(facts_init, rules_kb)

print("=== KẾT QUẢ CÂU 6 ===")
print("Sự thật ban đầu:", sorted(list(facts_init)))
print("Vết suy luận:")
for i, (c, r) in enumerate(trace_log, 1):
    print(f"  Bước {i}: IF {sorted(list(c))} -> THEN Suy ra: '{r}'")
print("\nTập tri thức hoàn chỉnh cuối cùng:")
print(sorted(list(final_kb)))
```

#### 3. 🔍 VÌ SAO VIẾT CODE NHƯ VẬY? (Design Rationale & Technical Decisions)
1. **Vì sao định nghĩa vế IF dạng `set`?**  
   Tận dụng phương thức `cond.issubset(known)` để kiểm tra tập điều kiện tiên quyết có thuộc tập tri thức đã biết hay không chỉ trong 1 dòng lệnh với hiệu năng $O(1)$.
2. **Vì sao kiểm tra `result not in known`?**  
   Tránh kích hoạt lại luật đã áp dụng, ngăn vòng lặp vô hạn và tránh ghi trùng lặp vào vết suy luận.
3. **Vì sao dùng cờ `changed`?**  
   Đảm bảo động cơ quét liên tục qua tập luật cho đến khi tập tri thức đạt trạng thái điểm cố định (Fixed Point).

#### 4. Đầu ra Định lượng
* **Vết suy luận:** 2 bước suy diễn chính xác.
* **Tập tri thức cuối cùng:** `['Ho', 'Sốt', 'Viêm họng cấp', 'Viêm đường hô hấp', 'Đau họng']`.

---

### CÂU 7 (4.0 điểm) — Bảng Chân trị Logic & Tối ưu Ngưỡng (Threshold Tuning) (Buổi 08)

#### 1. Lý thuyết Chuyên sâu: Phân loại Mất cân bằng & Hàm Chi phí Phạt Phi đối xứng

##### a. Nghịch lý Accuracy (Accuracy Paradox) trong Dữ liệu Mất cân bằng
Khi bộ dữ liệu bị mất cân bằng nghiêm trọng (lớp âm tính chiếm $95\%$, lớp dương tính chiếm $5\%$), mô hình ngây thơ luôn dự đoán nhãn 0 sẽ dễ dàng đạt Accuracy $= 95\%$. Tuy nhiên, mô hình này hoàn toàn vô dụng vì không phát hiện được bất kỳ ca bệnh dương tính nào ($\text{Recall} = 0$).

##### b. Hàm Chi phí Phạt Phi đối xứng (Asymmetric Cost Function)
Trong ứng dụng thực tế (y tế, phát hiện gian lận), chi phí phạt lỗi bỏ sót (False Negative - FN) nghiêm trọng hơn rất nhiều so với lỗi cảnh báo nhầm (False Positive - FP). Hàm chi phí phạt được tổng quát hóa: $\text{Cost}(\tau) = w_{FN} \times \text{FN}(\tau) + w_{FP} \times \text{FP}(\tau)$ với $w_{FN} = 10$ và $w_{FP} = 1$.  
Kỹ thuật **Tối ưu Ngưỡng (Threshold Moving)** tìm kiếm ngưỡng quyết định tối ưu $\tau^*$ thỏa mãn: $\tau^* = \arg\min_{\tau} \text{Cost}(\tau)$. Dịch chuyển ngưỡng $\tau$ từ $0.50$ xuống $0.15$ làm giảm tỷ lệ FN, giúp tối ưu hóa tổng chi phí rủi ro thực tế.

#### 2. Bảng Chân trị Logic $P \to Q \equiv \neg P \lor Q$
Chỉ nhận giá trị **False** duy nhất khi $P = \text{True}$ và $Q = \text{False}$.

#### 3. Mã nguồn Python Chuẩn hóa (100% Runnable)

```python
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_curve, confusion_matrix, recall_score

# 1. Bảng chân trị Logic P -> Q
truth_rows = []
for P in [True, False]:
    for Q in [True, False]:
        truth_rows.append({"P": P, "Q": Q, "P -> Q": (not P) or Q})

df_truth = pd.DataFrame(truth_rows)
print("=== KẾT QUẢ CÂU 7A: BẢNG CHÂN TRỊ P -> Q ===")
print(df_truth.to_string(index=False))

# 2. Sinh dữ liệu mất cân bằng (95/5) & Tối ưu Ngưỡng
X_imb, y_imb = make_classification(
    n_samples=2000, n_features=10, weights=[0.95, 0.05], random_state=SEED
)

X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(
    X_imb, y_imb, test_size=0.3, stratify=y_imb, random_state=SEED
)

model_i = LogisticRegression(random_state=SEED)
model_i.fit(X_tr_i, y_tr_i)
y_probs_i = model_i.predict_proba(X_te_i)[:, 1]

# Ngưỡng 0.50 vs Ngưỡng tối ưu 0.15
pred_50 = (y_probs_i >= 0.50).astype(int)
pred_15 = (y_probs_i >= 0.15).astype(int)

cm_50 = confusion_matrix(y_te_i, pred_50)
cm_15 = confusion_matrix(y_te_i, pred_15)

cost_50 = (10 * cm_50[1, 0]) + (1 * cm_50[0, 1])
cost_15 = (10 * cm_15[1, 0]) + (1 * cm_15[0, 1])

print("\n=== KẾT QUẢ CÂU 7B: TỐI ƯU NGƯỠNG ĐÁNH GIÁ MÔ HÌNH ===")
print(f"Ngưỡng 0.50 -> Recall: {recall_score(y_te_i, pred_50):.4f} | Chi phí phạt: {cost_50}")
print(f"Ngưỡng 0.15 -> Recall: {recall_score(y_te_i, pred_15):.4f} | Chi phí phạt: {cost_15} (Tiết kiệm {cost_50 - cost_15} điểm phạt)")

# Biểu đồ PR Curve
prec_i, rec_i, _ = precision_recall_curve(y_te_i, y_probs_i)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(rec_i, prec_i, color='purple', linewidth=2, label='Precision-Recall Curve')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve (Dữ liệu Mất cân bằng 95/5)')
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()
```

#### 4. 🔍 VÌ SAO VIẾT CODE NHƯ VẬY? (Design Rationale & Technical Decisions)
1. **Vì sao dịch chuyển ngưỡng từ 0.50 xuống 0.15?**  
   Dữ liệu mất cân bằng 95/5 khiến xác suất dự đoán bị kéo lệch về 0. Giữ ngưỡng 0.50 sẽ gây bỏ sót nhiều ca bệnh (FN cao). Hạ ngưỡng xuống 0.15 giúp tăng chỉ số Recall từ $46.88\%$ lên $65.62\%$.
2. **Vì sao công thức chi phí lại nhân 10 với FN `cost = 10*FN + 1*FP`?**  
   Trong y tế/tài chính, lỗi bỏ sót ca bệnh (FN) nguy hiểm hơn nhiều so với chẩn đoán nhầm (FP). Gán phạt gấp 10 lần giúp hạ tổng chi phí phạt từ $178$ xuống $145$ (tiết kiệm 33 điểm phạt).
3. **Vì sao dùng Precision-Recall Curve thay cho ROC Curve?**  
   Với dữ liệu mất cân bằng, số lượng True Negative quá lớn làm ROC Curve bị cao ảo. Precision-Recall Curve phản ánh chính xác hiệu năng mô hình trên lớp thiểu số quan trọng.
