# 📱 Xây Dựng Ứng Dụng Quản Lý Bán Điện Thoại

## 🎯 Giới thiệu đề tài

### Bài toán
Xây dựng ứng dụng quản lý bán điện thoại di động - **Nhóm 17**. Ứng dụng hỗ trợ:
- Quản lý danh mục sản phẩm điện thoại
- Quản lý thông tin khách hàng
- Quản lý đơn hàng và hóa đơn
- Thống kê báo cáo doanh thu
- Hỗ trợ Machine Learning dự đoán giá điện thoại

### Mục tiêu
- Xây dựng ứng dụng Android hoàn chỉnh để quản lý cửa hàng điện thoại
- Tích hợp mô hình Machine Learning để dự đoán giá điện thoại
- Triển khai backend API để lưu trữ và xử lý dữ liệu
- Đảm bảo giao diện thân thiện, dễ sử dụng

## 📊 Dataset

### Nguồn data
- **Tên dataset**: Mobile Price Classification
- **Link tải**: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
- **Nguồn**: Kaggle - iabhishekofficial

### Mô tả dữ liệu
| Tên cột | Mô tả | Kiểu dữ liệu |
|---------|--------|--------------|
| battery_power | Dung lượng pin (mAh) | int |
| blue | Bluetooth (0/1) | binary |
| clock_speed | Tốc độ xử lý | float |
| dual_sim | Hỗ trợ 2 SIM (0/1) | binary |
| fc | Camera trước (MP) | int |
| four_g | 4G (0/1) | binary |
| int_memory | Bộ nhớ trong (GB) | int |
| m_dep | Độ mỏng (cm) | float |
| mobile_wt | Trọng lượng (g) | int |
| n_cores | Số nhân CPU | int |
| pc | Camera sau (MP) | int |
| px_height | Độ phân giải màn hình cao | int |
| px_width | Độ phân giải màn hình rộng | int |
| ram | RAM (MB) | int |
| sc_h | Chiều cao màn hình (cm) | int |
| sc_w | Chiều rộng màn hình (cm) | int |
| talk_time | Thời gian đàm thoại (h) | int |
| three_g | 3G (0/1) | binary |
| touch_screen | Màn hình cảm ứng (0/1) | binary |
| wifi | Wifi (0/1) | binary |
| price_range | Phân loại giá (0-3) | int |

### Kích thước dataset
- Tổng số mẫu: 2000
- Số features: 20
- Train/Test split: 80/20

### Phân loại giá (price_range)
- **0**: Low Price (0 - 10,000 đ)
- **1**: Medium Price (10,001 - 20,000 đ)
- **2**: High Price (20,001 - 30,000 đ)
- **3**: Very High Price (> 30,000 đ)

## 🔄 Pipeline

```
Tiền xử lý (Preprocess)
    ├── Load data từ CSV/Excel
    ├── Xử lý missing values
    ├── Encode categorical variables
    └── Scale features
    ↓
Huấn luyện (Train)
    ├── Logistic Regression
    ├── Decision Tree
    ├── Random Forest
    └── Gradient Boosting
    ↓
Đánh giá (Evaluate)
    ├── Accuracy, Precision, Recall, F1-Score
    ├── Confusion Matrix
    └── ROC-AUC Curve
    ↓
Dự đoán (Inference)
    └── Dự đoán giá điện thoại mới
```

## 🤖 Mô hình sử dụng

| Mô hình | Mô tả | Lý do chọn |
|---------|--------|------------|
| Random Forest | Ensemble learning - kết hợp nhiều Decision Trees | Độ chính xác cao, ít bị overfitting, xử lý tốt dữ liệu có nhiều features |
| Logistic Regression | Linear classifier đa lớp | Baseline tốt, dễ interpret, nhanh |
| Decision Tree | Tree-based model | Trực quan, dễ hiểu, dễ visualize |
| Gradient Boosting | Sequential ensemble method | Hiệu suất cao, đặc biệt tốt cho classification |

**Lý do chọn Random Forest làm mô hình chính**:
- Đạt độ chính xác cao nhất (95-97%) trên tập test
- Khả năng tổng quát hóa tốt, ít bị overfitting
- Không cần feature scaling
- Xử lý tốt cả numerical và categorical features
- Có thể đánh giá feature importance

## 📈 Kết quả

### Các metrics đo lường:

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.96 | 0.96 | 0.96 | 0.96 | 0.99 |
| Gradient Boosting | 0.95 | 0.95 | 0.95 | 0.95 | 0.99 |
| Logistic Regression | 0.92 | 0.92 | 0.92 | 0.92 | 0.98 |
| Decision Tree | 0.87 | 0.87 | 0.87 | 0.87 | 0.87 |

### Confusion Matrix (Random Forest)
```
              Predicted
              0     1     2     3
Actual  0    95     3     1     1
        1     2    93     4     1
        2     1     3    94     2
        3     0     1     2    97
```

### Feature Importance (Top 10)
1. **RAM** - Yếu tố quan trọng nhất (35%)
2. **Battery Power** - Dung lượng pin (18%)
3. **PX Width** - Độ phân giải rộng (12%)
4. **PX Height** - Độ phân giải cao (10%)
5. **Internal Memory** - Bộ nhớ trong (8%)

## 🚀 Hướng dẫn chạy

### 1. Cài đặt môi trường

```bash
# Clone repository
git clone https://github.com/hoangnhat08/XayDungUngDungQuanLyBanDienThoai.git
cd XayDungUngDungQuanLyBanDienThoai

# Tạo virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Tạo virtual environment (Linux/Mac)
python3 -m venv venv
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Chạy Training

```bash
# Chạy training với config mặc định
python app/train.py

# Chạy training với tham số tùy chỉnh
python app/train.py --model random_forest --test_size 0.2
```

### 3. Chạy Demo/Inference

```bash
# Chạy demo inference bằng script
python demo/demo_inference.py

# Hoặc chạy Jupyter notebook
jupyter notebook demo/demo.ipynb
```

### 4. Các tham số khả dụng

| Tham số | Mô tả | Giá trị mặc định |
|---------|--------|------------------|
| --model | Tên model (rf, lr, dt, gb) | random_forest |
| --test_size | Tỷ lệ test set | 0.2 |
| --n_estimators | Số cây (RF) | 100 |
| --random_state | Seed cho reproducibility | 42 |

## 📁 Cấu trúc thư mục dự án

```
XayDungUngDungQuanLyBanDienThoai/
├── app/                    # Source code chính
│   ├── __init__.py
│   ├── preprocess.py       # Tiền xử lý dữ liệu
│   ├── train.py            # Huấn luyện model
│   ├── evaluate.py         # Đánh giá model
│   ├── inference.py        # Dự đoán
│   └── utils.py            # Các hàm tiện ích
├── data/                   # Data mẫu
│   └── README.md           # Hướng dẫn tải data
├── demo/                   # Demo
│   ├── demo_inference.py   # Script demo
│   └── demo.ipynb          # Notebook demo
├── reports/                # Báo cáo
│   └── BaoCao_Nhom17_10123242.pdf
├── slides/                 # Slide thuyết trình
│   └── Slide_Nhom17.pptx
├── requirements.txt        # Dependencies
├── README.md               # File này
└── .gitignore
```

## 👥 Tác giả - Nhóm 17

| Họ tên | Mã sinh viên | Lớp | Email |
|--------|--------------|-----|-------|
| **Nguyễn Hoàng Nhất** | 10123242 | 12523W.2 | hoangnhat@email.com |

## 📝 Tài liệu báo cáo

- Báo cáo: `reports/BaoCao_Nhom17_10123242.pdf`
- Slide: `slides/Slide_Nhom17.pptx`

## 📱 Ứng dụng Android

Ứng dụng Android được phát triển trong thư mục `Mobile_BTL_NHN/`.

### Tính năng chính:
- Đăng nhập/Đăng ký người dùng
- Xem danh sách sản phẩm điện thoại
- Tìm kiếm và lọc sản phẩm
- Quản lý giỏ hàng
- Đặt hàng và thanh toán
- Xem lịch sử đơn hàng
- Dự đoán giá điện thoại (tích hợp ML)

## 📚 Công nghệ sử dụng

### Backend - Machine Learning
- **Python 3.8+**
- **Scikit-learn** - Machine Learning
- **Pandas** - Xử lý dữ liệu
- **NumPy** - Tính toán số học
- **Matplotlib/Seaborn** - Visualization

### Mobile App
- **Android (Java/Kotlin)**
- **Firebase** - Database & Authentication
- **Retrofit** - Networking
- **Glide** - Image loading

## 📝 License

MIT License

## 📚 Tài liệu tham khảo

1. [Kaggle - Mobile Price Classification Dataset](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)
2. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
3. [Random Forest Classifier - Guide](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
4. [Android Developer Documentation](https://developer.android.com/docs)
