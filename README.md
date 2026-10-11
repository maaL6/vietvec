# 🇻🇳 VietVec: Vietnamese Word2Vec from Scratch

Xây dựng mô hình biểu diễn ngữ nghĩa từ (Word Embedding - Word2Vec) từ đầu (from scratch) dành riêng cho tiếng Việt, triển khai kiến trúc Skip-gram kết hợp Negative Sampling (SGNS) trên PyTorch.

---

## 📌 Tính năng chính

- **Tiền xử lý đặc thù tiếng Việt**: Hỗ trợ chuẩn hóa Unicode, loại bỏ ký tự rác và tách từ (Word Segmentation) với `pyvi` / `underthesea`.
- **Kiến trúc Skip-gram**: Tự cài đặt hai ma trận trọng số $W_{in}$ (center embeddings) và $W_{out}$ (context embeddings).
- **Negative Sampling Loss**: Tự cài đặt hàm mục tiêu binary cross-entropy và bộ lấy mẫu phân phối Unigram mũ $0.75$ ($U(w)^{3/4}$).
- **Subsampling kỹ thuật Mikolov**: Giảm thiểu ảnh hưởng của các hư từ/từ xuất hiện quá nhiều.
- **Đánh giá & Trực quan hóa**:
  - Tìm kiếm từ đồng nghĩa (Cosine Similarity).
  - Phép loại suy từ vựng (Word Analogy: $A - B + C = D$).
  - Trực quan hóa không gian vector 2D với t-SNE / PCA.
  - Xuất định dạng vector chuẩn `.vec`.

---

## 📂 Cấu trúc dự án

```text
vietvec/
├── data/
│   ├── raw/             # Dữ liệu văn bản thô (Wikipedia, tin tức, ...)
│   └── processed/       # Dữ liệu đã tách từ và làm sạch
├── checkpoints/         # Lưu checkpoint mô hình trong quá trình huấn luyện
├── notebooks/           # Jupyter Notebooks phân tích, visualize t-SNE
├── vietvec/             # Core module
│   ├── __init__.py
│   ├── preprocessing.py # Tokenizer, Vocabulary builder, Subsampling
│   ├── dataset.py       # Dataset, Dataloader & Negative Sampler
│   ├── model.py         # Mạng Skip-gram / CBOW
│   ├── trainer.py       # Vòng lặp huấn luyện, Learning Rate Scheduler
│   └── evaluate.py      # Cosine Similarity, Analogy & Export
├── requirements.txt
├── train.py             # Script chạy huấn luyện chính
└── README.md
```

---

## 🚀 Cài đặt & Bắt đầu

### 1. Cài đặt môi trường
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 📅 Lộ trình thực hiện (2 tuần)

- [x] **Khởi tạo dự án & Cấu trúc thư mục**
- [ ] **Giai đoạn 1**: Tiền xử lý dữ liệu tiếng Việt & Xây dựng Vocabulary
- [ ] **Giai đoạn 2**: Xây dựng Data Pipeline & Negative Sampler
- [ ] **Giai đoạn 3**: Xây dựng Kiến trúc Mô hình & Loss Function
- [ ] **Giai đoạn 4**: Huấn luyện mô hình trên GPU với Public Dataset
- [ ] **Giai đoạn 5**: Đánh giá định tính, định lượng & Trực quan hóa t-SNE

---

## 📄 Bản quyền
Dự án được phân phối dưới giấy phép MIT.
