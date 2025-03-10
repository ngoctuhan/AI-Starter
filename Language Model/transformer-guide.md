# Hướng dẫn từng bước về kiến trúc Transformer

## PHẦN 1: ENCODER TRANSFORMER

### Ví dụ đầu vào: "I am single" (3 tokens)

### Bước 1: Embedding và Position Encoding
* Mỗi token được embedding thành vector có kích thước 768
* Input: "I am single" → 3 tokens
* Kích thước sau embedding: [1, 3, 768]
  - 1: batch size
  - 3: số lượng tokens
  - 768: kích thước embedding

### Bước 2: Multi-Head Attention
* **Tính Q, K, V cho mỗi token:**
  - Q: [1, 3, 768]
  - K: [1, 3, 768]
  - V: [1, 3, 768]

* **Chia thành nhiều heads (giả sử 8 heads):**
  - 768 / 8 = 96 (kích thước mỗi head)
  - Q: [1, 3, 8, 96]
  - K: [1, 3, 8, 96]
  - V: [1, 3, 8, 96]

* **Tính attention scores:**
  - Chuyển vị K: [1, 8, 96, 3]
  - Q·K^T: [1, 3, 8, 3]
  - Chia cho sqrt(d_k) = sqrt(96)
  - Áp dụng softmax → attention weights: [1, 3, 8, 3]

* **Tính attention output:**
  - Weights · V: [1, 3, 8, 96]
  - Nối (concatenate) 8 heads lại: [1, 3, 768]
  - Áp dụng linear projection: [1, 3, 768]

### Bước 3: Add & Norm (Residual Connection)
* Cộng với input ban đầu (residual connection): [1, 3, 768]
* Áp dụng Layer Normalization: [1, 3, 768]

### Bước 4: Feed-Forward Network
* Input: [1, 3, 768]
* Mở rộng lên hidden size lớn hơn: [1, 3, 3072]
* Áp dụng ReLU
* Trở về kích thước ban đầu: [1, 3, 768]

### Bước 5: Add & Norm (lần 2)
* Cộng với đầu vào của FFN (residual connection): [1, 3, 768]
* Áp dụng Layer Normalization: [1, 3, 768]
* Đây là output của một block Transformer

**Lặp lại Bước 2-5 cho nhiều blocks (thường 12, 24 hoặc nhiều hơn)**

## PHẦN 2: CLASSIFICATION TASK

### Cách 1: Sử dụng [CLS] token
* Thêm token [CLS] ở đầu câu: "[CLS] I am single"
* Kích thước đầu vào: [1, 4, 768] (4 tokens)
* Sau khi qua toàn bộ encoder: [1, 4, 768]
* Lấy representation của token [CLS]: [1, 768]
* Đưa qua linear layer để phân loại: [1, num_classes]

### Cách 2: Các phương pháp khác
* Lấy trung bình tất cả tokens: [1, 768]
* Hoặc sử dụng max-pooling: [1, 768]
* [CLS] có thể đặt ở đầu hoặc cuối câu

## PHẦN 3: DECODER CHO MÔ HÌNH SEQUENCE-TO-SEQUENCE (DỊCH MÁY)

### Ví dụ đầu vào Encoder: "I am single"
### Output Encoder: [1, 3, 768]

### Bước 1: Token đầu tiên của Decoder
* Bắt đầu với token <START>
* Sau embedding: [1, 1, 768]

### Bước 2: Masked Self-Attention
* Input: [1, 1, 768]
* Tính Q, K, V tương tự như encoder
* Áp dụng masked attention (chỉ nhìn được các token đã có)
* Output: [1, 1, 768]

### Bước 3: Cross-Attention (kết nối với Encoder)
* **Query từ Decoder output:**
  - Q: [1, 1, 768] → [1, 1, 8, 96]

* **Key, Value từ Encoder output:**
  - K: [1, 3, 768] → [1, 3, 8, 96]
  - V: [1, 3, 768] → [1, 3, 8, 96]

* **Tính Cross-Attention:**
  - Chuyển vị K: [1, 8, 96, 3]
  - Q·K^T: [1, 1, 8, 3] (1 từ decoder × 3 từ encoder)
  - Chia cho sqrt(d_k), áp dụng softmax
  - Nhân với V: [1, 1, 8, 96]
  - Nối heads lại: [1, 1, 768]
  - Áp dụng linear projection: [1, 1, 768]

### Bước 4: Feed-Forward Network
* Tương tự như trong encoder
* Output: [1, 1, 768]

### Bước 5: Dự đoán token tiếp theo
* Lấy representation của token cuối cùng: [1, 768]
* Đưa qua linear layer: [1, vocab_size]
* Áp dụng softmax để có xác suất
* Chọn token có xác suất cao nhất (hoặc sampling)

### Bước 6: Lặp lại quá trình với token mới
* Giả sử token dự đoán là "Tôi"
* Đầu vào mới: "<START> Tôi"
* Kích thước mới: [1, 2, 768]

* **Masked Self-Attention mới:**
  - Token thứ 1 chỉ nhìn được token 1
  - Token thứ 2 nhìn được token 1 và 2
  - Output: [1, 2, 768]

* **Cross-Attention mới:**
  - Q: [1, 2, 8, 96] (từ decoder)
  - K, V: [1, 3, 8, 96] (từ encoder)
  - Output: [1, 2, 8, 96] → [1, 2, 768]

* **Dự đoán token tiếp theo:**
  - Chỉ sử dụng representation của token cuối cùng: [1, 768]
  - Dự đoán token tiếp theo

* Tiếp tục quá trình cho đến khi gặp token <END> hoặc đạt max_length

## PHẦN 4: ĐIỂM KHÁC BIỆT QUAN TRỌNG GIỮA ENCODER VÀ DECODER

1. **Self-Attention:**
   - Encoder: Mỗi token nhìn thấy tất cả các tokens khác
   - Decoder: Mỗi token chỉ nhìn thấy các tokens đứng trước nó (masked)

2. **Cross-Attention:**
   - Chỉ có trong Decoder
   - Q từ Decoder, K và V từ Encoder
   - Cho phép Decoder "tham khảo" thông tin từ câu gốc

3. **Quá trình sinh text:**
   - Encoder: Xử lý toàn bộ câu cùng một lúc
   - Decoder: Sinh ra từng token một, autoregressive

## PHẦN 5: TÓM TẮT KÍCH THƯỚC TENSOR

### Encoder:
* Input (3 tokens): [1, 3, 768]
* Self-Attention:
  - Q, K, V: [1, 3, 768] → [1, 3, 8, 96]
  - Attention Scores: [1, 3, 8, 3]
  - Output: [1, 3, 768]

### Decoder:
* Input ban đầu (1 token): [1, 1, 768]
* Masked Self-Attention: [1, 1, 768]
* Cross-Attention:
  - Q: [1, 1, 8, 96]
  - K, V: [1, 3, 8, 96]
  - Attention Scores: [1, 1, 8, 3]
  - Output: [1, 1, 768]
* Final Output: [1, 1, 768]
* Dự đoán: [1, vocab_size]

*Lưu ý: Số lượng blocks trong Transformer phụ thuộc vào độ phức tạp của task và lượng dữ liệu. BERT base có 12 blocks, BERT large có 24 blocks, GPT-3 có tới 96 blocks.*
