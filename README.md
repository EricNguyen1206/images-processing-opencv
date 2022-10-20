#images-processing-opencv

### Khử nhiễu, tăng/giảm độ sáng và cân bằng độ tương phản

1. Khử nhiễu (Gaussian, Poisson dùng Bilateral filter, salt and pepper dùng median filter).
2. Tăng/ giảm độ sáng dùng gamma operator (gamma > 1 độ sáng tăng, ngược lại)
3. Cân bằng độ tương phản (cân bằng hist)

### Phát hiện cạnh

- Dùng Canny filter

### Face detection

- Dùng Haar Cascade filter

### Face recognition

- Ứng dụng Face detection để phát hiện các khuôn mặt -> crop -> dùng sharping filter -> ảnh dùng để train.
- Trích xuất đặc trưng -> dùng thuật toán HOG (Histogram of oriented gradient) -> đặc trưng ảnh tương ứng với nhãn
- Dùng bộ phân lớp SVM với các tham số (kernel: poly, gamma: auto, descition_function_shape: ovo -> vì multiple label)
- Đưa các đặc trưng và nhãn ở bước 2 vào train -> model -> lưu model cho các lần predict

### App

##### Nhận vào 1 ảnh sau đó hệ thống sẽ thực hiện các bước sau:

- input -> Lọc nhiễu -> denoising image
- denoising image -> adjust gamma -> adjusted gamma image
- adjusted gamma image -> contrast stretching -> con_st image
- Show con_st image
- con_st -> edge detection -> show edge image
- con_st image -> face detector -> coordinator list and rect_image
- coordinator list and rect_image -> face recognitor -> image with face detection and name
