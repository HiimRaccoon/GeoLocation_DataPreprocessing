# GeoGuessr-50k country augmentation benchmark

Dự án triển khai benchmark `geoguessr50k_country_augmentation_v1_1` theo đặc tả đã khóa trong `docs/specs.md`. Kế hoạch và các điều kiện nghiệm thu nằm trong `docs/plan.md` và `docs/tasks.md`.

## Cài đặt và kiểm thử

Dùng Python 3.10 trở lên. Cài bản PyTorch/Torchvision tương thích với GPU và CUDA của máy chạy trước, rồi cài package:

```powershell
python -m pip install -e '.[dev]'
python -m geoguessr_benchmark --help
python -m pytest -q
```

Các kiểm thử dùng fixture không yêu cầu dataset benchmark hoặc model thật. Bài kiểm tra JavaScript của gói review cần Node.js; bài kiểm tra trình duyệt cần Playwright Python và Chrome hoặc Edge cài sẵn. Những bài tùy chọn này được bỏ qua nếu thiếu công cụ tương ứng. GPU, driver và các dependencies thực tế được kiểm tra riêng ở bước preflight; môi trường CPU dùng cho fixture không phải môi trường chạy benchmark chính thức.

## Dữ liệu và các bước chạy

Đặt ảnh gốc theo `data/compressed_dataset/<class_name>/*.jpg`. Sau khi có đủ dataset, model access và GPU phù hợp, chạy các lệnh theo thứ tự trong `docs/plan.md`: `inventory`, `split`, `preflight`, `augment`, review offline, `augment-freeze`, `init`, `train-all`, `freeze-checkpoints`, `test-all` và `report`. Mỗi lệnh có thể dùng `--root <project_root>` trước tên lệnh. `verify --stage <stage>` kiểm tra các tệp đã khóa của một stage.

`augment` trả `review_required` khi cần người duyệt diffusion. Dùng `review-export --output augmentation/manifests/review_queue.jsonl` để tạo `augmentation/review/`, chuyển cả thư mục sang máy người duyệt và mở `index.html` trực tiếp. Nhập file kết quả bằng `review-import --input <reviews.jsonl>`, rồi chạy lại `augment`. Khi membership đã khóa, `augment-freeze` chạy DINO diagnostics và kiểm tra membership trước khi xuất union.

Các model snapshots được tải và khóa revision/hash ở preflight. Cấu hình, mã nguồn, môi trường GPU và artifact đã khóa được kiểm tra trước các bước tính toán; nếu không khớp, pipeline dừng để xử lý sự sai lệch thay vì tiếp tục với thiết lập khác.

## Metadata vị trí tùy chọn

Nếu có metadata, đặt `data/location_metadata.csv` với header:

```csv
filepath,panorama_id,latitude,longitude
```

`filepath` là đường dẫn tương đối POSIX giống inventory, ví dụ `data/compressed_dataset/Class/image.jpg`. `panorama_id` có thể để trống; latitude và longitude phải cùng có hoặc cùng để trống. File có thể chỉ bao phủ một phần ảnh; báo cáo ghi rõ coverage. Pipeline kiểm tra các panorama xuất hiện ở nhiều split và báo các cặp khác split trong bán kính **25 m**. Bán kính này là mặc định chẩn đoán của implementation vì đặc tả không quy định con số cụ thể; nó không thay split, xóa ảnh hoặc lọc augmentation. Audit không chứng minh tính độc lập địa lý. Nếu không có file, báo cáo chỉ xác nhận phạm vi kiểm tra trùng ảnh chính xác.

## Phạm vi xác minh

Các bài kiểm thử trong repo kiểm tra logic bằng dữ liệu và model giả lập, bao gồm phục hồi sau gián đoạn. Chúng không thay thế việc kiểm tra dataset thật, preflight GPU, review của con người, sáu run chính thức và final evaluation theo T037–T042.
