# Technical specification: GeoGuessr-50k country classification benchmark

**Phiên bản:** 1.2

**Trạng thái:** **FROZEN**; các cổng thực nghiệm v1.2 cần bằng chứng mới.

**Ngôn ngữ báo cáo:** tiếng Việt; tên trường dữ liệu và artifact dùng tiếng Anh.

### Thay đổi ở v1.2

Benchmark ID là `geoguessr50k_country_augmentation_v1_2`. Cả hai condition dùng cùng ảnh cleaned sau khi che hai HUD cố định. Traditional và SD 1.5 img2img dùng kích thước native; diffusion chỉ thêm reflection padding tối thiểu cho VAE rồi crop bỏ padding. Classifier tiếp tục dùng letterbox ImageNet mean ở 384 × 160. Diffusion giữ FP32 và human review bắt buộc; DINO chỉ ghi chẩn đoán.

## 1. Câu hỏi và phạm vi

Benchmark kiểm tra: **với cùng model, cấu hình training và tập validation/test thật, việc bổ sung ảnh augmented/synthetic cho minority classes có làm thay đổi performance, đặc biệt Macro-F1, hay không?**

Đây là benchmark độc lập, không sử dụng E1/E2/E3 cũ. Chỉ có hai conditions và ba training seeds, tổng cộng sáu runs. Không tìm backbone tốt nhất, không tuning hyperparameter, không chọn cấu hình generation theo validation/test score.

| Condition | Training dataset | Validation/Test |
|---|---|---|
| `original` — Condition A | Cleaned real train | Fixed cleaned real splits |
| `augmented` — Condition B | Cùng cleaned real train + offline traditional + offline diffusion | Cùng các splits của A |

Biến can thiệp duy nhất là **training dataset**. Model, pretrained weights, seed ghép cặp, preprocessing, batch size, loss, optimizer, LR, scheduler theo tiến độ epoch, số epochs, checkpoint policy và metrics giống nhau. B sử dụng đúng một bộ ảnh offline đã khóa cho cả ba seeds.

Trong đặc tả này, “original train” hoặc “ảnh thật gốc” trong các split là ảnh real đã clean, chưa qua augmentation. Raw chỉ là đầu vào của bước `clean`.

Kết quả đo hiệu ứng của toàn bộ chính sách bổ sung dữ liệu. Với cùng 20 epochs, B có nhiều optimizer steps và tổng lượt xem ảnh hơn A; đây là hệ quả của dataset lớn hơn và phải được công bố. Benchmark này không tách riêng hiệu ứng traditional với diffusion, và không phải so sánh equal-compute.

## 2. Dataset và inventory

Raw input chỉ đọc tại `data/compressed_dataset`. Trước inventory, chạy `clean` để xuất JPEG native vào `data/cleaned_dataset` với cùng class directory, thư mục con và filename. Decode và áp EXIF orientation trước khi dùng tọa độ trên RGB. Chỉ che đúng hai bbox half-open bằng RGB `(128,128,128)`:

| Kích thước | HUD MAP/ROUND/SCORE | Minimap |
|---|---|---|
| 1536 × 662 | `[1253,15,1505,73]` | `[1257,448,1505,603]` |
| 1030 × 703 | `[747,15,999,73]` | `[832,554,1000,646]` |

Giữ nguyên dimensions, dùng JPEG RGB quality 95, subsampling 0, bỏ EXIF. Kích thước ngoài bảng làm cleaning thất bại. `reports/cleaning_manifest.json` ghi policy/version, benchmark ID, fill RGB, đường dẫn raw/cleaned, dimensions, bboxes và SHA-256 của file bytes và decoded RGB pixels cho cả hai bản. Inventory xác minh membership, kích thước và các hash cleaned theo manifest. Split chỉ dùng inventory của bản cleaned đó và khóa checksum manifest cùng các artifact split. Validation, test và mọi augmentation source đều xuất phát từ cleaned dataset.

| Thuộc tính | Giá trị đã thống kê |
|---|---:|
| Tổng ảnh real | 49,997 |
| Số classes | 124 |
| Định dạng | JPEG (`.jpg`) |
| Resolution 1536 × 662 | 49,942 ảnh |
| Resolution 1030 × 703 | 55 ảnh |
| Class lớn nhất | United States: 12,014 |
| Số ảnh nhỏ nhất/class | 1; nhiều class cùng có mức này, gồm Antarctica |

Tham chiếu inventory hiện tại: [báo cáo số lượng ảnh](../bao_cao_so_luong_anh_theo_class.md). Các số trên là baseline đầu vào, không phải kết quả thực nghiệm.

Trước split, triển khai phải tạo `reports/dataset_inventory.csv`: `image_id`, `filepath`, `class_name`, `class_id`, `width`, `height`, `bytes`, `sha256`, `decoded_pixel_sha256`, `readable`. Path tương đối so với project root, dấu `/`; đọc ảnh, áp EXIF orientation rồi chuyển RGB trước khi hash pixel. Hash pixel phải bao gồm width, height và RGB bytes để tránh nhập nhằng.

`image_id = SHA256(filepath UTF-8)`. Tạo `class_to_idx.json` bằng cách sắp xếp tên thư mục theo Unicode code point, không phụ thuộc locale/OS; `class_id` liên tiếp từ 0 đến 123. Không xây lại class mapping trong từng DataLoader.

Fail preflight nếu số ảnh/classes không khớp baseline, ảnh không đọc được, hoặc class mapping mâu thuẫn. Không âm thầm bỏ ảnh để tiếp tục benchmark.

Kiểm tra exact duplicates bằng file hash và decoded pixel hash trước khi freeze split. Các ảnh cùng label trùng theo một trong hai hash được gom theo liên kết bắc cầu và xếp cùng split, giữ nguyên quota từng class và giữ đủ mọi ảnh gốc. Dừng chuẩn bị benchmark nếu có label mâu thuẫn, không thể xếp nguyên nhóm trong quota, hoặc audit sau khi xếp vẫn phát hiện trùng giữa các split. Nếu có panorama/location metadata, audit cùng panorama/nearby captures và báo cáo rủi ro. Khi không có metadata, chỉ được khẳng định đã kiểm tra trùng ảnh; không khẳng định độc lập về địa lý.

## 3. Split deterministic, thực hiện trước augmentation

`split_seed = 42`, độc lập training seeds. Chỉ split một lần, tái sử dụng nguyên manifest cho cả sáu runs.

Với mỗi class có `n` ảnh real:

| Original count | Train | Validation | Test |
|---|---:|---:|---:|
| `n >= 10` | `n - 2q` | `q` | `q` |
| `3 <= n <= 9` | `n - 2` | 1 | 1 |
| `n = 2` | 1 | 0 | 1 |
| `n = 1` | 1 | 0 | 0 |

Trong đó `q = max(1, floor(0.1*n + 0.5))`. Đây là default làm tròn cố định, không sử dụng banker's rounding. Mọi class có ít nhất một train image; class `n >= 3` có đủ ba splits.

Thứ tự ảnh trong từng class được xác định bằng khóa `SHA256("split-v1|42|" + filepath)`, sắp xếp tăng dần theo hash; nếu trùng khóa dùng filepath. Phân bổ ban đầu lấy lần lượt `train_count` ảnh đầu, tiếp theo validation, cuối cùng test. Nếu nhóm trùng bị chia, xếp nguyên nhóm theo thứ tự kích thước giảm dần rồi theo vị trí đầu tiên trong thứ tự hash. Ưu tiên split đang có nhiều thành viên nhất của nhóm, dùng thứ tự train/val/test để phá hòa và thử phương án tiếp theo khi hết quota. Ảnh lẻ lấp các vị trí còn lại, ưu tiên giữ split ban đầu rồi dùng thứ tự hash. Cách này cố định qua OS và không phụ thuộc thứ tự quét filesystem hay phiên bản random library.

Ví dụ: `8 -> 6/1/1`, `4 -> 2/1/1`, `3 -> 1/1/1`, `2 -> 1/0/1`, `1 -> 1/0/0`.

Từ bảng counts hiện tại và công thức trên, các tổng **dự kiến** là:

| Split | Ảnh real | Classes có ground-truth support |
|---|---:|---:|
| Train | 39,978 | 124 |
| Validation | 5,007 | 106 |
| Test | 5,012 | 111 |

Đây là phép tính theo counts, chưa phải manifest đã tạo. Implementation phải đối chiếu lại sau inventory và split. Class không có sample thật trong split tương ứng phải ghi `not evaluable on this split`; không tạo synthetic validation/test.

### 3.1. Split artifacts và freeze

```text
splits/
  train.csv
  val.csv
  test.csv
  class_to_idx.json
  class_groups.csv
  split_metadata.json
```

Mỗi CSV có `image_id,filepath,class_name,class_id,split,sha256,decoded_pixel_sha256,is_synthetic`; `is_synthetic=false` cho toàn bộ split gốc. CSV UTF-8, header cố định, sắp xếp `(class_id, filepath)`.

`split_metadata.json` lưu `schema_version`, split seed, thuật toán thứ tự/hash, công thức allocation, dataset fingerprint, per-class/per-split counts, evaluable label IDs, SHA-256 của từng manifest, mapping và group file. Dataset fingerprint là SHA-256 của inventory canonical đã sắp theo filepath, loại timestamp khỏi nội dung hash.

Freeze bằng checksum trong `benchmark_lock.json`. Mỗi run xác minh checksum trước training/evaluation. Không regenerate split theo seed; timestamp chạy lại không được thay đổi sample membership.

### 3.2. Chống leakage

Pipeline bắt buộc: `original inventory -> fixed split -> augment ORIGINAL TRAIN ONLY -> freeze augmented manifest -> six runs`.

Source của mọi synthetic image phải là ảnh real thuộc `splits/train.csv`; không dùng synthetic làm source tiếp, không dùng val/test cho generation, prompt captioning, adaptation, quality threshold selection hoặc xem trước để sửa pipeline. Val/test không được copy vào augmented dataset.

Assert các image IDs và hashes giữa train/val/test không giao nhau. Kiểm tra toàn bộ source IDs của synthetic là tập con original train và không giao val/test. Kiểm tra exact output duplicates với inventory real là phép audit leakage cơ học, không được dùng test để xếp hạng chất lượng hay lựa chọn semantic similarity. Nếu phát hiện va chạm với holdout, dừng artifact freeze và ghi incident, không dùng thông tin đó để tối ưu generation.

## 4. Frequency groups và augmentation budget

Đặt `n_c` là **số ảnh ORIGINAL TRAIN của class c sau split**. Freeze groups một lần trong `class_groups.csv`; không dùng total dataset count hay augmented count để đổi nhóm.

| Group ID | Tên report | Điều kiện | Phương pháp ở B | Tổng target/class |
|---|---|---|---|---:|
| `many` | Many | `n_c >= 100` | Giữ nguyên | `n_c` |
| `medium` | Medium | `20 <= n_c <= 99` | Traditional offline | 100 |
| `tail` | Extreme Tail | `n_c < 20` | Diffusion offline | `min(100, 5*n_c)` |

Số ảnh cần **accepted** là `target_c - n_c`, không phải số lần thử generate. Default với singleton là 5 total = 1 real + 4 synthetic; không phải 100 synthetic. Ví dụ `15 -> 75 total`, `8 -> 40`, `4 -> 20`, `2 -> 10`, `1 -> 5`.

| Group | Classes dự kiến | Original train | Synthetic accepted cần thêm | B total |
|---|---:|---:|---:|---:|
| Many | 48 | 37,662 | 0 | 37,662 |
| Medium | 36 | 2,087 | 1,513 traditional | 3,600 |
| Extreme Tail | 40 | 229 | 916 diffusion | 1,145 |
| **Tổng** | **124** | **39,978** | **2,429** | **42,407** |

`class_groups.csv` lưu class ID/name, original train count, group, target, accepted budget và accepted thực tế. Những số trên là kế hoạch theo counts, chưa phải dữ liệu đã sinh.

## 5. Offline generation chung

`augmentation_seed = 314159`, khác training/split seed. Sinh một lần trước sáu runs. Ảnh real không bị overwrite; chỉ accepted synthetic mới được đưa vào Condition B.

Trong mỗi class, sắp source theo filepath. Attempt thứ `j`, bắt đầu từ 0, chọn `source[j mod n_c]`. Seed candidate là 32 bit đầu của SHA-256 chuỗi `aug-v1|314159|augmentation_type|class_id|j` theo big-endian. Lưu seed và toàn bộ tham số thực tế, không chỉ seed gốc.

Xử lý attempts theo thứ tự tăng dần kể cả khi generation song song. Lấy accepted candidates theo thứ tự attempt đến khi đủ budget. Giới hạn `10 * required_accepted` attempts/class, không thay strength/threshold/prompt để bù quota. Nếu chưa đủ target, ghi shortage và dừng trước main training; không tự đổi phương pháp hay dùng rejected samples. Một thay đổi cấu hình cần phiên bản benchmark mới trước khi chạy lại toàn bộ comparison.

Đầu ra JPEG RGB, quality 95, subsampling 0, không EXIF; decode lại file đã lưu để chạy QC. Các rejected records vẫn giữ metadata và reason. Manifest accepted cùng pixel/file hashes được freeze; không generate lại cho từng training seed.

## 6. Traditional offline augmentation cho Medium

Áp trên cleaned RGB native của split train, giữ width/height. Mỗi candidate chọn đều `k in {1,2}`, rồi chọn không hoàn lại k transform từ bảng, áp theo thứ tự bảng. Không sử dụng một transform nhiều lần trong một candidate.

| Transform | Fixed parameter distribution |
|---|---|
| Brightness | Multiply RGB bởi `Uniform(0.90, 1.10)` |
| Contrast | Blend quanh mean grayscale ảnh với factor `Uniform(0.90, 1.10)` |
| Gamma | Với RGB trong `[0,1]`, dùng `x**gamma`, `gamma ~ Uniform(0.90,1.10)` |
| Gaussian noise | Independent RGB noise, mean 0, `sigma ~ Uniform(1,3)/255` |
| Gaussian blur | Sigma `Uniform(0.10,0.60)` pixel, kernel 5 × 5 |
| Small rotation | `Uniform(-3,+3)` degrees, bilinear, reflection padding |
| JPEG variation | Encode/decode intermediate quality là integer đều trong `[85,95]` |

Clip RGB về `[0,1]` sau từng phép, chỉ quantize khi cần encode. Freeze image backend và rounding trong environment lock. Manifest lưu transform order và sampled values. Output JPEG cuối vẫn theo chính sách chung.

Không dùng flip ngang/dọc, rotation lớn, hue shift mạnh, aggressive crop, affine/perspective mạnh hoặc biến đổi làm đảo traffic side/mất geographic cues. Các biến đổi crop/affine/perspective tùy chọn trong đề bài **không được bật trong default này**, để tránh thêm quyết định implementation và search space.

## 7. Diffusion offline cho Extreme Tail

### 7.1. Biến thể sử dụng

Default là **DA-Fusion-inspired Stable Diffusion image-to-image**, trường manifest `augmentation_type=da_fusion`, `method_variant=sd15_img2img_fixed`. Đề bài cho phép hướng DA-Fusion / Stable Diffusion img2img; ở đây khóa biến thể img2img có conditioning theo class và source. Không tuyên bố tái lập nguyên bản toàn bộ DA-Fusion paper, không thêm textual inversion, LoRA hay fine-tuning diffusion trong default. Nếu sau này yêu cầu tái lập paper, đó là thay đổi đặc tả cần được ghi nhận trước benchmark. Tham chiếu phương pháp gốc: [DA-Fusion](https://arxiv.org/abs/2302.07944).

Luồng: `cleaned real train JPEG → native RGB → minimal reflection pad /8 → VAE encode → img2img → VAE decode → crop theo source_bbox → native JPEG content-only → QC/hash audit → human geographic review`. Resize cho classifier chỉ diễn ra tại train/eval. DINO similarity được ghi riêng cho chẩn đoán.

### 7.2. Fixed generation configuration

```yaml
augmentation_seed: 314159
method_variant: sd15_img2img_fixed
diffusion_model: stable-diffusion-v1-5/stable-diffusion-v1-5
pipeline: StableDiffusionImg2ImgPipeline
scheduler: DDIMScheduler
scheduler_source: diffusion_model_scheduler_config
eta: 0.0
strength: 0.35
guidance_scale: 5.0
num_inference_steps: 40
num_images_per_prompt: 1
generation_batch_size: 1
generation_resolution_policy: native_pad_to_multiple
generation_multiple: 8
generation_padding_mode: reflect
generation_downsample: false
precision: float32
attention_backend: standard
xformers: false
textual_inversion: false
lora: false
diffusion_finetuning: false
output_format: jpeg
output_quality: 95
output_subsampling: 0
```

Model lấy từ [SD 1.5 model repository](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5). Trước generation phải resolve repository revision thành immutable commit SHA, hash weights, tokenizer, VAE và scheduler config; lưu vào lock. Không tải nhánh mutable trong lúc benchmark. `num_inference_steps=40` là tham số pipeline yêu cầu, không phải cam kết 40 denoising iterations thực tế ở strength 0.35; log cả timestep schedule thực tế.

Generation giữ **FP32**, standard attention và `xformers=false`. GPU preflight chạy model thật với batch 1 trên cleaned train source có padded canvas lớn nhất. CUDA OOM làm preflight thất bại và báo geometry; không hạ resolution, đổi precision hoặc đổi backend.

Với cleaned RGB có kích thước `W × H`, canvas là `(ceil(W/8)*8, ceil(H/8)*8)`. Giữ nguyên pixel vùng source. Chia phần thiếu để center reflection-pad không lặp pixel biên; left/top lấy floor, phần lẻ ở right/bottom. Lưu `source_bbox=[left,top,left+W,top+H]` cùng dimensions và bốn offsets. Kiểm tra VAE decode trả đúng canvas rồi crop một lần theo bbox, lưu JPEG content-only cùng kích thước native của source.

| Cleaned source | SD canvas | `source_bbox` | JPEG sau crop |
|---|---|---|---|
| 1536 × 662 | 1536 × 664 | `[0,1,1536,663]` | 1536 × 662 |
| 1030 × 703 | 1032 × 704 | `[1,0,1031,703]` | 1030 × 703 |

Bbox dùng tọa độ canvas trước crop. QC, review, DINO và classifier đọc JPEG đã crop từ tọa độ `(0,0)`.

Prompt cố định theo template, chỉ thay `{class_name}` bằng tên thư mục:

```text
A realistic street-level road photograph in {class_name}, Google Street View style, preserve the original road layout, traffic side, lane markings, existing signs and visible writing, buildings, utility poles, vegetation and vehicle configuration, natural lighting, subtle realistic variation.
```

Negative prompt cố định:

```text
illustration, painting, cartoon, flipped road, reversed traffic, altered writing, invented road signs, distorted geometry, impossible objects, deformed vehicles, duplicated objects, severe blur, compression blocks, artificial textures, new watermark
```

Prompt không đảm bảo label preservation. Không dùng quốc kỳ, text tên nước hoặc cue nhân tạo để mã hóa label. Không tự viết thêm prompt theo ảnh sau khi xem evaluation. Giữ safety checker theo model snapshot; flagged outputs bị reject và ghi reason, không thay pipeline để lấy lại ảnh.

### 7.3. Geographic cues và human review

Mọi diffusion candidate vượt automatic QC phải được người review bằng cách so sánh **source train image với generated content-only file cuối sau crop**, trước training. Review không truy cập val/test hoặc classifier scores. Kiểm tra road geometry, traffic side, lane markings, signs/text, utility poles, architecture, vegetation, vehicle configuration, Street View appearance và cues đặc trưng quốc gia còn nhất quán.

Reject nếu có hallucinated signs, chữ/ngôn ngữ bị đổi, đảo traffic side, road structure vô lý, kiến trúc thay đổi quá mạnh, object phi thực tế hoặc synthetic texture rõ. Với source có text đọc được, text bị sửa/mất mà không xác minh được là lý do reject. Nếu khó xác định label preservation thì `reject_ambiguous`, không mặc định accept. Reviewer ID, UTC timestamp, rubric version, quyết định và reason phải được ghi. Review toàn bộ diffusion outputs được chấp nhận; training chỉ bắt đầu sau khi review kết thúc và manifest freeze.

## 8. Quality-control stage

QC chạy trên file output cuối, trước khi synthetic tham gia train. **Traditional: automatic QC only. Diffusion: automatic basic QC + hash duplicate audit + DINO diagnostic logging + mandatory human geographic-cue review.** DINO không quyết định acceptance/rejection; human review quyết định geographic consistency cho diffusion. Các ngưỡng blank/blur cố định dưới đây không được hiệu chỉnh theo val/test.

### 8.1. Basic và visual checks

1. Decode JPEG thành RGB hợp lệ, width/height dương, không NaN/Inf. Traditional phải giữ đúng dimensions source. Với diffusion, xác minh metadata canvas, bbox và offsets khớp với native padding theo §7.2; final RGB JPEG phải đúng kích thước native của cleaned source. Không dùng canvas dimensions làm yêu cầu kích thước final file, không áp minimum short edge lớn hơn content height.
2. Trên grayscale uint8 vùng ảnh thật, standard deviation phải `>= 5` và tỷ lệ pixel có intensity `<=2` hoặc `>=253` phải `< 0.98`; nếu không reject blank/saturation.
3. Kiểm tra source ID/hash thuộc ORIGINAL TRAIN, label không đổi và transform provenance khớp fixed policy. Traditional chỉ cần automatic checks về decode/RGB, dimensions, non-blank, exact duplicates và provenance; không yêu cầu reviewer xem từng ảnh hoặc review toàn bộ contact sheets.
4. Với **diffusion**, giữ blur check: lấy toàn bộ decoded content-only file và cleaned source ở native dimensions, rồi resize giữ aspect ratio về long edge 384. Không crop lại file theo `source_bbox`. Dùng grayscale `[0,255]`, Laplacian kernel `[[0,1,0],[1,-4,1],[0,1,0]]`, reflect boundary. Yêu cầu `variance(output) >= max(5, 0.5*variance(source))`. Không dùng generation padding làm vùng tính blur.
5. Với **diffusion**, reject severe distortion/visual generation failure, artifacts hoặc geographic inconsistency theo mandatory human review ở mục 7.3. Nếu ambiguous dùng `reject_ambiguous`. Không dựa chỉ vào blur metric để kết luận ảnh đúng label. Traditional không có human-review gate.

### 8.2. Exact duplicates và DINO diagnostics

Mọi output phải khác decoded-pixel hash của source, mọi original train image và synthetic đã accepted. Không dùng file-byte khác nhau do JPEG metadata làm bằng chứng diversity.

Với diffusion, dùng frozen `dino_vits16` từ [DINO official repository](https://github.com/facebookresearch/dino), pin code commit và pretrained weight SHA-256. Không train encoder này. Input diagnostics là cleaned source ở native dimensions và toàn bộ decoded generated content-only file đã crop theo mục 7.2; không crop lại bằng tọa độ canvas và không dùng generated reflection padding. Mỗi ảnh được letterbox toàn nội dung về 224 × 224, RGB padding ImageNet mean, bilinear antialias, ImageNet normalization; lấy global embedding và L2 normalize. Resolution 224 × 224 này chỉ thuộc encoder diagnostics, không thay generation canvas hoặc classifier input 384 × 160.

Log `dino_source_similarity` là cosine giữa source/output, và `dino_nearest_sibling_similarity` là cosine cao nhất với diffusion output khác cùng source trong tập accepted cuối cùng, không so ảnh với chính nó. Lưu cả ID sibling gần nhất; nếu không có sibling hợp lệ thì ghi `null` và lý do. Ghi diagnostics cho accepted và rejected candidates đọc được; lỗi tính embedding ghi `null` cùng diagnostic error.

**DINO similarity MUST NOT determine synthetic image acceptance/rejection.** Hoàn tất quyết định QC/human review độc lập với DINO; reviewer không dùng DINO score để chọn ảnh. Tính nearest-sibling diagnostics sau khi membership accepted đã khóa, không chọn lại samples theo scores. Không có hard similarity threshold, không tuning/search space; DINO lỗi hoặc score thiếu không làm sample bị reject. Các scores chỉ phục vụ audit/phân tích sau experiment vì chưa calibrated trên GeoGuessr-50k và benchmark không thêm phương pháp DINO-based sample selection. Lưu blur scores của diffusion riêng; Traditional không cần DINO diagnostics hoặc human review.

Mọi failure phải có reason enum, ví dụ `unreadable`, `invalid_rgb`, `invalid_dimensions`, `blank`, `invalid_source`, `invalid_provenance`, `blur`, `exact_duplicate`, `geographic_cue_changed`, `generation_artifact`, `safety_flag`, `reject_ambiguous`. `blur` và các human-review reasons chỉ áp dụng diffusion. Không có rejection reason dựa trên DINO similarity; không dùng classifier validation/test score làm bộ lọc.

## 9. Augmentation manifest và provenance

`augmentation/manifests/candidates.jsonl` chứa `benchmark_id` và provenance cho mọi attempt; `accepted.jsonl` chỉ chứa final accepted. Nested fields lưu JSON, không serialize Python repr.

Trường bắt buộc chung:

```text
synthetic_id, synthetic_path, source_image_id, source_image_path,
source_sha256, class_name, class_id, source_split,
augmentation_type, method_variant, augmentation_parameters,
augmentation_seed, generation_seed, attempt_index,
generation_timestamp, filter_status, filter_reason,
accepted, rejection_reason, qc_scores, reviewer_id, review_timestamp,
dino_source_similarity, dino_nearest_sibling_similarity,
dino_nearest_sibling_id, dino_diagnostic_error,
output_width, output_height, output_sha256, output_pixel_sha256,
config_sha256, environment_lock_sha256
```

Trường diffusion bổ sung: `diffusion_model`, `model_revision`, `weights_sha256`, `prompt`, `negative_prompt`, `strength`, `guidance_scale`, `num_inference_steps`, actual timesteps, scheduler config, `generation_canvas_width`, `generation_canvas_height`, `precision=float32`, `source_bbox`, `source_content_width`, `source_content_height`, padding mode/offsets theo mục 7.2, `output_content_only=true`. `source_bbox` là tọa độ half-open trên canvas trước crop và bắt buộc giữ trong manifest. `synthetic_path` trỏ tới content-only JPEG; `output_width`/`output_height` bằng bbox dimensions, còn output hashes tính trên final file/pixels sau crop và JPEG decode. Không dùng generation canvas width/height thay final-file dimensions. Hai similarity fields và sibling ID chỉ là diagnostic metadata, không tham gia acceptance. Traditional ghi `null` cho DINO, reviewer fields và các trường diffusion không áp dụng; mọi sampled transform lưu trong `augmentation_parameters`.

`filter_status` là `pending`, `accepted` hoặc `rejected`; chỉ `accepted=true` và `filter_status=accepted` được đưa vào `splits/train_augmented.csv`. Manifest này là union original train và accepted outputs, không lặp original record, không chứa rejected/pending. Mọi source phải có `source_split=train` và provenance một bước về real image.

Preflight và generation ghi scheduler config dưới dạng JSON chuẩn hóa. Danh sách tên `_use_default_values` được sắp xếp vì thứ tự của nó không mang ý nghĩa. Mọi giá trị cấu hình và thứ tự các danh sách còn lại đều được đối chiếu với preflight. Khác biệt có ý nghĩa hoặc `actual_timesteps` sai giá trị hay thứ tự làm provenance không hợp lệ.

V1.1 không tương thích với v1.2. Giữ khóa, manifests, review và checkpoint cũ ở run root riêng. Với v1.2, bắt đầu từ cùng raw JPEG trong run root mới, chạy `clean` → `inventory` → `split` → `verify --stage splits` → GPU `preflight` → `augment`. Tạo lại candidates, review, membership và train union theo v1.2. GPU preflight khóa lại `code_sha256`.

## 10. Model và classifier preprocessing

Chỉ dùng `torchvision.models.efficientnet_b0`, weights `EfficientNet_B0_Weights.IMAGENET1K_V1`; thay lớp linear cuối thành `1280 -> 124`, giữ dropout mặc định 0.2, fine-tune toàn bộ backbone, BatchNorm ở train mode khi train. Không thay architecture giữa conditions. Xem [Torchvision EfficientNet-B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html).

Với mỗi seed, tạo và hash một initial state dict gồm pretrained backbone cùng initialized 124-class head; cả A và B nạp đúng state dict đó. Default head: Kaiming uniform theo PyTorch Linear với `a=sqrt(5)`, bias uniform `±1/sqrt(1280)`. Training seed control cả initialization; không reuse best checkpoint của condition trước.

Classifier preprocessing giống hệt ở train/val/test cho real, traditional và diffusion:

1. Decode, EXIF transpose, convert RGB, chuyển float `[0,1]`.
2. `scale = min(384/W, 160/H)`; `new_W = min(384,max(1,floor(W*scale+0.5)))`, tương tự `new_H` với 160.
3. Bilinear resize có antialias; không crop. Center pad lên width 384, height 160; left/top lấy floor nửa phần dư, phần còn lại đặt right/bottom.
4. Padding RGB float bằng ImageNet mean `(0.485,0.456,0.406)`; sau normalize pad bằng zero.
5. Normalize mean trên, std `(0.229,0.224,0.225)`, tensor `float32`, shape **`3 × 160 × 384`**.

Với 1536 × 662, vùng resize là 371 × 160; với 1030 × 703 là 234 × 160 theo quy tắc trên. Không direct-resize toàn ảnh thành 384 × 160, không dùng default ImageNet center crop của weights transform. Không có random online augmentation, kể cả flip/crop/color jitter. Dropout của model vẫn là thành phần model giống nhau trong hai conditions.

Diffusion synthetic JPEG sau crop và traditional JPEG có đúng native dimensions của cleaned source. Classifier decode ảnh native, resize giữ aspect ratio rồi center pad bằng ImageNet mean thành 384 × 160.

## 11. Fixed training configuration

```yaml
benchmark_id: geoguessr50k_country_augmentation_v1_2
conditions: [original, augmented]
model: efficientnet_b0
weights: IMAGENET1K_V1
num_classes: 124
input_width: 384
input_height: 160
optimizer: AdamW
learning_rate: 0.0003
weight_decay: 0.0001
betas: [0.9, 0.999]
eps: 0.00000001
epochs: 20
scheduler: linear_warmup_then_cosine
warmup_epochs: 1
warmup_start_factor: 0.1
minimum_learning_rate: 0.000001
loss: CrossEntropyLoss
loss_reduction: mean
label_smoothing: 0.0
class_weights: null
batch_size_preferred: 128
batch_size_fallback_order: [64, 32, 16, 8, 4, 2]
gradient_accumulation_steps: 1
gradient_clipping: null
precision: float32
amp: false
shuffle_train: true
replacement_sampling: false
drop_last: false
num_workers: 4
pin_memory: true
persistent_workers: false
split_seed: 42
augmentation_seed: 314159
training_seeds: [42, 123, 2026]
early_stopping: false
checkpoint_metric: val_macro_f1_evaluable
checkpoint_mode: max
checkpoint_tie_break: earliest_epoch
```

AdamW weight decay áp cho toàn bộ trainable parameters, không thêm class-specific parameter groups. Mỗi epoch đi qua từng record đúng một lần bằng random permutation không replacement; ở B, các synthetic files là các records riêng. Không weighted/focal/class-balanced loss, weighted/balanced sampler, oversampling, MixUp, CutMix hoặc label smoothing.

### 11.1. Scheduler và batch size

Scheduler update theo optimizer step với tiến độ epoch, không theo validation score. Với condition có `L=ceil(train_size/batch_size)` steps/epoch, optimizer step toàn cục `k=0..20L-1` dùng `u=k/L`:

```text
0 <= u < 1: lr = base_lr * (0.1 + 0.9*u)
1 <= u < 20: lr = min_lr + 0.5*(base_lr-min_lr)*(1+cos(pi*(u-1)/19))
```

Định nghĩa này áp LR trước optimizer step; log LR mỗi step/epoch. Số step khác nhau vì dataset size, nhưng công thức và lịch theo epoch giống nhau.

Chọn batch size bằng memory-only preflight trước main experiment: thử 128 rồi lần lượt các fallback chỉ khi OOM, dùng tensor shape thật, model/optimizer/precision thật, không dùng val/test score. Thử cả forward/backward/AdamW state allocation; reset weights, optimizer, RNG sau preflight. Khóa batch size đã chọn trong config và dùng cho sáu runs, kể cả last partial batch policy. Nếu batch 2 vẫn OOM, dừng vì thiếu tài nguyên. Không tự giảm batch khi một main run OOM; khắc phục hạ tầng hoặc khởi tạo benchmark version mới cho cả sáu runs.

### 11.2. Seeds và reproducibility

Set Python `random`, NumPy, PyTorch CPU/CUDA, DataLoader generator và worker seed. Khởi động interpreter với `PYTHONHASHSEED` bằng training seed; worker seed lấy từ DataLoader initial seed modulo `2**32`. Mỗi run bắt đầu từ RNG sạch; evaluation không tiêu thụ RNG của training sampler.

Single GPU cho cả sáu runs; cùng GPU model, driver và software lock. Default `torch.use_deterministic_algorithms(true)`, cuDNN benchmark off, cuDNN deterministic on, CUDA matmul/cuDNN TF32 off, `CUBLAS_WORKSPACE_CONFIG=:4096:8` trước CUDA initialization. Nếu operation không hỗ trợ deterministic ở môi trường đã chọn, dừng preflight thay vì âm thầm tắt. Không hứa bitwise reproducibility trên GPU/library versions khác nhau.

Trước main experiment, lưu `environment.lock.txt`, Python/package versions có hashes nếu khả dụng, OS, GPU, CUDA/cuDNN, driver, code commit hoặc source archive hash, mọi pretrained weight SHA và resolved model/code commits. Những giá trị phụ thuộc máy này được materialize ở preflight, không phải tuning parameters; không để revision mutable hoặc lock field chưa resolve khi bắt đầu generation/training.

## 12. Run matrix và checkpoint policy

| Run | Condition | Training seed | Train manifest |
|---|---|---:|---|
| 1 | original | 42 | train.csv |
| 2 | augmented | 42 | train_augmented.csv |
| 3 | original | 123 | train.csv |
| 4 | augmented | 123 | train_augmented.csv |
| 5 | original | 2026 | train.csv |
| 6 | augmented | 2026 | train_augmented.csv |

Train đủ 20 epochs, evaluate validation cuối mỗi epoch bằng `model.eval()` và no-grad/inference mode. Validation chỉ monitor và chọn checkpoint, không thay LR/config, không early stop. Chọn epoch có **Validation Macro-F1 over evaluable classes** lớn nhất; hòa đúng giá trị chưa làm tròn thì lấy epoch sớm nhất. Không lấy checkpoint theo test hoặc train score.

Mỗi run lưu `best_checkpoint.pt` và `last_checkpoint.pt` kèm epoch, model/optimizer/scheduler state, RNG states, config/manifests/initial-state hashes. Resume chỉ từ checkpoint tương thích, không thay dữ liệu/config; ưu tiên resume ở epoch boundary, không xáo lại split. Run lỗi được retry đúng run/config, ghi lịch sử lỗi; không chọn seed tốt nhất.

Sau khi sáu runs hoàn tất và best checkpoints được khóa, mới mở final test evaluation cho từng best checkpoint, mỗi checkpoint một lần. Không test last checkpoint để chọn lại. Lưu predictions để tính lại báo cáo từ artifact, không cần inference lại; lỗi hạ tầng khi evaluation phải ghi log và không được thay checkpoint/config.

## 13. Evaluation metrics và label space

Output/logits luôn có 124 classes. Top-1/Top-5 xét toàn bộ label space này; không mask classes không xuất hiện trong evaluation split.

Với split `S`, định nghĩa `E_S = {c | support_S(c) > 0}` từ manifest thật và freeze trước training. Validation dự kiến `|E_val|=106`, test `|E_test|=111`.

Tạo confusion matrix 124 × 124 từ toàn bộ samples split, row=true, column=predicted. Với mỗi class:

```text
TP_c = C[c,c]
FN_c = sum_j C[c,j] - TP_c
FP_c = sum_i C[i,c] - TP_c
precision_c = TP_c/(TP_c+FP_c), hoặc 0 nếu denominator=0
recall_c = TP_c/(TP_c+FN_c), hoặc 0 nếu denominator=0
F1_c = 2*TP_c/(2*TP_c+FP_c+FN_c), hoặc 0 nếu denominator=0
```

Metrics:

- **Primary:** `Macro-F1 = mean(F1_c for c in E_S)`.
- Macro Precision/Recall: trung bình precision/recall trên chính `E_S`.
- Weighted-F1: `sum(support_c*F1_c)/N` trên `E_S`.
- Top-1 Accuracy: tỷ lệ prediction cao nhất đúng label trên toàn split.
- Top-5 Accuracy: ground truth nằm trong năm logits cao nhất trên 124 classes; hòa logits thì class ID nhỏ hơn đứng trước, quy tắc giống nhau mọi run.

Các metrics lưu số thực `[0,1]` đầy đủ precision, không làm tròn trước checkpoint selection/aggregation. Trong bảng cuối có thể nhân 100 và làm tròn 2 decimals; delta báo **percentage points (pp)**, không nhầm relative percentage.

Prediction rơi vào class không có support vẫn làm sample sai và tạo FN cho true class; không bỏ sample. Không đưa class support=0 vào primary macro denominator chỉ vì model dự đoán class đó. Per-class CSV vẫn đủ 124 rows: class không evaluable có metric `null`/blank, `support=0`, `evaluable=false`, status `not evaluable on this split`; JSON không dùng NaN.

Report bắt buộc có `num_classifier_classes=124`, `num_val_evaluable_classes`, `num_test_evaluable_classes`, danh sách class IDs được macro-average và support từng class. Không gọi kết quả này là Macro-F1 trên đủ 124 classes có test support.

## 14. Group-wise evaluation

Groups dùng original TRAIN counts đã freeze. Với group `G`, metric chỉ average trên `G intersect E_S`; không đổi group sau augmentation.

Tính `group_macro_f1` và `group_macro_recall` từ **các per-class counts của confusion matrix toàn split**, rồi average trên group. FP đến từ class ngoài group vẫn tính; prediction ngoài group vẫn tạo FN. Không giới hạn logits hay xóa samples để làm đẹp group metrics.

Mean per-class accuracy nếu dùng được định nghĩa bằng recall class trong single-label classification, nên bằng group macro recall; không trình bày như một bằng chứng độc lập. Mỗi group report số evaluable classes và tổng sample support. Empty intersection ghi `null`, không ghi 0 hoặc đưa vào aggregation.

| Group | Val evaluable classes dự kiến | Test evaluable classes dự kiến |
|---|---:|---:|
| Many | 48 | 48 |
| Medium | 36 | 36 |
| Extreme Tail | 22 | 27 |

## 15. Aggregation và final comparison

Mọi major metric cho mỗi condition phải report đủ ba seeds bằng mean và **sample standard deviation (`ddof=1`)**. Không report một seed đại diện làm kết quả cuối. Lưu bảng raw per-seed trong appendix.

```text
mean = sum(metric_seed)/3
std = sqrt(sum((metric_seed-mean)^2)/(3-1))
paired_delta_seed = metric_augmented_seed - metric_original_seed
Delta Macro-F1 = mean(paired_delta_seed)
```

Report cả mean ± std của paired delta. Độ lệch chuẩn qua training seeds không đo toàn bộ uncertainty do fixed split hoặc few-shot test support. Với ba seeds, không mặc định kết luận statistical significance hay suy ra generalization trên mọi split; benchmark không yêu cầu p-value.

Bảng chính trên test của best validation checkpoints:

| Condition | Top-1 | Top-5 | Macro-F1 (primary) | Weighted-F1 | Macro Precision | Macro Recall |
|---|---:|---:|---:|---:|---:|---:|
| Original | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std |
| Augmented | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std |
| Paired delta (pp) | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std |

| Group | Evaluable classes / test support | Original Macro-F1 | Augmented Macro-F1 | Paired delta (pp) | Original/augmented Macro Recall |
|---|---|---|---|---|---|
| Many (`>=100`) | từ manifest | mean ± std | mean ± std | mean ± std | mean ± std mỗi condition |
| Medium (`20–99`) | từ manifest | mean ± std | mean ± std | mean ± std | mean ± std mỗi condition |
| Extreme Tail (`<20`) | từ manifest | mean ± std | mean ± std | mean ± std | mean ± std mỗi condition |

Thêm per-class F1/recall/support và paired changes, số generation attempts/accepted/rejected theo reason, target coverage, epoch/step counts, tổng training duration. Top-1 gần như không đổi nhưng Macro-F1 và tail recall/F1 tăng vẫn là kết quả có ý nghĩa đối với câu hỏi benchmark. Kết quả giảm hoặc không đổi phải được báo cáo đầy đủ; không retry seeds hay sửa generator sau khi biết kết quả.

## 16. Directory structure và artifacts

```text
project/
  docs/specs.md
  configs/benchmark.yaml
  benchmark_lock.json
  environment.lock.txt
  data/
    compressed_dataset/                  # raw, chỉ đọc ở bước clean
    cleaned_dataset/<class_name>/       # cleaned real JPEG cho cả ba split
    augmented_dataset/
      traditional/<class_name>/
      da_fusion/<class_name>/
  splits/
    train.csv
    val.csv
    test.csv
    train_augmented.csv
    class_to_idx.json
    class_groups.csv
    split_metadata.json
  augmentation/
    traditional/config.yaml
    da_fusion/config.yaml
    manifests/candidates.jsonl
    manifests/accepted.jsonl
    manifests/qc_summary.json
  initializations/seed_<seed>.pt
  runs/
    original/seed_42/
    original/seed_123/
    original/seed_2026/
    augmented/seed_42/
    augmented/seed_123/
    augmented/seed_2026/
  reports/
    cleaning_manifest.json
    dataset_inventory.csv
    split_audit.json
    per_seed_metrics.csv
    aggregate_metrics.json
    group_comparison.csv
    final_report.md
```

Mỗi run directory tối thiểu:

```text
config.yaml
metrics.json
history.csv
best_checkpoint.pt
last_checkpoint.pt
confusion_matrix_val.csv
confusion_matrix_test.csv
per_class_metrics_val.csv
per_class_metrics_test.csv
group_metrics.json
predictions_test.csv
run_metadata.json
```

`history.csv`: epoch, train loss (sample-weighted mean), val loss, val metrics, LR, steps, duration. `predictions_test.csv`: image ID, true/predicted class ID, top-5 IDs và scores. Confusion matrices giữ đủ 124 × 124 và class mapping trong metadata.

`run_metadata.json` gồm condition, seed, code Git commit nếu có hoặc source hash, start/end UTC timestamp, duration, GPU/driver/library information, batch size, train/val/test counts, original/synthetic counts, evaluable class counts, epoch/optimizer-step counts, selected epoch, initial state hash, best checkpoint hash, manifest/config/environment hashes và resume/failure history. `metrics.json` phân biệt rõ validation-at-best với final-test; không ghi đè lịch sử training bằng test metrics.

## 17. Implementation contract và acceptance criteria

Thứ tự triển khai sau khi đặc tả được chấp nhận:

1. Chạy `clean`, xác minh cleaning manifest, rồi inventory cleaned, mapping, duplicate audit và baseline.
2. Tạo deterministic original split, group assignments, thống kê evaluable classes; khóa manifests.
3. Resolve/pin environment và model artifacts; materialize fixed configs, memory-only batch preflight.
4. Generate offline đúng source train; traditional chạy automatic QC, diffusion chạy automatic QC/hash audit và mandatory human geographic review. Khóa accepted membership độc lập DINO, log DINO diagnostics, freeze dataset và provenance. Fail nếu thiếu quota hoặc QC/leakage audit chưa pass.
5. Tạo initial states ghép cặp, xác minh mọi invariant, chạy đúng sáu training runs với cùng config chung.
6. Khóa best checkpoints theo validation, final test evaluation, aggregate/report.

Các checks implementation bắt buộc trước main training: split allocation ở các biên `n=1,2,3,9,10`; disjoint IDs/hashes; target/group boundaries `n_train=19,20,99,100`; synthetic-source membership; same per-seed initial state; letterbox geometry/RGB normalization; metric handling class support=0 và prediction ngoài group; aggregation ddof=1 và paired delta. Checks dùng tiny fixtures, không dùng final test performance để debug/tuning.

Một run hợp lệ khi shared config hash giống năm runs còn lại sau khi loại các trường được phép khác `condition`, training seed, paths, train manifest hash, dataset counts và runtime metadata. Cùng seed phải có cùng initial-state hash. Split/class/group/preprocessing/checkpoint policy hashes phải giống cả sáu runs. Không có accepted synthetic ở val/test và không có val/test source trong provenance.

## 18. Methodological risks và cách diễn giải

1. **Extreme-tail thiếu real evaluation samples:** singleton không có val/test; hai ảnh chỉ có train/test. Tail score chỉ mô tả subset evaluable và không chứng minh hiệu quả trên các class bị thiếu.
2. **Support rất nhỏ:** một prediction đổi có thể làm per-class F1 thay đổi mạnh. Luôn công bố support và per-seed metrics; không xem tăng F1 của class một sample là bằng chứng chắc chắn.
3. **Label corruption/hallucination:** diffusion có thể đổi signs, language, traffic side hay architecture; prompt và DINO diagnostics không bảo đảm label. Diffusion bắt buộc review source/output, lưu rejection reasons; traditional chỉ automatic QC theo policy conservative.
4. **Near-copy và diversity hạn chế:** nhiều descendants từ một source không thay thế nhiều cảnh thật độc lập. Hash reject exact duplicates; DINO chỉ log diagnostics, không loại near-copy theo similarity. Giới hạn 5× giảm nguy cơ phóng đại dữ liệu singleton.
5. **Domain gap:** diffusion texture, generation resolution hoặc JPEG artifacts có thể tạo shortcut và giảm real-test performance. Report cả deterioration, acceptance rate và model/generation parameters.
6. **Macro-F1 có giới hạn phạm vi:** val và test có evaluable label sets khác nhau; không so trực tiếp hai scores như cùng population. Model vẫn dự đoán 124 classes.
7. **Leakage:** source descendants, exact duplicates, panorama gần nhau hoặc external pretrained contamination có thể làm kết quả lạc quan. Audit provenance/hash; ghi rõ spatial independence và pretrained data overlap chưa xác minh nếu không có metadata.
8. **Exposure/compute khác nhau:** 20 epochs trên B dẫn đến thêm steps và thêm lượt xem ảnh. Kết luận thuộc chính sách thêm dữ liệu ở fixed epochs; không tách augmentation content khỏi tổng optimization exposure.
9. **Combined intervention:** hai conditions không thể quy improvement riêng cho traditional hay diffusion. Group results hỗ trợ mô tả, không phải ablation nhân quả từng phương pháp.
10. **Automatic QC và người review diffusion:** blank/blur checks có thể reject cảnh hợp lệ hoặc giữ cảnh sai; diffusion reviewer judgment không hoàn toàn tự động reproducible. Traditional không có human-review gate. Freeze accepted files, diagnostic scores và diffusion review logs; không chỉnh automatic thresholds theo evaluation và không dùng DINO để lựa chọn samples.
11. **Ba seeds và một split:** std chỉ phản ánh variation qua training seeds trên split này. Không diễn giải như uncertainty qua locations, data collection hoặc tất cả possible splits.

## 19. Fixed decisions bổ sung từ đặc tả

Các defaults cần thiết nhưng đề bài chưa chốt hoàn toàn: round-half-up 10% cho split; hash-based deterministic ordering; singleton target 5 total; augmentation seed 314159; transform distributions và k=1–2; SD 1.5 img2img variant, DDIM/strength/guidance/steps; generation native với reflection padding tối thiểu đến bội số 8, crop theo `source_bbox` và lưu generated content-only; JPEG encoding; automatic QC với blank/blur checks đúng phạm vi; DINO diagnostic logging không có acceptance threshold; human review chỉ diffusion; bounded retry/shortage policy; classifier RGB padding mean; AdamW betas/eps/decay policy; warmup/cosine step formula; FP32; batch fallback memory-only; deterministic execution; checkpoint tie-break; evaluable-label macro denominator; group metrics từ full confusion matrix; sample std ddof=1 và paired deltas.

Tất cả là fixed configuration, không phải search space. Exact environment versions/revisions, batch size được chọn bằng memory-only preflight và fingerprints phải được ghi vào lock trước main experiment. Không có kết quả experiment nào được giả định trong tài liệu này.

## 20. Final checklist

- [ ] Original dataset verified
- [ ] Fixed split created before augmentation
- [ ] Split manifest frozen
- [ ] Validation contains only real images
- [ ] Two fixed HUDs masked; raw files preserved; cleaning manifest verified
- [ ] Val/test contain only cleaned real images
- [ ] No val/test image used as augmentation source
- [ ] No real/synthetic descendant leakage detected
- [ ] Group assignments based on original train counts
- [ ] `>=100` classes unchanged
- [ ] `20–99` classes use traditional offline augmentation
- [ ] `<20` classes use DA-Fusion-inspired fixed img2img variant
- [ ] Synthetic images pass applicable automatic QC and hash duplicate audit
- [ ] Native diffusion canvas padded to multiples of 8; JPEG cropped to cleaned source dimensions
- [ ] Diffusion output cropped to source_bbox before saving content-only JPEG
- [ ] Final synthetic dimensions equal source_bbox dimensions; source_bbox retained in manifest
- [ ] QC, hashes, DINO diagnostics and diffusion human review use final content-only files
- [ ] Classifier letterboxes cropped diffusion content with ImageNet-mean padding
- [ ] Diffusion generation uses FP32
- [ ] Classifier input fixed at 384×160
- [ ] DINO similarity logged for diagnostics only
- [ ] No DINO hard acceptance threshold
- [ ] Traditional augmentation uses automatic QC only
- [ ] Diffusion outputs receive human geographic-cue review
- [ ] Targets met; accepted synthetic files and provenance frozen
- [ ] EfficientNet-B0 fixed
- [ ] Same ImageNet pretrained weights and paired initial states
- [ ] Input fixed at width 384 × height 160 using letterbox
- [ ] Same training config for both conditions
- [ ] Same batch size
- [ ] Same loss; no imbalance losses/samplers
- [ ] Same optimizer/scheduler and 20 epochs
- [ ] No online random augmentation; no hyperparameter tuning
- [ ] Seeds = 42, 123, 2026
- [ ] Environment, model revisions and config hashes locked
- [ ] Best checkpoint selected using Validation Macro-F1
- [ ] Test untouched until final evaluation
- [ ] 6 total runs
- [ ] Final report uses mean ± sample std
- [ ] Macro-F1 over evaluable classes is primary metric
- [ ] Evaluable class lists and real support reported
- [ ] Group-wise minority-class metrics reported
- [ ] Paired deltas, compute/exposure differences and limitations reported

