# Công việc triển khai benchmark GeoGuessr-50k

| Mục | Giá trị |
|---|---|
| Benchmark | `geoguessr50k_country_augmentation_v1_2` |
| Nguồn | [Đặc tả v1.2 đã khóa](specs.md) và [kế hoạch triển khai](plan.md) |
| Trạng thái | 43 việc dự kiến. Dấu `[x]` dùng khi đã có bằng chứng đạt điều kiện của việc đó. |
| Cách đọc | Mỗi dòng dùng `- [ ] TNNN [P?] [USn?] Mô tả`, kèm đường dẫn tính từ project root. `[P]` đánh dấu việc có thể làm song song sau khi đủ phụ thuộc, trên file riêng. |

`US1` đến `US4` là các luồng nghiệm thu rút từ thứ tự §17 của đặc tả.

| Luồng | Kết quả nghiệm thu |
|---|---|
| `US1` — Inventory và split | Có dữ liệu cleaned đã audit, mapping, splits, groups và số evaluable classes. |
| `US2` — Preflight và offline augmentation | Có cấu hình/môi trường đã khóa, đủ quota, provenance và bộ train augmented đã qua QC, review và freeze. |
| `US3` — Huấn luyện ghép cặp | Có khả năng chạy và resume đúng ma trận 2×3, chọn best checkpoints bằng validation. |
| `US4` — Final evaluation và báo cáo | Có khả năng mở test sau khi train hoàn tất, lưu predictions và báo cáo paired results từ artifacts. |

Viết fixture cho hành vi mới trước, chạy kiểm tra để thấy lỗi đúng kỳ vọng, rồi triển khai tới khi phép kiểm tra đạt. Với các việc chạy dữ liệu/GPU hoặc review người, bằng chứng là artifact và trạng thái tương ứng. Các mốc chốt theo quy tắc ở [plan.md](plan.md#3-trình-tự-dữ-liệu-và-điều-kiện-chuyển-bước).

## Giai đoạn 1 — Setup

**Nghiệm thu** — Cài được package local, gọi được CLI help và chạy được pytest.

- [x] T001 Tạo package và bộ cài trong `pyproject.toml`, `.gitignore`, `src/geoguessr_benchmark/__init__.py`, `src/geoguessr_benchmark/__main__.py`, `tests/test_cli.py`. Khai báo Python dependencies và extra `dev`, để PyTorch/Torchvision có thể cài bản tương thích CUDA trước khi khóa môi trường. Cho CLI nhận `--root`, import module GPU theo lệnh. Kiểm tra `python -m geoguessr_benchmark --help`; bỏ qua dataset, checkpoints và cache trong Git.

## Giai đoạn 2 — Nền tảng dùng chung

**Nghiệm thu** — Cấu hình, canonical serialization và geometry có test độc lập. Các stage gọi chung helper đọc ảnh và xác minh artifacts.

- [x] T002 [P] Tạo `configs/benchmark.yaml`, `augmentation/traditional/config.yaml`, `augmentation/da_fusion/config.yaml`, loader/validator trong `src/geoguessr_benchmark/config.py` và fixture `tests/test_config.py`. Điền đúng giá trị §5–§11, gồm prompt/negative prompt nguyên văn §7.2, seed, group policy, training matrix và policy batch. `load_config(root)` tách preferred/fallback khỏi batch size hiệu lực materialize ở preflight. Fixture thay seed, prompt, resolution, strength, AMP hoặc khóa định danh phải báo lỗi trước chạy thật.
- [x] T003 [P] Viết canonical CSV/JSON/JSONL, SHA-256, atomic file writes và `verify_stage(root, stage)` trong `src/geoguessr_benchmark/artifacts.py`, kiểm tra bằng `tests/test_artifacts.py`. Giữ phiên bản schema, key/header, POSIX relative paths và thứ tự hàng theo contract. Fixture đổi bytes, đổi timestamp ngoài projection hoặc ngắt publish chứng minh hash thay đổi đúng chỗ và bước sau chỉ dùng stage hoàn chỉnh.
- [x] T004 [P] Viết `decode_rgb(path)`, `pixel_sha256(rgb)`, `write_output_jpeg(path, rgb)` và `letterbox_geometry(width, height, canvas_width, canvas_height)` trong `src/geoguessr_benchmark/images.py`. Kiểm tra `tests/test_images.py` với ảnh có EXIF orientation, pixel hash chứa dimensions, quality 95/subsampling 0/no EXIF và geometry classifier `1536×662 → 371×160`, `1030×703 → 234×160`; half-open bbox và pixel dư đúng phía theo §10.

## Giai đoạn 3 — Inventory, split và frequency groups (`US1`)

**Kiểm thử độc lập** — Fixture nhiều class chạy offline, xác định mapping và membership giống nhau khi đảo thứ tự file. Duplicate khác split/nhãn chặn freeze. Inventory thật đối chiếu số liệu khi dataset có mặt.

- [x] T043 [US1] Implement CLI `clean` và manifest provenance v1.2 trong `src/geoguessr_benchmark/cleaning.py`. `tests/test_cleaning.py` chứng minh bốn bbox chính xác, RGB neutral fill, native dimensions, cùng class/path, JPEG deterministic, raw bất biến, fail closed với resolution lạ và khi gián đoạn giữa publish thư mục ảnh và manifest.
- [x] T005 [P] [US1] Viết `tests/test_inventory.py` cho dataset nhỏ, tên Unicode, unreadable file, sai tổng counts/mapping, duplicate file bytes và hai JPEG khác bytes nhưng cùng oriented RGB. Cố định expected class IDs/path/hash bằng fixture và kiểm tra lỗi có lý do.
- [x] T006 [US1] Viết `build_inventory(root)` và audit trong `src/geoguessr_benchmark/inventory.py`. Xác minh cả raw và cleaned theo cleaning manifest trước inventory; mapping tên thư mục theo Unicode, image ID từ UTF-8 `raw_path`, `filepath` trỏ đến `data/cleaned_dataset/`, file/pixel hashes, canonical dataset fingerprint không timestamp. Ghi `reports/dataset_inventory.csv`, `splits/class_to_idx.json` và thông tin audit tại `reports/split_audit.json`. Đối chiếu resolution distribution, United States và min class support. Dừng với lỗi đọc ảnh, lệch 49.997 ảnh/124 classes, mapping mâu thuẫn hoặc exact duplicate xung đột nhãn.
- [x] T007 [P] [US1] Viết `tests/test_split.py`. Assert allocation `n=1,2,3,9,10,15` lần lượt `1/0/0`, `1/0/1`, `1/1/1`, `7/1/1`, `8/1/1`, `11/2/2`. Với logical path `data/compressed_dataset/A/{a,b,c,d}.jpg`, fixture qua cleaning giữ thứ tự phân bổ seed 42 của v1.1: train `b,a`, val `d`, test `c` trước sort CSV. CSV vẫn trỏ đến JPEG cleaned. Test nhóm trùng cùng split, quota không đổi, không xếp được thì fail closed. Kiểm tra disjoint IDs/file/pixel hashes, metadata có/vắng, group boundaries `n_train=19,20,99,100` và singleton target 5.
- [x] T008 [US1] Viết `allocate_split(n)` và `build_split(root)` trong `src/geoguessr_benchmark/split.py` theo §3. Dùng cleaning manifest, inventory và class mapping đã xác minh. Xếp theo salted hash của `raw_path`, dùng `raw_path` khi hòa, gom nhóm theo hash raw trong quota. Audit hash cleaned sau phân bổ và từ chối trùng giữa các split thay vì xáo membership. Tạo `splits/train.csv`, `splits/val.csv`, `splits/test.csv` theo schema/sort canonical. Tính counts/support và evaluable IDs từ các real splits.
- [x] T009 [US1] Hoàn thành group assignment và split lock trong `src/geoguessr_benchmark/split.py`, `src/geoguessr_benchmark/artifacts.py`. Ghi `splits/class_groups.csv` từ original train, budget, `accepted_actual=0`, `splits/split_metadata.json` và `benchmark_lock.json`. Audit cross-split IDs/hashes và conflicting labels trước publish, ghi incident vào `reports/split_audit.json`. Khi có panorama/location metadata, audit trùng panorama hoặc chụp gần, còn khi vắng ghi giới hạn exact-duplicate audit. Đối chiếu counts/support dự kiến và giữ hash cột group bất biến.
- [x] T010 [US1] Gắn lệnh `clean`, `inventory`, `split`, `verify` vào `src/geoguessr_benchmark/__main__.py`, chạy `python -m pytest -q tests/test_inventory.py tests/test_split.py`. Trong `tests/test_inventory_split_cli.py`, thực thi các command handlers trên fixture và xác nhận `reports/split_audit.json` cùng `benchmark_lock.json` ở stage `splits`. Biên fixture cấp baseline riêng, CLI chính thức giữ baseline §2.

## Giai đoạn 4 — Preflight, generation, QC và freeze (`US2`)

**Kiểm thử độc lập** — Fixtures ảnh local kiểm tra thứ tự attempts, quota, content-only output, chặn holdout leakage và kiểm tra DINO độc lập. Preflight GPU dùng model thật và review người là bằng chứng riêng trước freeze dữ liệu thật.

- [x] T011 [P] [US2] Viết `classifier_transform(path)`, `build_model(weights_path)`, `build_optimizer(model, config)` trong `src/geoguessr_benchmark/classifier.py` và `tests/test_classifier.py`. Cố định EfficientNet-B0 IMAGENET1K_V1, 124 logits, initialized Linear và dropout 0.2. Test cả JPEG cleaned và synthetic ở hai native dimensions, shape `float32 [3,160,384]`, padding zero sau normalize, resize giữ ratio, preprocessing deterministic, tất cả backbone parameters trainable. Optimizer và loss helper dùng chung cho preflight/training theo §11.
- [x] T012 [P] [US2] Viết snapshot resolution và môi trường trong `src/geoguessr_benchmark/preflight.py`, kiểm thử `tests/test_model_lock.py`. Resolve SD 1.5 immutable commit và hash weights/tokenizer/VAE/scheduler config, pin DINO code commit/weights và EfficientNet weight SHA. Nạp snapshot theo đúng hash. `environment.lock.txt` ghi Python/packages, backend/rounding, OS/GPU/driver/CUDA/cuDNN, code commit hoặc archive hash. Fixture bỏ revision hoặc đổi một trọng số phải chặn preflight; lưu fingerprints thực khi chạy.
- [x] T013 [US2] Viết `run_preflight(root)` và memory ladder trong `src/geoguessr_benchmark/preflight.py`, fixture `tests/test_preflight.py`. Thử `128,64,32,16,8,4,2` chỉ khi OOM, với input shape thật, forward/backward/AdamW allocation và deterministic CUDA FP32. Reset model/optimizer/RNG; kiểm lỗi ngoài OOM và OOM ở 2 phải dừng. Định nghĩa chỗ gọi SD probe và state publish. T017 cung cấp probe thật cho canvas native pad /8, batch 1 trên cleaned train source lớn nhất; báo OOM rõ ràng. Publish batch size, config và environment hashes khi cả hai probe đạt.
- [x] T014 [P] [US2] Viết candidate planning trong `src/geoguessr_benchmark/augmentation/policy.py` và `tests/test_augmentation_policy.py`. Sort cleaned original train sources theo filepath, round robin theo `attempt_index`, tái lập ID, count quota accepted, cap `10×budget`, skip `many`. Check literal candidate seed `1121161069` cho `traditional,class_id=7,j=0`, `3767014528` cho `j=1`, `3439134376` cho `da_fusion,class_id=7,j=0`. Reject source lạc split, synthetic source hoặc class/source mismatch.
- [x] T015 [P] [US2] Viết `generate_traditional(source, attempt, config)` trong `src/geoguessr_benchmark/augmentation/traditional.py`, kiểm thử `tests/test_traditional.py`. Dùng cleaned train native. Sample k đều trong `{1,2}`, chọn không lặp trong bảy phép §6, chạy theo thứ tự bảng, clip sau mỗi phép và lưu params thực tế. Verify output giữ dimensions, sample order/params ổn định theo seed và output JPEG theo policy chung.
- [x] T016 [P] [US2] Viết native geometry, `prepare_generation_canvas(rgb)` và `crop_generated_content(canvas, geometry)` trong `src/geoguessr_benchmark/augmentation/diffusion.py`. Test giữ nguyên pixel, reflection tối thiểu bội số 8, hai canvas `1536×664` và `1032×704` với bbox `[0,1,1536,663]` và `[1,0,1031,703]`, crop về native và loại padding sentinel.
- [x] T017 [US2] Viết `generate_diffusion(source, attempt, config, snapshot)` trong `src/geoguessr_benchmark/augmentation/diffusion.py`, kiểm thử adapter ở `tests/test_diffusion_pipeline.py` và nối vào T013. Khóa SD1.5 + DDIM từ snapshot, FP32, standard attention, `strength=0.35`, guidance `5.0`, 40 requested steps, `eta=0.0`, batch 1 và đúng prompt. Ghi actual timesteps, model/revision, source/canvas/output geometry và hashes theo §9. `safety_flag` đưa candidate vào rejection log. Fixture stub adapter xác nhận input shape, tham số và crop trước JPEG. GPU preflight thật nằm ở T038.
- [x] T018 [P] [US2] Viết `check_candidate(root, candidate, source, config, hash_index)` và duplicate/leakage audit trong `src/geoguessr_benchmark/augmentation/qc.py`, test `tests/test_qc.py`. Decode final JPEG, kiểm dims, RGB, provenance, grayscale std `≥5`, near-black/white fraction `<0.98`. Diffusion có Laplacian blur floor `max(5,0.5×source_var)` trên content theo §8.1. Test boundary, file hỏng, bbox/dim sai, exact duplicate vào original/accepted và collision với holdout chặn freeze. Fixture traditional chứng minh blur/DINO/reviewer không ảnh hưởng quyết định.
- [x] T019 [US2] Tạo `export_review_queue(root, output)` và `import_review_decisions(root, input)` trong `src/geoguessr_benchmark/augmentation/review.py`, kiểm tra `tests/test_human_review.py`. Giữ schema `augmentation/manifests/review_queue.jsonl` gồm `synthetic_id`, source/output paths, `source_sha256`, `output_sha256`, `class_name`, `rubric_version`. `review-export` đồng thời tạo `augmentation/review/` gồm `index.html`, bản sao `review_queue.jsonl`, `sources/` và `generated/`, chỉ chứa các cặp diffusion đã qua automatic QC và đang chờ review. Copy nguyên bytes của cleaned source đã xác minh thuộc `splits/train.csv` và final generated JPEG content-only. HTML mở trực tiếp từ `file://`, dùng dữ liệu nhúng và đường dẫn ảnh tương đối, hiển thị class, `synthetic_id`/`attempt_index`, hai ảnh, hai SHA-256 và rubric §7.3. Chỉ export dữ liệu thuộc phạm vi review, loại ảnh val/test, classifier scores, predictions và DINO scores khỏi gói và UI.
  - UI cung cấp **Accept**, **Reject — geographic cue changed**, **Reject — generation artifact**, **Reject — ambiguous**. Xuất `reviews.jsonl` với `synthetic_id,source_sha256,output_sha256,reviewer_id,review_timestamp,rubric_version,decision,reason`, timestamp UTC và `decision` là `accept`/`reject`. Ba lựa chọn reject ghi `geographic_cue_changed`, `generation_artifact`, `reject_ambiguous`. `review-import --input` tiếp tục nhận cùng schema, ví dụ từ `augmentation/manifests/reviews.jsonl`. Đối chiếu candidate ID, rubric và cả hai hash với hàng đợi lẫn bytes file hiện tại trước khi áp dụng. Từ chối decision trùng mâu thuẫn, sai hash/candidate và ambiguous bị accept. Candidate chưa có review vẫn pending và chặn freeze. Fixture kiểm tra đủ các cặp hợp lệ, mở `index.html` offline sau khi copy, tải JSONL và import thành công. Thay bytes ở source hoặc output phải làm quyết định tương ứng bị từ chối.
- [x] T020 [US2] Viết `generate_offline(root)` và state transitions trong `src/geoguessr_benchmark/augmentation/workflow.py`, test `tests/test_augmentation_workflow.py`. Ghi mọi attempt vào `augmentation/manifests/candidates.jsonl`, lưu final JPEG đúng `data/augmented_dataset/{traditional,da_fusion}/<class_name>/`, apply automatic QC và review khi cần. Xử lý theo class ID rồi attempt order, finalize acceptance theo đúng thứ tự kể cả resume, dừng ở `review_required` nếu cần decision. Khi đủ quota và audit đạt, publish `membership_frozen`; nếu cap hết ghi shortage và dừng. Fixture retry không ghi đôi, không đổi seed, không nhận rejected, không ghi đè originals.
- [x] T021 [P] [US2] Viết `log_dino_diagnostics(root)` trong `src/geoguessr_benchmark/augmentation/diagnostics.py`, test `tests/test_dino_diagnostics.py`. Chỉ chạy sau `membership_frozen`. Dùng cleaned source native và output content-only, frozen `dino_vits16`, letterbox `224×224` ImageNet mean/normalize, global embedding L2. Ghi source cosine, nearest accepted sibling cùng source và sibling ID cho các diffusion outputs đọc được, loại chính ảnh khỏi tập tìm. Với lỗi của từng candidate đọc được, ghi `null` cho các trường bị ảnh hưởng và ghi `dino_diagnostic_error`. Test singleton không sibling, lỗi embedding, candidate rejected và score cực trị. Assert trạng thái accepted/rejected giữ nguyên và `membership hash before DINO == membership hash after DINO`. Diagnostics chỉ ghi metadata, tuyệt đối không thêm threshold hoặc lọc ảnh.
- [x] T022 [US2] Viết `freeze_augmented(root)` trong `src/geoguessr_benchmark/augmentation/workflow.py`, mở rộng `tests/test_augmentation_workflow.py`. Xác minh `membership_frozen` và hash đã khóa, gọi `log_dino_diagnostics(root)`, rồi tính lại accepted membership hash từ candidate records sau diagnostics. Chỉ khi hai hash trùng và các kiểm tra union, provenance, exact hashes, quotas, review và groups đều đạt mới publish `augmentation/manifests/accepted.jsonl`, `augmentation/manifests/qc_summary.json` và `splits/train_augmented.csv`. Cập nhật `accepted_actual` và group SHA trong `splits/class_groups.csv`/`splits/split_metadata.json` theo plan §3, stage các file rồi publish lock `augmented` sau cùng. Fixture cố ý đổi membership trong diagnostics phải chặn publish cả ba artifacts và final lock. Giữ kiểm thử khôi phục metadata bị gián đoạn và xác minh các real splits giữ nguyên.
- [x] T023 [US2] Đăng ký `preflight`, `augment`, `review-export`, `review-import`, `augment-freeze` trong `src/geoguessr_benchmark/__main__.py`. `augment-freeze` gọi `freeze_augmented(root)` ở T022 để thực thi diagnostics trước publish. Trong fixture tích hợp `tests/test_benchmark_workflow.py`, gọi CLI, gắn spy quanh `log_dino_diagnostics(root)` và xác minh hàm được gọi sau `membership_frozen`, trước khi publish ba artifacts và final lock. Fixture có lỗi DINO trên candidate đọc được phải ghi `null` kèm lỗi, giữ nguyên accepted membership hash và vẫn tới stage `augmented`. Chạy `python -m pytest -q tests/test_preflight.py tests/test_diffusion_geometry.py tests/test_qc.py tests/test_dino_diagnostics.py tests/test_augmentation_workflow.py tests/test_augmentation_cli.py` và các fixture cùng giai đoạn. Fixture preflight xác minh lệnh chính thức cần đủ splits, GPU/snapshots và probes trước khi publish stage `preflight`.
- [x] T024 [US2] Thử CLI augmentation từ fixture `tests/test_augmentation_cli.py`. Dùng stub model và review rows để đi qua `review_required` → import decisions → `membership_frozen` → `augmented`. Xác minh `augmentation/manifests/qc_summary.json`, `splits/train_augmented.csv`, `benchmark_lock.json` và đường đi lỗi khi thiếu quota, audit lỗi, sai output hash hoặc thiếu review. Kiểm tra candidate sinh lại đúng ID khi resuming và diagnostics lỗi vẫn xuất đúng union.

## Giai đoạn 5 — Code huấn luyện và khóa best checkpoints (`US3`)

**Kiểm thử độc lập** — Tiny fixtures kiểm chứng sampler, LR, metrics, paired initialization, checkpoint tie, resume ở epoch boundary và điều kiện chốt sáu runs. Sáu main runs được thực thi ở T040 sau khi các test bắt buộc §17 đạt.

- [x] T025 [P] [US3] Viết `learning_rate_at_step(step, steps_per_epoch, config)` và lịch LR/sample traversal trong `src/geoguessr_benchmark/training.py`. Kiểm tra `tests/test_schedule.py` cho `k=0 → 3e-5`, `k=L → 3e-4`, `u=10.5 → 0.0001505`, `k=20L-1 > 1e-6`, cùng `u` cho A/B và batch cuối giữ đủ records. Seed worker modulo `2**32`, epoch shuffle không replacement, batch size chung, apply LR trước optimizer step và lưu từng step.
- [x] T026 [P] [US3] Viết `tests/test_metrics.py`. Với true `[0,0,1,2]`, pred `[0,2,3,0]` trên 124 classes, assert Top-1 `0.25`, Macro-F1 `1/6`, group `{0,1}` F1/Recall `0.25` và class 3 support 0 có metrics `null`. Kiểm tra full confusion nhận FP từ class 2 vào group, empty group `null`. Fixture logits hòa xác nhận top-5 `[0,1,2,3,4]` và target class 5 trượt top-5.
- [x] T027 [US3] Viết `rank_logits(logits)` và `evaluate_predictions(rows, evaluable_ids, class_groups)` trong `src/geoguessr_benchmark/metrics.py`. Dùng 124 logits, hòa chọn class ID nhỏ hơn, confusion `124×124` true/pred, support/TP/FP/FN, macro trên real evaluable IDs, weighted theo support. Group F1/recall lấy từ confusion toàn split, export đủ 124 per-class rows, `not evaluable on this split` cho support 0. Reuse metrics cho validation, final test và report.
- [x] T028 [US3] Viết `train_run(root, condition, seed)` trong `src/geoguessr_benchmark/training.py`, test `tests/test_training.py`. Verify lock, nạp init đúng seed, seed sạch, val ở eval/no-grad, khôi phục train mode, 20 epochs, compute sample-weighted train loss. Chọn best theo unrounded val Macro-F1, hòa giữ sớm nhất. Lưu best/last checkpoint và RNG (gồm train loader generator), scheduler step, config/data/init hashes, `history.csv`, `lr_steps.csv`, validation-at-best metrics/confusion/per-class/group vào `runs/{condition}/seed_<seed>/`. Fixture ngắt giữa checkpoint/log và resume ở epoch boundary chứng minh giữ sample order, step counter, lịch sử và best epoch. Sai hash phải từ chối resume.
- [x] T029 [P] [US3] Viết `initialize_pairs(root)` trong `src/geoguessr_benchmark/experiment.py` và `tests/test_pairing.py`. Với seeds `42,123,2026` tạo `initializations/seed_<seed>.pt` từ cùng pretrained snapshot và seeded head, ghi SHA-256 vào lock `initialized`. Assert A/B cùng seed nạp đúng bytes và state values. Check shared config projection loại đúng trường §17, split/map/group/preprocessing/checkpoint policy hashes khớp và ba augmented runs trỏ cùng union manifest.
- [x] T030 [US3] Viết launcher trong `src/geoguessr_benchmark/experiment.py`, kiểm tra `tests/test_launcher.py`. Khởi động subprocess run với `PYTHONHASHSEED=seed`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` trước CUDA import/init, cấu hình deterministic, TF32 off. Truyền đúng snapshot/config và phân tách RNG train/eval. Ghi vào `runs/{condition}/seed_<seed>/run_metadata.json` theo §16, gồm timestamps UTC, runtime identity, counts, steps, selected epoch, hashes và failure/resume history.
- [x] T031 [US3] Tạo `train_all(root)` và `freeze_checkpoints(root)` trong `src/geoguessr_benchmark/experiment.py`, đăng ký `init`, `train-all`, `freeze-checkpoints` trong `src/geoguessr_benchmark/__main__.py`, thêm `tests/test_run_matrix.py`. Orchestrate đúng thứ tự A42/B42/A123/B123/A2026/B2026 trên một GPU. Retry đúng run bị lỗi với dữ liệu/cấu hình giữ nguyên và verify trước mỗi run. `benchmark_lock.json` chỉ đạt `trained` khi đủ sáu completed runs/20 epochs và best-checkpoint hashes đã khóa. Fixture kiểm tra 5/6 runs, thiếu history, sai hash và thiếu completed epoch đều chặn freeze.

## Giai đoạn 6 — Code final evaluation và reporting (`US4`)

**Kiểm thử độc lập** — Test gate fail trước khóa sáu best checkpoints. Từ bộ predictions fixture có thể tạo đủ báo cáo, định dạng null đúng và tái tổng hợp mà không chạy inference lần nữa.

- [x] T032 [P] [US4] Viết `tests/test_final_evaluation.py`. Stub inference để xác nhận `test-all` bị chặn trước `trained`, dùng best checkpoint hash đúng, đủ test IDs/true labels/support và tái tính metrics từ `predictions_test.csv` đã khóa. Kiểm tra rerun tái sử dụng predictions hợp lệ. Fixture lỗi hạ tầng cho phép phục hồi cùng checkpoint/config, giữ lịch sử lỗi.
- [x] T033 [US4] Viết `evaluate_test_once(root)` và `load_test_predictions(root, condition, seed)` trong `src/geoguessr_benchmark/evaluate.py`. Verify test real manifest, trained-stage lock và từng best checkpoint trước inference. Ghi và khóa `runs/{condition}/seed_<seed>/predictions_test.csv` theo plan §8. Xuất test confusion/per-class, cập nhật `metrics.json`/`group_metrics.json` giữ riêng validation-at-best và final-test. Tái tạo metrics dở dang từ predictions hợp lệ đã khóa cùng run. Ghi lỗi hạ tầng vào metadata, publish `tested` khi đủ sáu test artifacts.
- [x] T034 [P] [US4] Viết `tests/test_aggregation.py`. Khớp theo seed dù input bị đảo hàng, reject thiếu/duplicated seed. Với A `[.2,.4,.6]`, B `[.3,.6,.7]` ghép `42,123,2026`, A mean `.4`, sample std `.2`, B mean `0.5333333333…`, paired delta mean `0.1333333333…` và sample std `0.0577350269…`. Assert bảng paired delta theo pp. Fixture group không có support giữ `null` khi aggregate.
- [x] T035 [US4] Viết `build_report(root)` trong `src/geoguessr_benchmark/reporting.py` và `tests/test_reporting.py`. Verify stage `tested`, đọc predictions/metadata đã khóa, aggregate đúng ba seeds với `ddof=1` và paired deltas trên đầy đủ precision. Tạo `reports/per_seed_metrics.csv`, `reports/aggregate_metrics.json`, `reports/group_comparison.csv`, `reports/final_report.md` theo §15–§18. Báo sáu metrics chính, group F1/recall/support, per-class F1/recall/support và paired changes, QC coverage, steps/exposures/duration, raw per-seed appendix và diễn giải fixed epochs bằng tiếng Việt. Đăng ký `test-all`, `report` trong `src/geoguessr_benchmark/__main__.py`.

## Giai đoạn 7 — Tích hợp và thực thi chính thức

**Nghiệm thu** — Toàn bộ §17 và checklist §20 có bằng chứng. Mọi lệnh thật chỉ chạy khi bước trước đã xuất artifact đủ điều kiện.

- [x] T036 Tạo kiểm thử tích hợp `tests/test_benchmark_workflow.py` trên tiny fixtures, chạy `python -m pytest -q`. Dùng dữ liệu giả và stub cho model/reviewer. Thử clean → inventory → split → preflight → augment → review → accepted freeze → init → training gate → test gate → report, bao gồm checksum hỏng, resume và final test gate. Xác nhận đủ các checks §17 về allocation, disjointness, group boundaries, train-only sources, paired init, geometry/RGB, absent classes/outside-group predictions và paired delta với `ddof=1`. Fixture smoke nằm trong workdir tạm, tách khỏi artifacts chính thức.
- [ ] T037 Chạy `clean` trên raw, rồi `inventory`, `split`, `verify --stage splits` chỉ trên cleaned. Kiểm tra `reports/cleaning_manifest.json` trước khi khóa. Đối chiếu `reports/dataset_inventory.csv`, `reports/split_audit.json`, `splits/split_metadata.json`, `benchmark_lock.json` với baseline và counts/support dự kiến train `39.978/124`, val `5.007/106`, test `5.012/111`. Báo counts/support thực tế và mọi sai lệch. Việc này cần bộ ảnh thật, T036 dùng fixture độc lập.
- [ ] T038 Chạy `preflight`, `verify --stage preflight` trên runtime chính thức. Kiểm trace SD img2img FP32/standard attention, batch 1 với padded canvas native lớn nhất và crop lại native dimensions. Kiểm tra riêng classifier input `batch×3×160×384`, forward/backward/AdamW state, deterministic ops, batch được chọn và immutable revisions. So `environment.lock.txt`, `configs/benchmark.yaml` và `benchmark_lock.json` với code commit/archive, GPU, model hashes thực. Stage chỉ đạt khi cả probe SD và probe classifier thành công.
- [ ] T039 Chạy `augment` đến khi có pending, dùng `review-export`, copy `augmentation/review/` sang máy reviewer, mở `index.html` offline, rồi nhập `reviews.jsonl` bằng `review-import`. Reviewer duyệt mọi diffusion candidate vượt automatic QC trong hàng đợi theo §7.3. Lặp đến `membership_frozen`, rồi chạy `augment-freeze` để ghi diagnostics, kiểm tra membership hash và publish stage `augmented`. Xác minh `augmentation/manifests/accepted.jsonl`, `augmentation/manifests/qc_summary.json`, `splits/train_augmented.csv` và `benchmark_lock.json`. Đối chiếu dự kiến 1.513 traditional + 916 diffusion accepted = 2.429, B total 42.407, và đủ quota từng class từ inventory thật. Ghi shortage, incident hoặc review còn thiếu nếu stage dừng.
- [ ] T040 Chạy `verify --stage augmented`, `init`, `train-all`, `freeze-checkpoints`. Tại `initializations/seed_<seed>.pt`, `runs/{condition}/seed_<seed>/` và `benchmark_lock.json`, kiểm đủ ba init hashes ghép cặp, cùng batch/config/evaluable IDs, sáu runs đủ 20 epochs, counts/steps chính xác và sáu best checkpoints đã chốt. Nếu run lỗi, log nguyên nhân và resume/retry đúng run theo §12.
- [ ] T041 Chạy `test-all` sau khi `benchmark_lock.json` đạt `trained`. Đối chiếu sáu `runs/{condition}/seed_<seed>/predictions_test.csv` với test manifest real, best-checkpoint hash, support và full `124×124` confusion. Đối chiếu đủ group/per-class test metrics và validation-at-best trước khi chốt `tested`.
- [ ] T042 Chạy `report` và `verify --stage reported`. Đối chiếu `reports/final_report.md`, `reports/per_seed_metrics.csv`, `reports/aggregate_metrics.json`, `reports/group_comparison.csv` với §20 và predictions đã khóa. Báo counts thực, evaluation coverage toàn split/theo group, run/retry history, hạn chế và kết quả kể cả hiệu ứng âm hoặc gần 0. Kiểm report có thể xây lại số liệu từ artifacts bằng cách đọc predictions đã khóa.

## Phụ thuộc và cách triển khai

Thứ tự này áp dụng khi chạy dữ liệu thật. Sau nền tảng, các việc viết code và kiểm thử có thể chạy song song ở những biên được đánh dấu `[P]`, miễn là đầu vào contract đã sẵn sàng. Khi một số việc cùng ghi file, thực hiện lần lượt.

| Việc | Phụ thuộc trực tiếp |
|---|---|
| T001 | Project root |
| T002, T003, T004 | T001 |
| T005, T007 | T002–T004 |
| T043 | T003, T004 |
| T006 | T005, T043 |
| T008 | T006, T007 |
| T009 | T008 |
| T010 | T009 |
| T011, T012, T016 | T002–T004 |
| T013 | T011, T012. Adapter thật được tích hợp ở T017 |
| T014 | T002–T004, T009 |
| T015 | T014 |
| T017 | T012, T013, T014, T016 |
| T018 | T014, T016, schema output/provenance §6 của plan |
| T019 | T018 |
| T020 | T015, T017, T018, T019 |
| T021 | T012, T014, T016. Viết bằng fixture theo membership contract, chỉ chạy thật sau T020. |
| T022 | T020, T021 |
| T023 | T010, T013, T017, T022 |
| T024 | T023 |
| T025 | T002, T011 |
| T026 | T002, metric definitions §13–§14 |
| T027 | T026 |
| T028 | T011, T022, T025, T027. Dùng init schema §6–§7 của plan và fixture state dict. |
| T029 | T002–T004, T009, T011, T022 |
| T030 | T028, T029 |
| T031 | T030 |
| T032 | T027, T031 |
| T033 | T027, T031, T032 |
| T034 | T002, T027 |
| T035 | T027, T033, T034 |
| T036 | T001–T035, T043 |
| T037 | T036, dataset thật |
| T038 | T037, GPU và immutable model snapshots |
| T039 | T038, người review geographic cues cho các candidates cần duyệt |
| T040 | T039 và lock `augmented` hợp lệ |
| T041 | T040 chốt đủ sáu best checkpoints |
| T042 | T041 |

- **Mốc đầu tiên** — Chạy `US1` với tiny fixtures đến khi tái tạo được inventory, split, groups và checksum. Mốc này chứng minh pipeline dữ liệu gốc.
- **Ví dụ song song** — Sau T001, T002/T003/T004 triển khai config, artifact helpers và image helpers độc lập. Khi T014/T016 hoàn tất, traditional, diffusion QC và DINO có thể làm trên module/fixture riêng theo bảng phụ thuộc. T026 và T034 có thể thiết kế fixture số độc lập. Việc ghi accepted manifest, group metadata và lock final dùng một writer.
- **Kết quả benchmark** — Cần đủ bốn luồng, preflight thật, offline set đã khóa, đủ sáu training runs và final evaluation. Tiny fixtures kiểm tra tính đúng của code. Dataset thật, người review và GPU là đầu vào cho phần thực nghiệm.

## Bằng chứng xác minh v1.2

`python -m pytest -q` sau bản sửa giữ danh tính và split v1.2: **140 passed in 75.02s**. Lần kiểm tra package v1.2 trước đó đã build/cài với dependencies có sẵn và xác nhận CLI hiển thị lệnh `clean`.

| Việc đã đánh dấu | Bằng chứng fixture |
|---|---|
| T001–T004 | CLI help, config bất biến, lock/atomic-write và JPEG/RGB/letterbox. |
| T005–T010, T043 | HUD masking, raw bất biến và provenance raw/cleaned được kiểm tra, chặn publish dở, ID và split order giữ theo v1.1, CSV vật lý cleaned, quota, duplicate audit và CLI fixture. |
| T011–T018 | Classifier, snapshot/hash, native diffusion geometry/FP32, preflight worst-case/OOM bằng stub, source policy, traditional native và QC. |
| T019–T024 | Gói review offline và JSONL round-trip, holdout/shortage, resume, DINO và khóa membership/union. |
| T025–T031 | Scheduler, metrics, launcher, paired init, checkpoint resume/identity và điều kiện đủ sáu runs fixture. |
| T032–T036 | Final test gate, predictions khóa, aggregation/report và smoke từ `clean` đến báo cáo. |

T037–T042 vẫn mở. Cần dataset thật đã clean, GPU preflight v1.2, human review và đủ sáu run thật trước final test/report.

## Ma trận truy vết đặc tả và nghiệm thu

| Mục `specs.md` | Việc chịu trách nhiệm | Bằng chứng |
|---|---|---|
| §1 — Hai condition và biến can thiệp | T002, T029–T031, T035, T040, T042 | Shared config projection, ma trận 2×3, manifest A/B và per-seed table |
| §2 — Cleaning/dataset/inventory | T004–T006, T010, T037, T043 | `cleaning_manifest.json`, `dataset_inventory.csv`, baseline/mapping checks và `split_audit.json` |
| §3, §3.1, §3.2 — Split và leakage | T007–T010, T018, T022, T036–T037 | Fixed original CSVs, disjoint IDs/hashes, source subset và lock |
| §4 — Groups và quota | T007, T009, T014, T022, T024, T039 | `class_groups.csv`, quotas, counts và `qc_summary.json` |
| §5 — Policy generation | T004, T014, T020, T024, T039 | Seed/order fixtures, attempt cap, JPEG/rejection logs và shortage gate |
| §6 — Traditional | T002, T015, T018, T024, T039 | Sampled transforms, source dimensions và automatic QC |
| §7.1, §7.2, §7.3 — Native diffusion | T012–T013, T016–T019, T023–T024, T038–T039 | SD snapshot, FP32/canvas/crop, prompt, actual timesteps, bundle offline và human decisions |
| §8.1, §8.2 — QC/DINO | T018–T024, T039 | Blank/blur/hash fixtures, review logs, DINO null/error, CLI gọi diagnostics trước publish và membership hash giống nhau trước/sau |
| §9 — Provenance/union | T014, T017, T019–T022, T024, T039 | `candidates.jsonl`, `accepted.jsonl`, dims, hashes và `train_augmented.csv` |
| §10 — Model/preprocessing | T004, T011, T016, T029, T036 | 124 logits, paired init, classifier mean padding và tests hai bbox mẫu |
| §11, §11.1, §11.2 — Cấu hình, LR, seeds | T002, T011–T013, T023, T025, T028–T031, T038, T040 | OOM/determinism preflight, effective batch, LR/steps, RNG và environment lock |
| §12 — Runs/checkpoints/final test gate | T028–T033, T040–T041 | Best/last, earliest tie, resume evidence, trained lock và test predictions |
| §13 — Metrics/label space | T026–T028, T032–T033, T041 | Full confusion, Top-5 tie, evaluable denominator và 124 per-class rows |
| §14 — Group evaluation | T007, T009, T026–T027, T035, T041 | Full-confusion FP/FN qua group, group support và null khi trống |
| §15 — Paired aggregation | T034–T035, T041–T042 | Three-seed mean/sample std, paired deltas pp, raw per-seed và group/per-class tables |
| §16 — Artifact layout | T043, T003, T006, T009, T012, T020–T024, T028–T031, T033, T035 | Các file yêu cầu có schema, checksum và bên tạo tương ứng |
| §17 — Contract và implementation acceptance | T023–T024, T029, T031, T036–T042 | Mandatory tiny checks, shared config/initial hashes, các mốc freeze |
| §18 — Rủi ro và diễn giải | T006, T009, T019, T021, T035, T042 | Leakage/domain gap/review evidence, support, exposure và giới hạn kết luận |
| §19 — Fixed defaults | T002–T004, T007–T009, T011–T018, T021, T023, T025–T029, T034, T038 | Config validation, fixtures theo literal values và khóa các giá trị resolve từ runtime |
| §20 — Final checklist | T010, T023–T024, T036–T042 | Stage locks, kiểm thử bắt buộc, sáu runs, predictions đã khóa và báo cáo tái tổng hợp |
