# Lab Day 21 - CI/CD for AI Systems

## Sinh viên: Nguyễn Trung Hiếu
## MSSV: 2A202601457

---

## Bước 1: MLflow Experiments

### Kết quả các thí nghiệm:

| Run | Model | Hyperparameters | Accuracy | F1 Score |
|-----|-------|-----------------|----------|----------|
| 1 | ExtraTrees | n_estimators=500 | 0.7520 | 0.7516 |
| 2 | ExtraTrees | n_estimators=1500 | 0.7560 | 0.7556 |
| 3 | RandomForest | n_estimators=300 | 0.7480 | 0.7465 |
| 4 | GradientBoosting | n_estimators=100, max_depth=5 | 0.6520 | 0.6505 |
| 5 | HistGradientBoosting | max_iter=100, max_depth=5 | 0.6680 | 0.6666 |
| 6 | ExtraTrees | n_estimators=2000 | 0.7520 | 0.7515 |

### Tham số tốt nhất:
- **Model**: ExtraTrees
- **n_estimators**: 1000
- **max_depth**: null
- **min_samples_split**: 2
- **min_samples_leaf**: 1
- **max_features**: null

### Phân tích:
- ExtraTrees cho kết quả tốt nhất do tính ngẫu nhiên trong feature splits
- GradientBoosting kém hơn vì overfit với learning_rate cao
- RandomForest ổn định nhưng chậm hơn ExtraTrees
- HistGradientBoosting nhanh nhưng cần tuning kỹ hơn

### Dataset:
- 5996 mẫu training (sau khi merge train_phase1 + train_phase2)
- 500 mẫu evaluation

---

## Bước 2: AWS S3 và GitHub Actions

### AWS Resources:
- **S3 Bucket**: `mlops-hieu-202608211828`
- **Model path**: `s3://mlops-hieu-202608211828/models/latest/model.pkl`
- **Region**: `us-east-1`

### GitHub Actions Pipeline:

4 jobs đều PASS:
1. **Unit Test** ✅ - Chạy pytest với 3 tests
2. **Train** ✅ - Huấn luyện model với ExtraTrees
3. **Deploy to S3** ✅ - Upload model lên AWS S3
4. **Eval** ✅ - Kiểm tra accuracy >= 0.70

### Secrets GitHub:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

---

## Bước 3: Continuous Training

Khi commit code mới (data changes hoặc params thay đổi), pipeline tự động:
1. Chạy tests
2. Huấn luyện lại model
3. Upload model mới lên S3
4. Kiểm tra chất lượng

---

## Tổng kết

### Điểm mạnh:
- Pipeline CI/CD hoàn chỉnh với 4 jobs
- Model đạt accuracy >= 0.70 (ExtraTrees ~0.76)
- Tự động deploy lên AWS S3
- Code có tests với coverage 100%

### Hạn chế:
- EC2 SSH bị block từ môi trường CI (chỉ deploy qua S3)
- Model size lớn (~460MB) nên upload chậm