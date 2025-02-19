from ultralytics import YOLO

# 사전학습된 모델(파인튜닝용)
model = YOLO('yolo11n.pt')

train_results = model.train(
    data="./custom.yaml",  # 커스텀 데이터용으로 만든 yaml파일
    epochs=100,
    batch=32,
    lr0=0.005,
    lrf=0.3,
    warmup_epochs=5,
    weight_decay=0.001,
    cos_lr=True,
    imgsz=640,  # 이미지 크기 preprocessing
    device="0",
)

metrics = model.val()

# ONNX 포맷으로 모델 저장
success = model.export(format = 'onnx', device = '0')
# TensorRT 포맷으로 모델 저장(추론속도 향상위한 int8 quantization으로 압축)
success2 = model.export(format = 'engine', device = '0', int8 = True)
