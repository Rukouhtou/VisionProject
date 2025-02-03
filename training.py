from ultralytics import YOLO

# 사전학습된 모델 로드(파인튜닝용)
model = YOLO('yolo11n.pt')

def main():
    train_results = model.train(
        data="./custom.yaml",  # 커스텀 데이터용으로 만든 yaml파일
        epochs=300,
        imgsz=640,  # 이미지 자동 resize(비율 유지)를 위한 큰 차원쪽 사이즈 입력
        device="0",
    )

    metrics = model.val()

    # ONNX 포맷으로 모델 저장
    success = model.export(format = 'onnx', device = '0')
    # TensorRT 포맷으로 모델 저장(추론속도 향상위한 int8 quantization으로 압축)
    success2 = model.export(format = 'engine', device = '0', int8 = True)

if __name__ == "__main__":
    main()
