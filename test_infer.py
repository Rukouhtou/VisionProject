from ultralytics import YOLO


model = YOLO('yolo11n.pt')

if __name__ == '__main__':
    train_results = model.train(
        data = 'coco8.yaml',
        epochs = 10,
        imgsz = 640,
        device = '0',
    )

    metrics = model.val()

    results = model.predict('./image_input/img3.jpg')

    for i, result in enumerate(results):
        boxes = result.boxes
        masks = result.masks
        # keypoints = result.keypoints
        probs = result.probs
        result.show()
        result.save(filename = f'output/result_{i}.jpg')


    # ONNX 포맷으로 모델 저장
    success = model.export(format = 'onnx', device = '0')
    # TensorRT 포맷으로 모델 저장(추론속도 향상위한 int8 quantization으로 압축)
    # success = model.export(format = 'engine', device = '0', int8 = True)
