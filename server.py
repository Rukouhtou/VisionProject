import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile


app = FastAPI()
model = YOLO('pretrained_model/best.onnx')

# 클라이언트에서 cv2로 그릴 정보들
names = model.names     # 클래스 딕셔너리
class_len = len(names)
colors = np.random.randint(0, 256, size=(class_len, 3)).astype(int) # 클래스별 바운딩박스 색깔 담을 넘파이배열
font_scale = 0.5
font_thickness = np.round(np.linspace(2, 1, class_len)).astype(int)   # 중요한 바운딩박스일수록 큰 라벨 텍스트
box_thickness = np.round(np.linspace(3, 1, class_len)).astype(int)

@app.post("/detect")
async def detect(file: UploadFile):
    # 클라이언트에서 받은 바이트형태의 이미지파일을 넘파이 배열로 변환후, cv2의 이미지배열로 변환
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    results = model.predict(image)

    # yolo의 바운딩박스 데이터(텐서객체이므로 cpu메모리로 옮긴후 넘파이배열로 변환)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    scores = results[0].boxes.conf.cpu().tolist()
    classes = results[0].boxes.cls.cpu().tolist()
    
    # 클라이언트에서 쓸 리스트로 데이터를 반환
    output = []
    for box, score, cls in zip(boxes, scores, classes):
        print(colors[int(cls)])
        output.append({
            'x1': float(box[0]),
            'y1': float(box[1]),
            'x2': float(box[2]),
            'y2': float(box[3]),
            'confidence': float(score),
            'class': names[int(cls)],
            'color': tuple(colors[int(cls)].tolist()),
            'font_scale': font_scale,
            'font_thickness': int(font_thickness[int(cls)]),
            'box_thickness': int(box_thickness[int(cls)]),
        })
        
    return {'detections': output}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
