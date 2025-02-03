##
[![SVG Banners](https://svg-banners.vercel.app/api?type=glitch&text1=비전AI%20🤖&width=800&height=150)](https://github.com/Akshay090/svg-banners)
# VisionProject
해당 프로젝트는 비전AI 모델 서비스를 fastAPI서버를 사용해 client로 배포하는 내용을 담고 있습니다.

웹캠이달린 클라이언트쪽에서는 WebSocket을 통하여 서버와 영상 송수신 + 웹브라우저를 통한 모니터링,  
서버쪽에서는 fastAPI를 통해 받은 영상을 파인튜닝된 yolo모델로 처리후 다시 전송하는 역할을 합니다.

## Prerequisites
- Python 3.10.16
- pip
- conda
- webcam

## Install
conda 가상환경을 만든 뒤 다음 커맨드를 이용하여 [yolo11](<https://github.com/ultralytics/ultralytics>)을 설치합니다:  
```
pip install ultralytics
```
그런 다음 requirements를 설치합니다:
```
pip install -r requirements.txt
```

## How to use
1. 한 터미널에서 서버를 실행합니다:
```
python server.py
```
2. 다른 터미널에서 클라이언트를 실행합니다:
```
python client.py
```
3. 웹브라우저에 접속해 실시간 영상을 모니터링합니다:
```
http://localhost:9000
```
4. 웹캠으로 결함을 감지할 수 있는지 확인합니다.

##

<br/>

## YOLO
yolo는 하나의 신경망을 통해서 이미지에서 바로 바운딩 박스와 클래스 확률을 예측하는 물체 감지 모델입니다.[[1]](<https://arxiv.org/abs/1506.02640>)  
1-stage 디텍터인 yolo는, 여러 형태로 미리 지정된 앵커박스를 통해 바운딩박스를 찾는 Regional Proposal과  
클래스를 분류하는 Classification이 CNN을 통해 동시에 이루어져 다른 이미지 감지 모델보다 속도가 빠른 것이 특징입니다.[[2]](<https://velog.io/@qtly_u/Object-Detection-Architecture-1-or-2-stage-detector-%EC%B0%A8%EC%9D%B4>)  
저는 엣지디바이스에서의 실시간 물체 감지에 적합한 속도, 정확도를 갖춘 최신 버전의 SOTA(state-of-the-art)모델인 yolo11[[3]](<https://docs.ultralytics.com/models/yolo11/>)을 사용했습니다.  
![performance-comparison](https://github.com/user-attachments/assets/64a285a8-ec7f-4ab1-85f1-23bf2bf15e28)


## 이미지 데이터 & Annotation
사진은 스마트폰을 사용해 찍었으며, 이미지는 3000x4000해상도의 jpg파일입니다.  
yolo모델은 학습시에 이미지 사이즈를 비율을 유지한 채 자동 변환해주며, 전처리를 자동으로 해줍니다.[[4]](<https://docs.ultralytics.com/guides/preprocessing_annotated_data/#resizing-images>) 저는 큰 차원 쪽 사이즈를 640으로 설정하였습니다.  
사진은 차 부품 결함 검사를 가장해 paint(도장 결함), dent(움푹 패임), crack(금이 감) 3가지 클래스로 나누었습니다.  
포스트잇을 사용하였으며 칼로 긁어내어 paint, 볼펜 뒷끝으로 꾹 눌러 dent, 칼로 흠집을 내어 crack을 표현하였습니다.  
총 50장의 사진을 7:2:1의 비율로 train, val, test셋으로 나눴으며, 각각 35, 10, 5장의 사진입니다.  
데이터는 [cvat](<https://app.cvat.ai/tasks>)[[5]](<https://app.cvat.ai/tasks>)을 이용하여 annotation했습니다.

## 학습
저는 윈도우환경이라서 WSL2 + Ubuntu20.04.5로 가상 리눅스 환경을 만들어 학습을 돌렸습니다.  
##### 빠른 학습 - 다음 커맨드를 실행합니다:
```
python training.py
```

##### 학습 코드 일부(training.py)  
yaml파일을 바탕으로 학습합니다.  
yolo는 imgsz 패러미터를 입력해주면 입력이미지의 비율을 유치한 채 학습시 resize해주며, 이 때 이미지의 큰쪽 차원 크기를 imgsz로 맞춥니다.  
더욱 빠른 추론을 위한 TensorRT포맷으로의 저장도 되며, 이미지 데이터셋이 300개 이상이라면 int8로의 양자화도 가능합니다.

```python
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
.
.
.
```

## yaml 파일
프로젝트 폴더 내의 yaml_creator.py로 만들거나, 다른 yaml파일을 복사후 수정해서 사용합니다.  
이 때, 데이터셋의 경로와 nc(클래스 수), names(클래스 이름)를 준비한 데이터셋에 맞게 수정해줍니다.  
```yaml
train: custom_data/train/images
val: custom_data/valid/images
test: custom_data/test/images

nc: 3
names: ['paint', 'dent', 'crack']
```
  
## fastAPI 배포 준비
server.py에 해당하는 코드 일부입니다.  
fastAPI를 통해 내가 원하는 형태로의 배포가 쉽다는걸 보여주기위해 바운딩박스의 커스터마이징이나 라벨크기 같은 요소[[6]](<https://www.kaggle.com/code/jadsherif/number-detection-using-yolov11/notebook>)를 넣어봤습니다.  
```python
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
    .
    .
    .

    results = model.predict(image)

    # yolo의 바운딩박스 데이터(텐서객체이므로 cpu메모리로 옮긴후 넘파이배열로 변환)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    
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
.
.
.
```

## 학습 결과
![results](https://github.com/user-attachments/assets/e22decdc-62c9-4497-8277-76a4bdddbdc7)

loss가 전반적으로 감소하는 성향을 보이며, 검증데이터의 loss는 노이즈가 있는걸 봐서 데이터셋이 작아서 생긴 과적합일 수도 있겠습니다.  
정밀도도 증가하고있으며, 물체 감지의 성능 측정 지표로 사용되는 mAP(mean Average Precision)도 증가하는 성향을 보이는 것을 보아,  
전반적으로 잘 학습됐다고 보여집니다.  

## 추론
파인튜닝한 모델로 학습에 사용되지 않은 unseen데이터를 실시간 감지해 보았습니다.  
실시간 감지 예시:

https://github.com/user-attachments/assets/6780c168-8037-4199-8651-183d882c1623

## 마치며
fastAPI서버를 통한 모델 서비스 배포와 클라이언트의 상호작용에 대해 공부할 수 있었던 시간이었습니다.  
오토에버에 구축된 머신비전 시스템은 공장에 있는 Edge 비전 시스템(GPU)과 비전 통합관리 시스템 + 클라우드에 있는 AI 플랫폼간의 통신이며,  
클라우드에서 모델을 배포하고, 비전 통합관리 시스템에선 받은 모델을 edge시스템으로 전달하고 모니터링하며, edge에서 실제로 inference해서 검사이미지를 다시 학습이미지로 공급하는 구조라고 알고있습니다.  
이 때 공장의 머신비전 장비는 PLC같은 산업용 장비로, 좀 더 raw한 통신을 하므로 통신계층(OSI)의 저계층에 있는 socket을 사용하며,  
프로젝트는 HTTP를 사용하므로 메시지형식의 데이터를 다루어 윗계층에 있는 websocket을 사용한다는 차이가 있다는걸 알게 되었습니다.  

한편 validation loss에서 보여진 노이즈는 yolo의 데이터 증강[[7]](<https://docs.ultralytics.com/guides/preprocessing_annotated_data/#data-augmentation-methods>)으로 해결할 수 있을 것 같습니다. 하지만 웹캠의 화질이 낮다보니 조명에 민감하고 paint같은 미세한건 잘 감지해내지 못했던건 개선할 사항으로 남았습니다.  

<br/>
  
#### References
Number Detection using YOLOV11. <https://www.kaggle.com/code/jadsherif/number-detection-using-yolov11/notebook>  
[1]. You Only Look Once: Unified, Real-Time Object Detection. <https://arxiv.org/abs/1506.02640>  
[2]. Object Detection Architecture - Difference between 1 and 2 stage detector. <https://velog.io/@qtly_u/Object-Detection-Architecture-1-or-2-stage-detector-%EC%B0%A8%EC%9D%B4>  
[3]. Overview of Ultralytics YOLO11. <https://docs.ultralytics.com/models/yolo11/>  
[4]. Preprocessing Annotated Data - Ultralytics YOLO Docs. <https://docs.ultralytics.com/guides/preprocessing_annotated_data/#resizing-images>  
[5]. Computer Vision Annotation Tool. <https://app.cvat.ai/tasks>  
[6]. Annotating Images with Bounding Boxes and Class Labels for Object Detection - Number Detection using YOLOV11. <https://www.kaggle.com/code/jadsherif/number-detection-using-yolov11/notebook>  
[7]. Data Augmentation Methods - Ultralytics YOLO Docs. <https://docs.ultralytics.com/guides/preprocessing_annotated_data/#data-augmentation-methods>

