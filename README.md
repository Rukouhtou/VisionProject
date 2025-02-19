##
[![SVG Banners](https://svg-banners.vercel.app/api?type=glitch&text1=비전AI%20🤖&width=800&height=150)](https://github.com/Akshay090/svg-banners)
# VisionProject
해당 프로젝트는 비전AI 모델 서비스를 fastAPI서버를 사용해 client로 배포하는 내용을 담고 있습니다.

웹캠이달린 클라이언트쪽에서는 WebSocket을 통하여 서버와 영상 송수신 + 웹브라우저를 통한 모니터링,  
서버쪽에서는 fastAPI를 사용하여, 받은 영상을 파인튜닝된 yolo모델로 처리후 클라이언트로 다시 전송하는 역할을 합니다.
  
## 목차
1. [Prerequisites](#1.-Prerequisites)
2. [Install](#2.-Install)
3. [How to use](#3.-How-to-use)
4. [YOLO](#4.-YOLO)
5. [이미지 데이터 & Annotation](#5.-이미지-데이터-&-Annotation)
6. [학습](#6.-학습)
7. [yaml 파일](#7.-yaml-파일)
8. [fastAPI 배포 준비](#8.-fastAPI-배포-준비)
9. [학습 결과](#9.-학습-결과)
10. [데이터 증강](#10.-데이터-증강)
11. [증강 후 학습 결과들](#11.-증강-후-학습-결과들)
12. [추론](#12.-추론)
13. [마치며](#13.-마치며)

<br><br>

## 1. Prerequisites
- Python 3.10.16
- pip
- conda
- webcam

## 2. Install
conda 가상환경을 만든 뒤 다음 커맨드를 이용하여 [yolo11](<https://github.com/ultralytics/ultralytics>)을 설치합니다:  
```
pip install ultralytics
```
그런 다음 requirements를 설치합니다:
```
pip install -r requirements.txt
```

## 3. How to use
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

## 4. YOLO
yolo는 하나의 신경망을 통해서 이미지에서 바로 바운딩 박스와 클래스 확률을 예측하는 물체 감지 모델입니다.[[1]](<https://arxiv.org/abs/1506.02640>)  
1-stage 디텍터인 yolo는, 여러 형태로 미리 지정된 앵커박스를 통해 바운딩박스를 찾는 Regional Proposal과  
클래스를 분류하는 Classification이 CNN을 통해 동시에 이루어져 다른 이미지 감지 모델보다 속도가 빠른 것이 특징입니다.[[2]](<https://velog.io/@qtly_u/Object-Detection-Architecture-1-or-2-stage-detector-%EC%B0%A8%EC%9D%B4>)  
저는 엣지디바이스에서의 실시간 물체 감지에 적합한 속도, 정확도를 갖춘 최신 버전의 SOTA(state-of-the-art)모델인 yolo11[[3]](<https://docs.ultralytics.com/models/yolo11/>)을 사용했습니다.  
![performance-comparison](https://github.com/user-attachments/assets/64a285a8-ec7f-4ab1-85f1-23bf2bf15e28)


## 5. 이미지 데이터 & Annotation
사진은 스마트폰을 사용해 찍었으며, 이미지는 3000x4000해상도의 jpg파일입니다.  
yolo모델은 학습시에 이미지 사이즈를 비율을 유지한 채 자동 변환해주며, 전처리를 자동으로 해줍니다.[[4]](<https://docs.ultralytics.com/guides/preprocessing_annotated_data/#resizing-images>) 여기서는 큰 차원 쪽 사이즈를 640으로 설정하였습니다.  
사진은 차 부품 결함 검사를 가정해 paint(도장 결함), dent(움푹 패임), crack(금이 감) 3가지 클래스로 나누었습니다.  
포스트잇을 사용하였으며 칼로 긁어내어 paint, 볼펜 뒷끝으로 꾹 눌러 dent, 칼로 흠집을 내어 crack을 표현하였습니다.  
총 50장의 사진을 7:2:1의 비율로 train, val, test셋으로 나눴으며, 각각 35, 10, 5장의 사진입니다.  
데이터는 [cvat](<https://app.cvat.ai/tasks>)[[5]](<https://app.cvat.ai/tasks>)을 이용하여 annotation했습니다.

## 6. 학습
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

## 7. yaml 파일
프로젝트 폴더 내의 yaml_creator.py로 만들거나, 다른 yaml파일을 복사후 수정해서 사용합니다.  
이 때, 데이터셋의 경로와 nc(클래스 수), names(클래스 이름)를 준비한 데이터셋에 맞게 수정해줍니다.  
```yaml
train: custom_data/train/images
val: custom_data/valid/images
test: custom_data/test/images

nc: 3
names: ['paint', 'dent', 'crack']
```
  
## 8. fastAPI 배포 준비
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

## 9. 학습 결과
![results](https://github.com/user-attachments/assets/e22decdc-62c9-4497-8277-76a4bdddbdc7)

loss가 전반적으로 감소하는 성향을 보이며, 검증데이터의 loss는 노이즈가 있는걸 봐서 데이터셋이 작아서 생긴 과적합일 수도 있겠습니다.  
정밀도도 증가하고있으며, 물체 감지의 성능 측정 지표로 사용되는 mAP(mean Average Precision)도 증가하는 성향을 보이는 것을 보아,  
적은 데이터셋 치고는 전반적으로 잘 학습됐다고 보여집니다.  
여기서 이 변동폭이 큰 학습양상을 잡기위해, 추가적으로 데이터를 증강하여 데이터셋 수 자체를 늘려 학습해보겠습니다.  

## 10. 데이터 증강
빠르고 편리한 증강이 가능항 albumentations라이브러리를 설치합니다.  
```
pip install albumentations
```

다음을 실행하면 datasets/custom_data에 augmented폴더가 생성됩니다.  
```python
python augmentation.py
```

##### 증강코드 일부(augmentation.py)
라벨정보도 불러와 넘파이로 저장했으며, 카메라로 촬영한 데이터셋에 효과적인 증강방법들을 적용해봤습니다.
```python
from itertools import combinations
from pathlib import Path
import albumentations as A
import cv2
import numpy as np
.
.
.
        labels = load_label(src_label_path / f'{name}.txt')
        
        if len(labels) > 0:
            bboxes = labels[:, 1:]
            class_labels = labels[:, 0]

            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

            # 공간적 정보가 변하는 spatial-level transform을 넣어놨으므로,
            # 증강 이후의 bbox정보를 가져오고 yolo의 라벨형식으로 합쳐주도록함
            if len(augmented['bboxes']) > 0:
                aug_bboxes = np.array(augmented['bboxes'])
                aug_labels = np.column_stack([augmented['class_labels'], aug_bboxes])
            else:
                aug_labels = np.array([])
            
            cv2.imwrite(str(dst_img_path / f'aug_{aug_idx}_{name}.jpg'), augmented['image'])
            # savetxt함수 정의를 보면 기본포맷이 %.18e이므로, 
            # yolo의 라벨에 맞게 클래스id는 int, bbox정보는 float으로 저장해줌
            np.savetxt(dst_label_path / f'aug_{aug_idx}_{name}.txt', aug_labels, fmt='%d %.6f %.6f %.6f %.6f')

def main():
    # 카메라로 촬영한 데이터셋이므로, 카메라 플래시나 노이즈등에 대응할만한 transform들 생성
    augmentations = [
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Rotate(limit=5, p=0.5),
        A.Sharpen(alpha=(0.2, 0.3), lightness=(0.7, 1.0), p=0.5),
        A.RandomShadow(num_shadows_limit=(1,2), shadow_dimension=5, p=0.5),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.5),
    ]

    # SomeOf를 이용하여 매번 랜덤으로 2개의 증강을 적용하게 함(호출마다 랜덤)
    # 반복으로 같은 조합이 나올 수도 있지만, transform들의 수치가 랜덤이라 완전히 같지는 않음
    # 원본 데이터셋의 이미지는 50장이므로, 총 50 + 50*6 = 350장이 만들어짐
    transform = A.Compose([
        A.SomeOf(augmentations, n=2, p=0.7, replace=False),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
.
.
.
```

## 11. 증강 후 학습 결과들
runs/detect/  
<details><summary>train15:</summary>  
해당 시도에선 itertools의 combinations를 이용,  
5가지 트랜스폼의 모든 조합(5C1=5, 5C2=10, 5C3=10, 5C4=5, 5C5=1)인 총 31가지를 이미지 한 장당 적용하여,  
50장 * 31 + 50장(원본) = 1550장 + 50장(원본) = 1600장으로 학습을 돌렸습니다.  
    
- 학습 파라미터
    
    ```
    epochs=300
    ```

결과: 58에포크에 best결과가 나왔고, 얼리스타핑으로 158에포크에 학습이 조기종료되었습니다.  
눈여겨 볼 만한 것은, 클래스 불균형을 나타내는 val/dfl_loss(Distribution Focal Loss)가 증가하고 있다는 것입니다.  
즉, 클래스간 등장 빈도수의 차이가 커져서 학습에 지장이 생겼다는 의미이므로, 적은 원본 데이터셋 대비 과도하게 늘린 데이터셋을 줄여보도록 하겠습니다.  
![results](https://github.com/user-attachments/assets/ad5d8ddf-ee15-476a-a805-ab8332492603)

</details>

<details><summary>train16:</summary>  
해당 시도에선 combinations의 사용을 버리고, albumentations의 SomeOf를 사용하여 5개의 트랜스폼중 랜덤하게 2개를 조합하도록 했습니다.  
그리고 한 장당 6번의 증강이 적용되도록하여,  
50장 * 6 + 50장(원본) = 350장으로 학습을 돌렸습니다.  

- 학습 파라미터

    ```
    epochs=300
    ```

결과: 96에포크에 best결과가 나왔고, 얼리스타핑으로 196에포크에 학습이 조기종료되었습니다.  
여전히 변동폭이 크지만 val/dfl_loss가 줄어드는 양상을 보아, 원본 데이터셋 대비 과도하게 많았던 증강 데이터를 줄인건 옳은 방향인 것 같습니다.  
![results](https://github.com/user-attachments/assets/1ae02868-d3ab-407c-94a8-3df6214ec213)

</details>

<details><summary>train17:</summary>  
해당 시도에선 augmentation.py는 그대로 두어 증강은 이전과 같이하고, 학습 파라미터를 변경해보았습니다.  

- 학습 파라미터

    ```
    epochs=100,    # best가 다 100 이전에 나왔었으므로 줄입니다
    batch=32,    # 한 번의 배치에 편향된 증강패턴이 등장하는걸 막아서 학습 안전성을 높여보려는 취지입니다
    lr0=0.005    # 원본 데이터셋의 수가 적으므로 학습률을 줄여서 과적합을 막아보려는 취지입니다
    ```

결과: 얼리스타핑 안하고 100에폭 다 학습하였으며, mAP50(IoU(Intersection over Union) 0.5 이상에서 측정한 평균 정밀도)가 0.8을 넘겼으므로 객체감지 자체는 어느정도 기준점을 넘긴 괜찮은 모델로 학습된 것 같습니다.  
하지만 mAP50-95(높은 IoU에서 정확한 바운딩박스를 찾는 능력)가 0.4에서 머문다는 것은, 바운딩박스의 정확도는 부족하다는걸 알 수 있습니다.  
![results](https://github.com/user-attachments/assets/610bd043-38bd-48d5-aaa7-b6accf54ae91)

</details>

<details><summary>train18:</summary>  
해당 시도에선 augmentation.py에서 증강된 이미지크기를 1280으로 키웠고, RandomScale증강을 추가하여 모델의 성능을 올려보려고 시도해보았습니다.  

- 학습 파라미터
    ```
    epochs=100,
    batch=32,
    lr0=0.005,
    imgsz=1040    # 1280으로 두면 초반부터 vram이 터졌습니다...
    ```

결과: 학습중 터졌습니다.  

</details>

<details><summary>train19:</summary>  
해당 시도에선 train18과 똑같은 조건에서, imgsz만 960으로 낮춰 vram초과를 막아보려 했습니다.  

- 학습 파라미터
    ```
    epochs=100,
    batch=32,
    lr0=0.005,
    imgsz=960
    ```

결과: 역시나 터졌습니다.  

</details>

<details><summary>train20:</summary>  
해당 시도에선 객체크기 변형으로 bbox학습을 어렵게 만든 것 같은 RandomScale증강을 제거하고, train17의 증강조건으로 돌아갔습니다.  
validation loss들이 대체적으로 잘나오고, mAP50이 0.8을 넘었으며, mAP50-95값만 개선하면 되는 train17의 결과에 학습 파라미터만 변형해보았습니다.  

- 학습 파라미터
    ```
    epochs=100,
    batch=32,
    lr0=0.005,
    warmup_epochs=5,    # validation loss들이 초기에 큰 변동폭을 보이므로 초기의 lr증가를 늦추기위한 목적으로 증가시켰습니다 (기본값: 3)
    weight_decay=0.001    # 역시 초기의 변동이 큰 업데이트를 방지하려는 목적으로 증가시켰습니다 (기본값: 0.0005)
    ```

결과: mAP50과 mAP50-95값이 train17때보다도 낮게 나왔고, validation/bbox_loss의 진동폭은 여전히 크게 나왔습니다.  
![results](https://github.com/user-attachments/assets/7514ded9-665d-4ab2-9720-d0d09ed47285)

</details>

<details><summary>train21:</summary>  
해당 시도에선 train20과 같은 조건에서, 학습 파라미터에 cos_lr=True만 추가시켜 후반부의 큰 진동폭을 잡아 수렴시키려고 해봤습니다.  

- 학습 파라미터
    ```
    epochs=100,
    batch=32,
    lr0=0.005,
    warmup_epochs=5,
    weight_decay=0.001,
    cos_lr=True
    ```

결과: 후반부(75~100에포크쯤)에서 학습률이 부족한지 언더피팅의 양상을 보입니다.  
![results](https://github.com/user-attachments/assets/0b08d757-69a9-4ca7-813d-f99857e155d3)

</details>

train22:  
마지막으로 train21과 같은 조건에, 학습 파라미터에 lrf=0.1만 추가해 yolo모델의 최종 학습률을 높여서 후반부에서의 언더피팅을 완화해 보겠습니다.  

- 학습 파라미터
    ```
    epochs=100,
    batch=32,
    lr0=0.005,
    lrf=0.1,    # final learning rate을 증가시켜서 후반부의 학습률을 높여 수렴을 돕기위한 목적입니다 (기본값: 0.01)
    warmup_epochs=5,
    weight_decay=0.001,
    cos_lr=True
    ```

결과: val/bbox_loss가 아쉽지만 전체적인 후반부에서의 언더피팅이 어느정도 완화된 모습을 보이며, mAP50값이 0.8대로 괜찮은 성능에, mAP50-95값도 0.5근처로 처음보다 상승한 모델로 학습이 되었습니다. 
![results](https://github.com/user-attachments/assets/4048787e-28e2-4c7d-9a5f-653355d1a20f)


## 12. 추론
파인튜닝한 모델로 학습에 사용되지 않은 unseen데이터를 실시간 감지해 보았습니다.  
실시간 감지 예시:

https://github.com/user-attachments/assets/6780c168-8037-4199-8651-183d882c1623

## 13. 마치며
데이터 증강[[7]](<https://docs.ultralytics.com/guides/preprocessing_annotated_data/#data-augmentation-methods>)과 학습 파라미터 조절로 처음보다는 어느정도 일반화된 모델을 학습할 수 있었습니다. 하지만 usb로 연결된 웹캠의 화질이 낮다보니 조명에 민감하고 paint같은 미세한걸 잘 감지해내지 못했던건 edge단에서의 예상치 못한 일이었습니다.  

fastAPI서버를 통한 모델 서비스 배포와 클라이언트의 상호작용에 대해 공부할 수 있었던 시간이었습니다.  
오토에버에 구축된 머신비전 시스템은 공장에 있는 Edge 비전 시스템(GPU)과 비전 통합관리 시스템 + 클라우드에 있는 AI 플랫폼간의 통신으로 이루어지며,  
클라우드에서 모델을 배포하고, 비전 통합관리 시스템에선 받은 모델을 edge시스템으로 전달하고 모니터링하며, edge에서 실제로 inference해서 검사이미지를 다시 학습이미지로 공급하는 구조라고 알고있습니다.  
이 때 공장의 머신비전 장비는 PLC같은 산업용 장비로, 좀 더 raw한 통신을 하므로 통신계층(OSI)의 저계층에 있는 socket을 사용하며,  
프로젝트는 HTTP상에서의 실시간 통신으로 영상재생을 하고, 메시지형식의 데이터를 다루므로 윗계층에 있는 WebSocket을 사용한다는 차이가 있습니다.


<br/>
  
#### References
[1]. You Only Look Once: Unified, Real-Time Object Detection. <https://arxiv.org/abs/1506.02640>  
[2]. Object Detection Architecture - Difference between 1 and 2 stage detector. <https://velog.io/@qtly_u/Object-Detection-Architecture-1-or-2-stage-detector-%EC%B0%A8%EC%9D%B4>  
[3]. Overview of Ultralytics YOLO11. <https://docs.ultralytics.com/models/yolo11/>  
[4]. Preprocessing Annotated Data - Ultralytics YOLO Docs. <https://docs.ultralytics.com/guides/preprocessing_annotated_data/#resizing-images>  
[5]. Computer Vision Annotation Tool. <https://app.cvat.ai/tasks>  
[6]. Annotating Images with Bounding Boxes and Class Labels for Object Detection - Number Detection using YOLOV11. <https://www.kaggle.com/code/jadsherif/number-detection-using-yolov11/notebook>  
[7]. Data Augmentation Methods - Ultralytics YOLO Docs. <https://docs.ultralytics.com/guides/preprocessing_annotated_data/#data-augmentation-methods>

