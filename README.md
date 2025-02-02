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

## 이미지 데이터 & Annotation
사진은 스마트폰을 사용해 찍었으며, 이미지는 3000x4000해상도의 jpg파일입니다.  
yolo모델은 학습시에 이미지 사이즈를 비율을 유지한 채 자동 변환해주며, 전처리를 자동으로 해줍니다.[[1]](<https://docs.ultralytics.com/guides/preprocessing_annotated_data/#resizing-images>) 저는 큰 차원 쪽 사이즈를 640으로 설정하였습니다.  
사진은 차 부품 결함 검사를 가장해 paint(도장 결함), dent(움푹 패임), crack(금이 감) 3가지 클래스로 나누었습니다.  
포스트잇을 사용하였으며 칼로 긁어내어 paint, 볼펜 뒷끝으로 꾹 눌러 dent, 칼로 흠집을 내어 crack을 표현하였습니다.  
총 50장의 사진을 7:2:1의 비율로 train, val, test셋으로 나눴으며, 각각 35, 10, 5장의 사진입니다.  
데이터는 [cvat](<https://app.cvat.ai/tasks>)[[2]](<https://app.cvat.ai/tasks>)을 이용하여 annotation했습니다.

## 학습
저는 윈도우환경이라서 WSL2 + Ubuntu20.04.5로 가상 리눅스 환경을 만들어 학습을 돌렸습니다.  
다음 커맨드를 실행합니다:
```
python training.py
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
fastAPI서버를 통한 모델 서비스 배포와 클라이언트의 상호작용에 대해 공부할 수 있었던 시간이었고, 생각보다 웹브라우저의 프레임도 잘 나오는 것 같습니다.  
yolo모델의 학습자체도 잘 되는 것 같고 validation loss에서 보여진 노이즈는 yolo의 데이터 증강[[3]](<https://docs.ultralytics.com/guides/preprocessing_annotated_data/#data-augmentation-methods>)으로 해결할 수 있을 것 같습니다.  
하지만 조명 상태를 맞추기 어려웠고 웹캠의 화질이 낮다보니 paint같은 미세한건 잘 감지해내지 못했던건 개선사항으로 남았습니다.  

  
#### References
Number Detection using YOLOV11. <https://www.kaggle.com/code/jadsherif/number-detection-using-yolov11/notebook>  
[1]. Preprocessing Annotated Data - Ultralytics YOLO Docs. <https://docs.ultralytics.com/guides/preprocessing_annotated_data/#resizing-images>  
[2]. Computer Vision Annotation Tool. <https://app.cvat.ai/tasks>  
[3]. Data Augmentation Methods - Ultralytics YOLO Docs. <https://docs.ultralytics.com/guides/preprocessing_annotated_data/#data-augmentation-methods>

