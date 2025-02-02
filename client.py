import cv2
import threading
import requests
import base64
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

# 메인 event loop를 저장할 변수 (uvicorn에서 사용하는 loop)
main_event_loop = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()
    yield

# FastAPI 앱 초기화 (lifespan 이벤트 핸들러 사용)
app = FastAPI(lifespan=lifespan)

# 웹캠을 스트리밍하는 WebSocket 클라이언트 목록
websocket_clients = set()

# FastAPI 서버 URL (YOLO 처리용 서버)
FASTAPI_SERVER_URL = "http://127.0.0.1:8000/detect"

# 웹캠을 캡처하여 FastAPI 서버로 전송하고 결과를 받는 함수
def capture_and_detect():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임을 JPEG로 인코딩
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()

        # FastAPI 서버에 프레임 전송
        response = requests.post(
            FASTAPI_SERVER_URL,
            files={"file": ("image.jpg", image_bytes, "image/jpeg")}
        )

        # 디텍션 결과 수신
        if response.status_code == 200:
            detections = response.json().get("detections", [])

            # 프레임에 오버레이
            for detection in detections:
                x1, y1, x2, y2 = int(detection["x1"]), int(detection["y1"]), int(detection["x2"]), int(detection["y2"])
                confidence = detection["confidence"]
                class_name = detection["class"]
                class_color = detection["color"]
                font_scale = detection["font_scale"]
                font_thickness = detection["font_thickness"]
                box_thickness = detection["box_thickness"]

                # 텍스트 사이즈
                tw, th = cv2.getTextSize(class_name, 0, font_scale, font_thickness)[0]
                # 바운딩박스
                cv2.rectangle(frame, (x1, y1), (x2, y2), class_color, box_thickness)
                # 텍스트 배경
                cv2.rectangle(frame, (x1, y1), (x1+tw+40, y1-th-10), class_color, -1)
                cv2.putText(frame, f"{class_name} {confidence * 100:.0f}%", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

            # 프레임을 JPEG로 인코딩 후 base64 변환
            _, encoded_frame = cv2.imencode('.jpg', frame)
            encoded_base64 = base64.b64encode(encoded_frame).decode('utf-8')
            print(f"전송준비 완료. 사이즈: {len(encoded_base64)}")

            # 웹 브라우저에 실시간 스트리밍 (비동기 실행)
            if main_event_loop and main_event_loop.is_running():
                print('asyncio 루프 실행 중')
                asyncio.run_coroutine_threadsafe(send_to_web_clients(encoded_base64), main_event_loop)
            else:
                print('!!asyncio 루프 실행되지 않음')
        else:
            print(f"FastAPI 서버 요청 실패: {response.status_code}")

    cap.release()

# WebSocket을 통해 감지된 영상 데이터를 브라우저에 전송
async def send_to_web_clients(frame_base64):
    global websocket_clients
    if not websocket_clients:
        print("클라이언트가 없음.")
        return
    for ws in list(websocket_clients):
        try:
            print('브라우저에 전송 준비 완료')
            await asyncio.sleep(0.01)
            await ws.send_text(frame_base64)
            print('브라우저에 전송 완료')
        except Exception as e:
            print(f"브라우저 전송 오류: {e}")
            websocket_clients.remove(ws)

# WebSocket 엔드포인트
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_clients.add(websocket)
    print(f"클라이언트 연결됨: {websocket}")
    try:
        while True:
            msg = await websocket.receive_text()  # ping 유지 목적
            print(f"메시지 수신: {msg}")
    except Exception as e:
        print(f"websocket 오류: {e}")
    finally:
        websocket_clients.remove(websocket)
        print(f"클라이언트 연결 종료: {websocket}")

# HTML 페이지 제공 (브라우저에서 실시간 영상 보기)
@app.get("/")
async def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-Time Detection</title>
    </head>
    <body>
        <h2>edge 실시간 모니터링</h2>
        <img id="video_feed" width="640" height="480">
        <script>
            var ws = new WebSocket("ws://127.0.0.1:9000/ws");
            ws.onmessage = function(event) {
                document.getElementById("video_feed").src = "data:image/jpeg;base64," + event.data;
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# 클라이언트 실행 (FastAPI 웹 서버 및 웹캠 처리 쓰레드)
if __name__ == "__main__":
    # 웹캠을 처리하는 쓰레드를 별도로 실행
    threading.Thread(target=capture_and_detect, daemon=True).start()

    # FastAPI 실행 (웹 브라우저 스트리밍)
    uvicorn.run(app, host="0.0.0.0", port=9000)
