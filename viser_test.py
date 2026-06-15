import time
import viser

# 1. Viser 서버 실행 (기본 포트 8080)
server = viser.ViserServer(port=8080)

# 2. 3D 화면에 빨간색 박스 하나 생성하기
server.scene.add_box(
    name="/my_box",
    dimensions=(1.0, 1.0, 1.0), # 가로, 세로, 높이
    color=(255, 0, 0),          # RGB 색상 (빨간색)
    position=(0.0, 0.0, 0.5),   # Z축으로 0.5 띄움
)

print("🚀 Viser 서버가 실행되었습니다! 브라우저에서 localhost:8080 으로 접속하세요.")

# 3. 서버가 꺼지지 않고 계속 실행되도록 무한 루프 유지
while True:
    time.sleep(1.0)