import cv2
import depthai as dai
import numpy as np


DEPTHAI_MAJOR_VERSION = int(dai.__version__.split(".", maxsplit=1)[0])


def create_pipeline(is_extended):
    """
    모드에 맞춰 최적화된 파이프라인을 생성합니다.
    """
    pipeline = dai.Pipeline()

    # 1. 모노 카메라(좌/우) 노드 생성
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    depth = pipeline.create(dai.node.StereoDepth)

    # DepthAI 3.x streams node outputs directly to a host queue. XLinkOut is
    # only needed by the DepthAI 2.x API.
    if DEPTHAI_MAJOR_VERSION < 3:
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("depth")

    # 2. 해상도 설정 (800p가 가장 정밀도가 높습니다)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # 3. 모드별 최적 설정 적용
    if is_extended:
        # [Extended Mode] 초근접(20cm~) 감지 가능, FPS 소폭 하락(50), Subpixel 불가
        print(">>> [모드 변경] Extended Disparity 켜짐 (초근접 모드) / 50 FPS")
        monoLeft.setFps(50)
        monoRight.setFps(50)
        depth.setExtendedDisparity(True)
        depth.setSubpixel(False)
    else:
        # [Normal Mode] 일반(37cm~), 최고 정밀도, 부드러운 깊이 표현, 최대 FPS(60)
        print(">>> [모드 변경] Normal Mode + Subpixel 켜짐 (고정밀 모드) / 60 FPS")
        monoLeft.setFps(60)
        monoRight.setFps(60)
        depth.setExtendedDisparity(False)
        depth.setSubpixel(True)

    # 설정 통일 (LR Check는 깊이 노이즈를 줄여주므로 항상 켜두는 것이 좋습니다)
    depth.setLeftRightCheck(True)

    # 4. 노드 연결
    monoLeft.out.link(depth.left)
    monoRight.out.link(depth.right)
    if DEPTHAI_MAJOR_VERSION < 3:
        depth.disparity.link(xout.input)

    # Subpixel mode stores fractional disparity values, so its maximum is not
    # simply 96. Reading it from the configured node keeps visualization valid
    # in both normal and extended modes.
    max_disparity = depth.initialConfig.getMaxDisparity()

    return pipeline, depth.disparity, max_disparity


def show_depth_stream(q_depth, is_extended, max_disparity):
    """Display frames until the user quits or requests a mode change."""
    print("카메라 실행 중... (단축키: 'e' = 모드 전환, 'q' = 종료)")

    while True:
        in_depth = q_depth.get()
        if in_depth is None:
            continue

        frame = in_depth.getFrame()

        # Disparity를 8-bit 영상으로 변환한 뒤 컬러맵을 적용합니다.
        frame_norm = np.clip(
            frame.astype(np.float32) * (255.0 / max_disparity), 0, 255
        ).astype(np.uint8)
        frame_color = cv2.applyColorMap(frame_norm, cv2.COLORMAP_JET)

        mode_text = (
            "Extended Mode (Near, <40cm)"
            if is_extended
            else "Normal Mode (Subpixel, >40cm)"
        )
        cv2.putText(
            frame_color,
            mode_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        cv2.imshow("OAK-D Depth Test", frame_color)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return "quit"
        if key == ord("e"):
            return "toggle"


def main():
    is_extended = False

    while True:
        pipeline, disparity_output, max_disparity = create_pipeline(is_extended)

        if DEPTHAI_MAJOR_VERSION >= 3:
            # DepthAI 3.x: create the host queue from the node output and start
            # the pipeline directly.
            with pipeline:
                q_depth = disparity_output.createOutputQueue(
                    maxSize=4, blocking=False
                )
                pipeline.start()
                action = show_depth_stream(q_depth, is_extended, max_disparity)
        else:
            # DepthAI 2.x: XLinkOut exposes the named queue through Device.
            with dai.Device(pipeline) as device:
                q_depth = device.getOutputQueue(
                    name="depth", maxSize=4, blocking=False
                )
                action = show_depth_stream(q_depth, is_extended, max_disparity)

        cv2.destroyAllWindows()
        if action == "quit":
            print("프로그램을 종료합니다.")
            return

        is_extended = not is_extended


if __name__ == "__main__":
    main()
