# P1 — Specialist-advisor consults

<!-- One line per consult: - YYYY-MM-DD [agent] Q: … → conclusion [memory: slug] -->

- 2026-06-22 [hardware domain · answered by main] Q: BlueROV2 C3 (= MarineSitu C3) depth camera에 IMU 내장? → C3 사양엔 IMU 미표기. OAK-D(OV9282 stereo + IMX378 12MP) 기반이라 Bosch BMI270/BNO08x가 있을 수 있으나 미보장·미캘리브, 다수 OAK-D는 IMU 미탑재 출하. 자세 소스는 BlueROV2 Navigator의 IMU가 정식. (hardware-advisor가 아직 dispatch 불가였어 main이 직접 답함 → 이 빌드로 수정됨)
- 2026-06-22 [hardware domain · answered by main] Q: BlueROV2에 IMU 내장? → 예. Navigator 보드 ICM-20602 6축(gyro ±2000°/s, accel ±16g, ~1kHz) + MMC5983/AK09915 compass + BMP280; 외장 Bar30이 수중 depth. consumer-grade MEMS, 단독 dead-reckoning 불가(DVL/USBL 필요), 수중 magnetometer heading 신뢰 낮음.
- 2026-06-22 [hardware-advisor] Q: C3를 BlueROV2에 추가 시 2차 하드웨어 영향/BOM 주의점? → C3는 gigabit이지만 tether는 Fathom-X ~80Mbps(실측 15–50) 병목 → tether 가로질러 gigabit 불필요; 압축 stereo+color 전송 후 depth는 topside 재구성. onboard gigabit non-PoE 스위치(GigaBlox) 필요(BR 10/100 스위치 부족), penetrator/end-cap 여유 확인, 12–24V 별도 fused feed, 전방 장착 → 재-trim. 사전확정 필수: tether 길이·perception 시나리오·full-res 30FPS live 필요 여부(Fathom-X vs fiber 결정). [first live dispatch of the fixed hardware-advisor]
