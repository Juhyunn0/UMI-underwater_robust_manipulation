# NUMPY314_AUDIT — python 3.14 시기 결과물 감사

**결론: 저장된 산출물 중 오염이 확인된 것은 없다.** 손상되는 함수는 두 개 찾았지만,
둘 다 그 출력을 디스크에 남긴 적이 없다.

감사일 2026-08-06 · 대상 구간 **2026-05-04 21:28 ~ 2026-08-06** (94일) ·
방법: 옛 환경 `robust314_bad`(py3.14.4) vs 새 환경 `robust`(py3.12.13)에서 같은 코드를
돌리고 산출물을 **내용 기준**으로 대조.

## 무엇이 문제였나

`robust`가 python 3.14였을 때, numpy 1.26.4는 cp314 휠이 없어 소스에서 빌드된 상태였다.
그 조합에서 **산술 연산이 결과를 피연산자 버퍼에 덮어썼다.** 반환값은 맞지만 입력이
파괴되므로, 그 변수를 뒤에서 다시 읽는 코드가 조용히 틀린다.

발동 조건은 실측으로 확정했다 (`robust314_bad`):

```
덮어써짐 :  a*b(→a), b*a(→b), a+b(→a), a*2.0(→a), 2.0*a(→a), -a(→a)
            (a+1)*b 는 a 와 b 를 모두 덮어쓴다
무사     :  np.sqrt(a) 등 단항 ufunc, 팬시 인덱싱 a[m], a.sum(), np.multiply(a,b,out=c)

크기 임계 : 256 KB  (dtype 무관 — float64 32768개, float32 65536개에서 시작)
범위     : 함수 지역변수일 때만. 모듈 전역은 재현 안 됨
형태 의존 : 같은 연산도 한 줄로 몰아 쓰면 재현되지 않는 경우가 있다
실제 오류 : 덮어써진 변수를 "그 뒤에 다시 읽을 때"만
```

원인은 numpy의 임시배열 재사용이 refcount 1인 피연산자를 버려도 되는 임시로 판단하는데,
CPython 3.14의 `LOAD_FAST_BORROW`가 지역변수를 스택에 올릴 때 refcount를 올리지 않아
**살아 있는 지역변수가 임시로 보이는** 것이다. 3.14에서는 **파라미터로 넘긴 배열도**
refcount 1로 보이므로 정적 추론이 통하지 않는다 — 그래서 판정을 전부 실행 대조로 했다.

## 감사 방법과 그 한계

1. **정적 스캔** — AST로 "함수 안에서 지역변수가 산술 피연산자로 쓰인 뒤 다시 읽히는 곳"을
   전수 추출. 182개 파일에서 **630개 후보**.
2. **실행 대조** — 같은 명령을 두 환경에서 돌리고 산출물을 **경로를 제외한 내용**으로 해시
   비교 + stdout diff.
3. **노이즈 바닥 확정** — 두 numpy 빌드는 SIMD 커널이 달라서 결과가 **최대 3.0e-16
   (≈1.4 ULP)** 까지 다르다. 이 이하 차이는 버그가 아니라 빌드 차이다. 이 값은
   `make_synthetic_capture`의 `ground_truth.json`에서 실측했다(다른 float 61개, 중앙값
   1.6e-16). **판정 기준: 상대오차 1e-14 초과여야 오염으로 본다.**

**한계**: 하드웨어(ZED 카메라, 갠트리, C3 라이브)가 있어야 도는 경로는 실행 대조를 할 수
없었다. 그 부분은 배열 크기 논거로 판정했다(아래).

## 실행 대조 결과 — 전부 동일

| 대상 | 규모 | 결과 |
|---|---|---|
| `marinegym/tests/` 13개 | — | 동일 (같은 1건 실패: `test_dobmpc` casadi 4x1/6x1, 기존 문제) |
| `verify_acados/hydro/hydro_precise/meta` | — | 동일 PASS |
| `verify_eaob` | — | 동일 FAIL (NIS 49.5 같은 값) |
| `verify_state_source` | 출력 60줄 | **radRMS/NIS/NEES 전부 동일** (벽시계만 다름) |
| `c3_camera` 테스트 4개 스위트 | 193개 | 동일 (depth_accuracy 58, host_depth 45, offline 58, option_sweep 32) |
| `umi_handheld.warp` | PNG 456개 | **바이트 동일** |
| `umi_handheld.extract_gripper_width` | — | 바이트 동일 |
| `umi_handheld.record --source synthetic` | — | 바이트 동일 |
| `tools/make_synthetic_capture` | 파일 167개 | 이미지 166개 바이트 동일; JSON만 ≤1.4 ULP 차이 → **빌드 차이** |
| `src/tools/rebuild_trajectory_html` (tagslam 플롯/HTML) | 궤적 7,547행 | **바이트 동일** |
| `calib/fov_audit --selftest` | — | 동일 |
| `gen_pool_apriltags`, `c3_dataset_check` | — | 동일 |

## 확인된 손상 2건 — 그러나 저장된 산출물 없음

### 1. `c3_camera/c3_encode_quality.py:385` `ssim()`
동일 이미지에 대해 SSIM이 1.0이어야 하는데 **7.05e8**이 나왔다. 원인은 `a*a`, `b*b`,
`a*b`가 차례로 `a`와 `b`를 덮어써서 마지막 `a*b`가 사실상 `a²·b²`를 계산한 것.
`test_encode_quality`가 옛 환경에서 4/36 실패한 것이 이 버그의 최초 발견 경로였다.

**영향 없음**: 이 도구는 `c3_camera/encode_quality/`에 결과를 쓰는데 그 디렉터리가
존재하지 않는다 — **한 번도 실행된 적이 없다.**

### 2. `c3_camera/geometry.py:100-102` `depth_to_xyz()`
```python
x = (uu - i.cx) * z / i.fx     # <- z 가 여기서 덮어써진다
y = (vv - i.cy) * z / i.fy     #    다음 줄은 망가진 z 를 쓰고
return np.stack([x, y, z], ...)  #    반환값의 Z 채널도 그것이다
```
같은 프레임(`c3_camera/datasets/dataset_20260803_162223`)으로:

| | 점 개수 | z 범위 | x 범위 |
|---|---|---|---|
| 정상 (py3.10/3.12) | 86,403 | 0.150 – 4.270 m | −2.25 – 2.38 m |
| 옛 robust (py3.14) | 42,146 | **0.054 – 109405.688 m** | −2.25 – **0.31** m |

**영향 없음**: 호출자는 셋뿐이다 — `source.py`의 라이브 미리보기,
`c3_stream.py:89 save_pointcloud`(저장된 `.ply` **0개** = 한 번도 안 씀),
`c3_depth_accuracy.py:586`(`c3_camera/depth_accuracy/` 부재 = 한 번도 실행 안 됨).

## 실행 대조를 못 한 경로 — 크기 논거로 판정

하드웨어가 필요한 15개 파일. 가장 큰 노출은 `src/tagslam_core.py`(후보 52개, 이 기간
산출물 442개)였다. 판정 근거:

- 그 52곳의 변수는 **3-벡터**(`normal`, `axis`, `up_world`), **태그 코너 점**
  (`points`, `dirs`, `air_dirs` — 태그당 4개), **LM 야코비안**(점 수 × 6),
  **프레임별 스칼라 배열**(`c_xm`, `g_xa` 등)이다.
- `robust_fit_plane`의 입력은 `tag_points`(tagslam_core.py:1844) — 수백 개.
- 이 기간 저장된 궤적 중 **가장 긴 것이 7,547행** = float64로 60 KB. 임계 256 KB의 1/4.
- 같은 모듈의 오프라인 절반(플롯·HTML, 후보 4280~6299)은 그 7,547행 궤적에서 **바이트
  동일**함을 실행으로 확인했다.

따라서 tagslam 온라인 경로는 임계를 넘길 배열이 없다. 갠트리 GUI·C3 라이브 스트림도 같다.

## 남은 위험

- 위 크기 논거는 **저장된 궤적 길이**에 근거한다. 32,768 프레임(float64 기준) 이상을
  한 배열에 담는 실행이 있었다면 재검토가 필요하다 — 현재 저장된 것 중엔 없다.
- 옛 환경은 `robust314_bad`로 보존돼 있다(2.6 GB). 나중에 의심되는 결과가 나오면
  그 환경에서 재현해 새 환경과 대조할 수 있다. 이 감사가 끝났다고 판단되면 지워도 된다.

## 재현 방법

```bash
# 발동 조건 확인
/home/bdml/miniforge3/envs/robust314_bad/bin/python -c '
import numpy as np
def f():
    a = np.full(100_000, 3.0); _ = a * a; return a[0]
print(f())        # 9.0 = 손상,  3.0 = 정상'

# 현재 환경 점검
./c3 env          # 마지막 줄 numpy elide : OK

# 임의의 스크립트를 두 환경에서 대조 (경로 제외 내용 비교)
scratchpad/diffrun.sh <label> "/tmp/out_{ENV}" <script> --out "{OUT}"
```
