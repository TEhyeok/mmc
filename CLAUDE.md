# CLAUDE.md — 3DGS-to-GART Textbook Project

## Project Overview

3D Gaussian Splatting 기반 인체 재구성 + 야구 투구 생체역학 분석 교재 프로젝트.
LaTeX (XeLaTeX) 기반 한국어 교과서 시리즈.

## Documents

| File | Topic |
|------|-------|
| `main.tex` | Part 1: NeRF → 3DGS → SMPL → GART 기초 이론 |
| `part2_motion_analysis.tex` | Part 2: 마커리스 모션캡처 + 보행 분석 |
| `part3_pitching.tex` | Part 3: 7시점 다시점 야구 투구 분석 |
| `part0_camera_math.tex` | 카메라 수학 완전 가이드 (한국어 정리명) |
| `part0_colmap_theory.tex` | COLMAP 이론 해설 (한국어 정리명) |
| `part0_progress_report.tex` | 개발 진행 보고서 |
| `part4_data_analysis.tex` | 데이터 분석 |
| `part4_openpose_smpl_fitting.tex` | OpenPose → SMPL ���팅 가이드 |
| `part5_constraints_setup.tex` | 제약조건 분석 + 개발 세팅 |
| `part5_dev_setup.tex` | 개발 환경 A-to-Z 매뉴얼 |
| `part5_research_proposal.tex` | 연구 제안서 |
| `part6_opensim_ik_viewer.tex` | Part 6: Rajagopal 2016 IK 파이프라인 + Three.js 3D 뷰어 |
| `fitting_guide.tex` | SMPL 피팅 실전 가이드 & 오답노트 |
| `methodology_survey.tex` | SMPL-free 방법론 서베이 (독립 프리앰블) |
| `vitpose_cam3_analysis.tex` | ViTPose cam3 분석 보고서 (article 클래스, 독립) |

## Build

XeLaTeX 필수 (`fontspec` + `kotex` 사용).

```bash
make all          # 전체 빌드
make main.pdf     # 개별 빌드
make clean        # 임시 파일 제거
```

## File Structure

```
preamble/           ← 공유 프리앰블 (10개 파일이 사용)
  packages.tex      ← 공통 패키지 + geometry
  colors.tex        ← 전체 색상 정의
  theorems.tex      ← 영문 theorem 환경 (Definition, Theorem, ...)
  tcolorboxes.tex   ← 전체 tcolorbox 환경 (~25개)
  commands.tex      ← 수학 매크로 (벡터, 행렬, 단위)
  listings.tex      ← listings 패키지 + lstset
  formatting.tex    ← header/footer + titleformat
```

### Preamble Rules

- 새 패키지/색상/tcolorbox/커맨드 추가 시 → `preamble/` 파일 수정
- Part 전용 패키지는 `\input{preamble/packages}` 바로 아래에 추가
- `part0_camera_math.tex`, `part0_colmap_theory.tex`는 한국어 정리명 사용 → `theorems.tex`, `tcolorboxes.tex`를 쓰지 않고 자체 정의
- `vitpose_cam3_analysis.tex`는 `article` 클래스 → 공유 프리앰블 미사용
- `methodology_survey.tex`는 다른 geometry → 공유 프리앰블 미사용

## Data

```
data/
  openpose/         ← OpenPose 2D/3D 결과
  vitpose/          ← ViTPose 2D/3D 결과
  smpl_fitting/     ← SMPL 파라미터 (.npy, .npz)
  segments/         ← 세그멘테이션 결과
  da3_calibration/  ← DA3 캘리브레이션
  easymocap_result/ ← EasyMoCap 결과
colmap/             ← COLMAP 워크스페이스 + 보정 보고서
pitching_pipeline/  ← Python 파이프라인 코드
```

## Writing Conventions

- 본문: 한국어 + 영어 기술용어 혼용
- tcolorbox 환경 적극 활용 (keyidea, summary, biomechbox, injurybox 등)
- 수학 매크로: `\vx`, `\mR`, `\vtheta`, `\RR` 등 (`preamble/commands.tex` 참조)
- PDF는 git에 포함 (의도적)
