const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "최재혁";
pres.title = "학위논문 파이프라인 — 현황, 문제점, 해결 방안";

const C = {
  bg: "0F172A", bgLight: "1E293B", card: "1E293B", cardLight: "334155",
  text: "F1F5F9", textMuted: "94A3B8", accent: "38BDF8",
  green: "4ADE80", red: "F87171", orange: "FB923C", white: "FFFFFF",
  cbg: "F8FAFC", ccard: "FFFFFF", ct: "1E293B", cm: "64748B",
};

const mkS = () => ({ type: "outer", blur: 4, offset: 2, angle: 135, color: "000000", opacity: 0.1 });

// ===== SLIDE 1: Title =====
let s1 = pres.addSlide();
s1.background = { color: C.bg };
s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
s1.addText("학위논문 파이프라인 개발", {
  x: 0.8, y: 1.0, w: 8.4, h: 1, fontSize: 38, fontFace: "Arial Black", color: C.white, bold: true, margin: 0
});
s1.addText("현황 · 문제점 · 해결 방안", {
  x: 0.8, y: 1.9, w: 8.4, h: 0.6, fontSize: 22, fontFace: "Calibri", color: C.accent, margin: 0
});

// Pipeline flow with status colors
const steps = [
  { name: "SAM-3D\nBody", ok: true },
  { name: "MHR→\nSMPL", ok: true },
  { name: "gsplat\n정제", ok: true },
  { name: "가상\n마커", ok: true },
  { name: "OpenSim\nIK", ok: false },
  { name: "Vicon\n비교", ok: false },
];
steps.forEach((st, i) => {
  const x = 0.6 + i * 1.5;
  s1.addShape(pres.shapes.RECTANGLE, {
    x: x, y: 3.0, w: 1.3, h: 0.8,
    fill: { color: st.ok ? "064E3B" : "7F1D1D" },
    line: { color: st.ok ? C.green : C.red, width: 1.5 }
  });
  s1.addText(st.name, {
    x: x, y: 3.0, w: 1.3, h: 0.8, fontSize: 10, fontFace: "Calibri",
    color: st.ok ? C.green : C.red, align: "center", valign: "middle", margin: 0
  });
  if (i < steps.length - 1) {
    s1.addText("→", {
      x: x + 1.3, y: 3.15, w: 0.2, h: 0.5, fontSize: 16, color: C.textMuted, align: "center", valign: "middle", margin: 0
    });
  }
});

s1.addText("현재: OpenSim IK 단계에서 마커 매칭 + 모델 스케일링 문제", {
  x: 0.8, y: 4.1, w: 8.4, h: 0.35, fontSize: 13, fontFace: "Calibri", color: C.red, italic: true, margin: 0
});
s1.addText("2026.04.16  |  최재혁  |  동아대학교 스포츠의학과", {
  x: 0.8, y: 5.0, w: 8.4, h: 0.3, fontSize: 10, fontFace: "Calibri", color: C.textMuted, margin: 0
});

// ===== SLIDE 2: What Works =====
let s2 = pres.addSlide();
s2.background = { color: C.cbg };
s2.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.85, fill: { color: C.bg } });
s2.addText("해결된 것들", {
  x: 0.8, y: 0.12, w: 8, h: 0.6, fontSize: 28, fontFace: "Arial Black", color: C.green, bold: true, margin: 0
});

const solved = [
  { t: "gsplat differentiable rendering", d: "999프레임 · 28분 · A100 · loss 5.8% 개선" },
  { t: "BSM 105-마커 SMPL 정점 매핑", d: "SMPL2AddBiomechanics bsm_markers.yaml 활용" },
  { t: "TRC 파일 생성", d: "init (정제 전) + refined (정제 후) 222프레임" },
  { t: "OpenSim 4.5.2 설치", d: "Rajagopal 전신 모델 (39 DOF, 66 마커) 로딩 확인" },
  { t: "Ablation 데이터 확보", d: "body_pose_init + body_pose_refined 동시 저장" },
];
solved.forEach((item, i) => {
  const y = 1.0 + i * 0.85;
  s2.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: y, w: 8.4, h: 0.75, fill: { color: C.ccard }, shadow: mkS() });
  s2.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: y, w: 0.06, h: 0.75, fill: { color: C.green } });
  s2.addText(item.t, { x: 1.1, y: y + 0.05, w: 7.8, h: 0.3, fontSize: 14, fontFace: "Calibri", color: C.ct, bold: true, margin: 0 });
  s2.addText(item.d, { x: 1.1, y: y + 0.38, w: 7.8, h: 0.3, fontSize: 11, fontFace: "Calibri", color: C.cm, margin: 0 });
});

// ===== SLIDE 3: Problem 1 + Solution =====
let s3 = pres.addSlide();
s3.background = { color: C.cbg };
s3.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.85, fill: { color: C.bg } });
s3.addText("문제 1 + 해결: body_pose_params 형식 불일치", {
  x: 0.8, y: 0.12, w: 8.4, h: 0.6, fontSize: 20, fontFace: "Arial Black", color: C.red, margin: 0
});

// Problem (left)
s3.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.0, w: 4.3, h: 2.8, fill: { color: "FEF2F2" }, shadow: mkS() });
s3.addText("문제", { x: 0.7, y: 1.05, w: 1, h: 0.3, fontSize: 12, fontFace: "Calibri", color: C.red, bold: true, margin: 0 });
s3.addText([
  { text: "SAM-3D-Body 출력:\n", options: { bold: true, fontSize: 12, breakLine: true } },
  { text: "  body_pose_params: (133,)\n", options: { fontSize: 11, color: C.red, breakLine: true } },
  { text: "  → MHR 127관절 자체 포맷\n\n", options: { fontSize: 10, breakLine: true } },
  { text: "SMPL 기대값:\n", options: { bold: true, fontSize: 12, breakLine: true } },
  { text: "  body_pose: (69,)  23관절×3\n\n", options: { fontSize: 11, color: "16A34A", breakLine: true } },
  { text: "133 ≠ 69  직접 대응 불가!", options: { fontSize: 13, color: C.red, bold: true } },
], { x: 0.7, y: 1.4, w: 3.9, h: 2.2, fontFace: "Consolas", color: C.ct, margin: 0 });

// Arrow
s3.addText("→", { x: 4.6, y: 2.0, w: 0.5, h: 0.5, fontSize: 28, color: C.accent, align: "center", valign: "middle", margin: 0 });

// Solution (right)
s3.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.0, w: 4.3, h: 2.8, fill: { color: "F0FDF4" }, shadow: mkS() });
s3.addText("해결", { x: 5.4, y: 1.05, w: 1, h: 0.3, fontSize: 12, fontFace: "Calibri", color: "16A34A", bold: true, margin: 0 });
s3.addText([
  { text: "7시점 reproj fitting 결과를\n초기값으로 사용\n\n", options: { bold: true, fontSize: 12, breakLine: true } },
  { text: "SAM-3D-Body body_pose (X)\n", options: { fontSize: 11, color: C.red, breakLine: true } },
  { text: "        ↓ 교체\n", options: { fontSize: 11, breakLine: true } },
  { text: "reproj fitting SMPL (O)\n", options: { fontSize: 11, color: "16A34A", breakLine: true } },
  { text: "  → Rh, Th, poses, shapes\n", options: { fontSize: 10, breakLine: true } },
  { text: "  → 222프레임, 153.5px reproj", options: { fontSize: 10 } },
], { x: 5.4, y: 1.4, w: 3.9, h: 2.2, fontFace: "Consolas", color: C.ct, margin: 0 });

// Result
s3.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.0, w: 9, h: 0.7, fill: { color: C.ccard }, shadow: mkS() });
s3.addText([
  { text: "결과: ", options: { bold: true } },
  { text: "v1 SD 120~170° → v2 SD 9~26° (안정화), 내부 상관 r=0.79~0.96", options: {} },
], { x: 0.7, y: 4.1, w: 8.6, h: 0.5, fontSize: 13, fontFace: "Calibri", color: C.ct, margin: 0 });

// ===== SLIDE 4: Problem 2 + Solution =====
let s4 = pres.addSlide();
s4.background = { color: C.cbg };
s4.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.85, fill: { color: C.bg } });
s4.addText("문제 2 + 해결: 관절각도 산출 방식", {
  x: 0.8, y: 0.12, w: 8.4, h: 0.6, fontSize: 20, fontFace: "Arial Black", color: C.red, margin: 0
});

// Three failed approaches
const approaches = [
  { name: "v2: 정점 기반 JCS", result: "left-handed, sign flip", r: "0.49", ok: false },
  { name: "v3: 관절 기반 JCS", result: "개선, 하지만 불안정", r: "0.80", ok: false },
  { name: "v4: body_pose 오일러", result: "안정, Vicon과 불일치", r: "0.96*", ok: false },
];
approaches.forEach((a, i) => {
  const x = 0.5 + i * 3.1;
  s4.addShape(pres.shapes.RECTANGLE, { x: x, y: 1.0, w: 2.9, h: 1.6, fill: { color: "FEF2F2" }, shadow: mkS() });
  s4.addShape(pres.shapes.RECTANGLE, { x: x, y: 1.0, w: 2.9, h: 0.35, fill: { color: "DC2626" } });
  s4.addText(a.name, { x: x, y: 1.0, w: 2.9, h: 0.35, fontSize: 11, fontFace: "Calibri", color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });
  s4.addText(a.result, { x: x + 0.1, y: 1.45, w: 2.7, h: 0.4, fontSize: 10, fontFace: "Calibri", color: C.ct, margin: 0 });
  s4.addText("r = " + a.r, { x: x + 0.1, y: 1.9, w: 2.7, h: 0.3, fontSize: 14, fontFace: "Calibri", color: C.red, bold: true, margin: 0 });
  s4.addText("*내부 상관", { x: x + 0.1, y: 2.2, w: 2.7, h: 0.2, fontSize: 8, fontFace: "Calibri", color: C.cm, margin: 0 });
});

// Arrow down
s4.addText("↓", { x: 4.5, y: 2.7, w: 1, h: 0.4, fontSize: 24, color: C.accent, align: "center", margin: 0 });

// Solution
s4.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.2, w: 9, h: 2.0, fill: { color: "F0FDF4" }, shadow: mkS() });
s4.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.2, w: 9, h: 0.4, fill: { color: "16A34A" } });
s4.addText("해결: SAM4DCap 방식 — OpenSim IK Solver", {
  x: 0.7, y: 3.22, w: 8.6, h: 0.36, fontSize: 14, fontFace: "Calibri", color: C.white, bold: true, margin: 0
});

// Solution flow
const solFlow = [
  { name: "SMPL\n6,890 정점", c: C.accent },
  { name: "BSM 105\n가상 마커", c: C.accent },
  { name: "TRC\n파일", c: C.accent },
  { name: "OpenSim\nIK Solver", c: "16A34A" },
  { name: "ISB 표준\n관절각도", c: "16A34A" },
];
solFlow.forEach((sf, i) => {
  const x = 0.8 + i * 1.7;
  s4.addShape(pres.shapes.RECTANGLE, {
    x: x, y: 3.8, w: 1.5, h: 0.7,
    fill: { color: C.ccard }, line: { color: sf.c, width: 1.5 }
  });
  s4.addText(sf.name, {
    x: x, y: 3.8, w: 1.5, h: 0.7, fontSize: 9, fontFace: "Calibri",
    color: C.ct, align: "center", valign: "middle", margin: 0
  });
  if (i < solFlow.length - 1) {
    s4.addText("→", { x: x + 1.5, y: 3.95, w: 0.2, h: 0.4, fontSize: 14, color: C.cm, align: "center", valign: "middle", margin: 0 });
  }
});

s4.addText("Vicon Plug-in Gait과 동일한 ISB 표준으로 각도 산출 → 사과 대 사과 비교 가능", {
  x: 0.8, y: 4.7, w: 8.4, h: 0.3, fontSize: 11, fontFace: "Calibri", color: "16A34A", bold: true, margin: 0
});

// ===== SLIDE 5: Problem 3 + Solution (Marker Matching) =====
let s5 = pres.addSlide();
s5.background = { color: C.cbg };
s5.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.85, fill: { color: C.bg } });
s5.addText("문제 3 + 해결: 마커 이름 매칭", {
  x: 0.8, y: 0.12, w: 8.4, h: 0.6, fontSize: 20, fontFace: "Arial Black", color: C.orange, margin: 0
});

// Problem
s5.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.0, w: 4.3, h: 2.4, fill: { color: "FEF2F2" }, shadow: mkS() });
s5.addText("BSM 마커 (TRC)", { x: 0.7, y: 1.05, w: 2, h: 0.3, fontSize: 12, fontFace: "Calibri", color: C.red, bold: true, margin: 0 });
s5.addText([
  { text: "LSHO  → 어깨\n", options: { breakLine: true } },
  { text: "LELB  → 팔꿈치\n", options: { breakLine: true } },
  { text: "LFWT  → 골반 전방\n", options: { breakLine: true } },
  { text: "LKNE  → 무릎\n", options: { breakLine: true } },
  { text: "LANK  → 발목", options: {} },
], { x: 0.7, y: 1.4, w: 3.9, h: 1.8, fontSize: 11, fontFace: "Consolas", color: C.ct, margin: 0 });

s5.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.0, w: 4.3, h: 2.4, fill: { color: "FEF2F2" }, shadow: mkS() });
s5.addText("Rajagopal 모델", { x: 5.4, y: 1.05, w: 2, h: 0.3, fontSize: 12, fontFace: "Calibri", color: C.red, bold: true, margin: 0 });
s5.addText([
  { text: "LACR  → 견봉\n", options: { breakLine: true } },
  { text: "LLEL  → 외측 팔꿈치\n", options: { breakLine: true } },
  { text: "LASI  → 전상장골극\n", options: { breakLine: true } },
  { text: "LLFC  → 외측 대퇴과\n", options: { breakLine: true } },
  { text: "LLMAL → 외측 복사뼈", options: {} },
], { x: 5.4, y: 1.4, w: 3.9, h: 1.8, fontSize: 11, fontFace: "Consolas", color: C.ct, margin: 0 });

// Arrow
s5.addText("≠ 이름 다름!", { x: 4.3, y: 1.8, w: 1.4, h: 0.5, fontSize: 14, color: C.red, bold: true, align: "center", valign: "middle", margin: 0 });

// Solution
s5.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.6, w: 9, h: 1.6, fill: { color: "F0FDF4" }, shadow: mkS() });
s5.addText("해결: BSM → Rajagopal 이름 변환 테이블", {
  x: 0.7, y: 3.65, w: 4, h: 0.3, fontSize: 13, fontFace: "Calibri", color: "16A34A", bold: true, margin: 0
});
s5.addText([
  { text: "LSHO → LACR    LELB → LLEL    LFWT → LASI\n", options: { breakLine: true } },
  { text: "LKNE → LLFC    LANK → LLMAL   LWRA → LFAradius\n", options: { breakLine: true } },
  { text: "... 총 38개 매칭 (66개 모델 마커 중 58% 커버)", options: {} },
], { x: 0.7, y: 4.0, w: 8.6, h: 0.8, fontSize: 11, fontFace: "Consolas", color: C.ct, margin: 0 });
s5.addText("남은 이슈: TRC 파일 파싱 시 탭 구분자 오류 → 수정 필요 (1일)", {
  x: 0.7, y: 4.85, w: 8.6, h: 0.25, fontSize: 11, fontFace: "Calibri", color: C.orange, margin: 0 });

// ===== SLIDE 6: Problem 4 + Solution (Scaling) =====
let s6 = pres.addSlide();
s6.background = { color: C.cbg };
s6.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.85, fill: { color: C.bg } });
s6.addText("문제 4 + 해결: 모델 스케일링", {
  x: 0.8, y: 0.12, w: 8.4, h: 0.6, fontSize: 20, fontFace: "Arial Black", color: C.orange, margin: 0
});

// Before
s6.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.0, w: 4.3, h: 2.2, fill: { color: "FEF2F2" }, shadow: mkS() });
s6.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.0, w: 4.3, h: 0.35, fill: { color: "DC2626" } });
s6.addText("스케일링 없이 IK", { x: 0.5, y: 1.0, w: 4.3, h: 0.35, fontSize: 12, fontFace: "Calibri", color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });
s6.addText([
  { text: "기본 모델: 170cm\n", options: { fontSize: 11, breakLine: true } },
  { text: "피험자:    179.7cm, 82kg\n\n", options: { fontSize: 11, breakLine: true } },
  { text: "knee_angle_l:  0.0 ± 0.0°\n", options: { fontSize: 10, color: C.red, breakLine: true } },
  { text: "arm_flex_r:    0.0 ± 0.0°\n", options: { fontSize: 10, color: C.red, breakLine: true } },
  { text: "→ range limit 포화!", options: { fontSize: 11, color: C.red, bold: true } },
], { x: 0.7, y: 1.45, w: 3.9, h: 1.6, fontFace: "Consolas", color: C.ct, margin: 0 });

// After
s6.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.0, w: 4.3, h: 2.2, fill: { color: "F0FDF4" }, shadow: mkS() });
s6.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.0, w: 4.3, h: 0.35, fill: { color: "16A34A" } });
s6.addText("스케일링 후 IK (목표)", { x: 5.2, y: 1.0, w: 4.3, h: 0.35, fontSize: 12, fontFace: "Calibri", color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });
s6.addText([
  { text: "ScaleTool로 체형 맞춤:\n", options: { fontSize: 11, breakLine: true } },
  { text: "segment 길이 × scale_factor\n\n", options: { fontSize: 11, breakLine: true } },
  { text: "knee_angle_l:  실제 ROM\n", options: { fontSize: 10, color: "16A34A", breakLine: true } },
  { text: "arm_flex_r:    투구 동작\n", options: { fontSize: 10, color: "16A34A", breakLine: true } },
  { text: "→ Vicon과 비교 가능", options: { fontSize: 11, color: "16A34A", bold: true } },
], { x: 5.4, y: 1.45, w: 3.9, h: 1.6, fontFace: "Consolas", color: C.ct, margin: 0 });

// How to solve
s6.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.4, w: 9, h: 1.8, fill: { color: C.ccard }, shadow: mkS() });
s6.addText("해결 방법 (2가지)", { x: 0.7, y: 3.45, w: 4, h: 0.3, fontSize: 13, fontFace: "Calibri", color: C.accent, bold: true, margin: 0 });

s6.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 3.85, w: 4, h: 1.1, fill: { color: "EFF6FF" } });
s6.addText([
  { text: "방법 A: OpenSim ScaleTool\n", options: { bold: true, fontSize: 12, breakLine: true } },
  { text: "마커 쌍 간 거리 비율로\nbody segment 길이 자동 조정\n", options: { fontSize: 10, breakLine: true } },
  { text: "→ 로컬 실행, 완전 제어", options: { fontSize: 10, color: C.accent } },
], { x: 0.9, y: 3.9, w: 3.6, h: 1.0, fontFace: "Calibri", color: C.ct, margin: 0 });

s6.addShape(pres.shapes.RECTANGLE, { x: 5.0, y: 3.85, w: 4.3, h: 1.1, fill: { color: "FFF7ED" } });
s6.addText([
  { text: "방법 B: AddBiomechanics 웹\n", options: { bold: true, fontSize: 12, breakLine: true } },
  { text: "TRC 업로드 → 자동 스케일링 +\nIK + 역동역학까지 한번에\n", options: { fontSize: 10, breakLine: true } },
  { text: "→ addbiomechanics.org", options: { fontSize: 10, color: C.orange } },
], { x: 5.2, y: 3.9, w: 3.9, h: 1.0, fontFace: "Calibri", color: C.ct, margin: 0 });

// ===== SLIDE 7: Full Solution Pipeline =====
let s7 = pres.addSlide();
s7.background = { color: C.bg };
s7.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
s7.addText("전체 해결 파이프라인", {
  x: 0.8, y: 0.2, w: 8.4, h: 0.6, fontSize: 28, fontFace: "Arial Black", color: C.white, bold: true, margin: 0
});

// Full pipeline with numbers
const pipeline = [
  { n: "1", t: "TRC 마커 이름\n파싱 수정", d: "BSM→Rajagopal\n38개 매칭", day: "Day 1", c: C.accent },
  { n: "2", t: "OpenSim\nScaleTool", d: "피험자 체형\n모델 스케일링", day: "Day 2", c: C.accent },
  { n: "3", t: "OpenSim\nIK 재실행", d: "39 DOF\n관절각도 산출", day: "Day 2", c: C.accent },
  { n: "4", t: "SFC 시간\n정렬", d: "force plate 기반\n투구 구간 동기화", day: "Day 3", c: C.green },
  { n: "5", t: "CMC/ICC\nRMSE", d: "Vicon 비교\n8개 변수", day: "Day 3-4", c: C.green },
];
pipeline.forEach((p, i) => {
  const x = 0.3 + i * 1.9;
  // Number circle
  s7.addShape(pres.shapes.OVAL, { x: x + 0.55, y: 1.0, w: 0.6, h: 0.6, fill: { color: p.c } });
  s7.addText(p.n, { x: x + 0.55, y: 1.0, w: 0.6, h: 0.6, fontSize: 20, fontFace: "Arial Black", color: C.bg, align: "center", valign: "middle", margin: 0 });
  // Card
  s7.addShape(pres.shapes.RECTANGLE, { x: x, y: 1.8, w: 1.7, h: 1.6, fill: { color: C.cardLight } });
  s7.addText(p.t, { x: x, y: 1.85, w: 1.7, h: 0.6, fontSize: 11, fontFace: "Calibri", color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });
  s7.addText(p.d, { x: x, y: 2.5, w: 1.7, h: 0.5, fontSize: 9, fontFace: "Calibri", color: C.textMuted, align: "center", valign: "middle", margin: 0 });
  s7.addText(p.day, { x: x, y: 3.1, w: 1.7, h: 0.25, fontSize: 9, fontFace: "Calibri", color: p.c, align: "center", margin: 0 });
  // Arrow
  if (i < pipeline.length - 1) {
    s7.addText("→", { x: x + 1.7, y: 2.2, w: 0.2, h: 0.5, fontSize: 16, color: C.textMuted, align: "center", valign: "middle", margin: 0 });
  }
});

// Bottom: what we get
s7.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.8, w: 9, h: 1.5, fill: { color: C.cardLight } });
s7.addText("최종 산출물", { x: 0.7, y: 3.85, w: 3, h: 0.3, fontSize: 14, fontFace: "Calibri", color: C.accent, bold: true, margin: 0 });
s7.addText([
  { text: "Ablation Table: gsplat 정제 전/후 CMC 비교 (논문 핵심 기여)\n", options: { breakLine: true, fontSize: 12 } },
  { text: "8개 변수: knee flex, trunk tilt×3, shoulder abd/horiz/rot, elbow flex\n", options: { breakLine: true, fontSize: 11 } },
  { text: "통계: CMC ≥ 0.85, ICC ≥ 0.80, RMSE < 7° (목표)\n", options: { breakLine: true, fontSize: 11 } },
  { text: "그래프: Bland-Altman, 시간 정규화 파형 비교", options: { fontSize: 11 } },
], { x: 0.7, y: 4.2, w: 8.6, h: 1.0, fontFace: "Calibri", color: C.text, margin: 0 });

// ===== SLIDE 8: Timeline =====
let s8 = pres.addSlide();
s8.background = { color: C.bg };
s8.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
s8.addText("남은 타임라인", {
  x: 0.8, y: 0.2, w: 8, h: 0.6, fontSize: 28, fontFace: "Arial Black", color: C.white, bold: true, margin: 0
});

const tl = [
  { day: "Day 1", task: "TRC 마커 매칭 수정 + 파싱 오류 해결", st: "urgent" },
  { day: "Day 2", task: "모델 스케일링 + IK 재실행", st: "urgent" },
  { day: "Day 3-4", task: "Vicon 비교 + 통계 (CMC, ICC, RMSE, Bland-Altman)", st: "normal" },
  { day: "Week 2", task: "논문 III장 파이프라인 + IV장 결과 재작성", st: "normal" },
  { day: "Week 3-4", task: "지도교수 리뷰 → 수정 → 본심사 제출", st: "final" },
];

s8.addShape(pres.shapes.LINE, { x: 1.5, y: 1.1, w: 0, h: 3.4, line: { color: C.cardLight, width: 2 } });
tl.forEach((item, i) => {
  const y = 1.1 + i * 0.7;
  const dc = item.st === "urgent" ? C.red : item.st === "final" ? C.green : C.accent;
  s8.addShape(pres.shapes.OVAL, { x: 1.35, y: y + 0.05, w: 0.3, h: 0.3, fill: { color: dc } });
  s8.addText(item.day, { x: 1.8, y: y, w: 1.2, h: 0.35, fontSize: 13, fontFace: "Calibri", color: dc, bold: true, margin: 0 });
  s8.addText(item.task, { x: 3.0, y: y, w: 6.5, h: 0.35, fontSize: 14, fontFace: "Calibri", color: C.text, margin: 0 });
});

s8.addText("목표: 5~6월 본심사 제출", {
  x: 0.8, y: 4.8, w: 8.4, h: 0.4, fontSize: 16, fontFace: "Calibri", color: C.accent, bold: true, align: "center", margin: 0
});

// Save
const outPath = "/Users/choejaehyeog/3dgs_to_gart_textbook/.claude/worktrees/youthful-jepsen/data/pipeline_issues.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("Created: " + outPath);
});
