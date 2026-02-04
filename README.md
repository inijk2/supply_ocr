# 공급/선택/큐 OCR MVP (3개만 한다)

## 핵심 아이디어 (3개만 한다) OCR 버전

### (A) 서플라이 used/total을 OCR로 정확히 읽는다

1~2자리/1~2자리 포맷을 알고 있으니, 일반 OCR보다 **슬래시(/) 앵커 기반 분할 + OCR**로 견고하게 간다.

이게 되면: 막힘/오버로드파일론디폿 타이밍(증가), 유닛 생산 흔적(used 증가) 같은 뼈대가 생김.

### (B) 선택 패널 변화 순간만 캡처하고, 그 프레임에서 OCR로 선택 대상/정보 텍스트를 읽는다

1인칭에서 확실한 정보는 선택했을 때 UI가 띄워주는 텍스트/수치임.

선택 패널 ROI에서 **텍스트 라인(예: 유닛/건물 이름 영역)**을 OCR로 읽고, evidence로 저장.

### (C) 생산 큐 변화 순간만 캡처하고, 그 프레임에서 OCR/아이콘을 최소로 읽는다

MVP에서는 아이콘 분류 대신, 큐/커맨드 카드에 텍스트가 뜨는 경우는 OCR, 텍스트가 없으면 템플릿 아이콘 Top-1만(옵션).

즉, **OCR이 1순위, 아이콘은 플랜B**.

결론: 여전히 3개만 하지만, 읽기는 OCR 중심으로 바뀜.

---

## 1) 입력 (동일)

- mp4 (유튜브 다운로드는 별도 스크립트/파이프)
- 해상도 고정: `854x480` (480p)만
- 구간: `0~420초`

---

## 2) 출력 JSON (OCR 정보를 signals에 더 넣는다)

```json
{
  "version": 1,
  "segment": { "start_sec": 0, "end_sec": 420 },
  "roi_profile": "rm_854x480_v1",
  "signals": {
    "supply_series": [
      {
        "t": 0.0,
        "used": 4,
        "total": 9,
        "raw_text": "4/9",
        "conf": 0.95,
        "frame": "evidence/supply_000001.jpg"
      }
    ],
    "selection_changes": [
      {
        "t": 33.2,
        "frame": "evidence/sel_000123.jpg",
        "ocr": {
          "selected_name": { "text": "SCV", "conf": 0.72 },
          "hp_text": { "text": "45/45", "conf": 0.60 }
        }
      }
    ],
    "queue_events": [
      {
        "t": 61.0,
        "frame": "evidence/q_000220.jpg",
        "ocr": {
          "queue_text": { "text": "Marine", "conf": 0.66 }
        },
        "fallback_icon": { "item_id": "marine", "conf": 0.83 }
      }
    ]
  },
  "events": [
    {
      "t": 61.0,
      "id": "marine_started",
      "count": 1,
      "conf": 0.66,
      "evidence": ["evidence/q_000220.jpg"],
      "source": "queue_ocr"
    }
  ],
  "diagnostics": {
    "warnings": [],
    "ocr_engine": "tesseract|easyocr|paddleocr",
    "preprocess": "upscale3x+adaptive_threshold"
  }
}
```

핵심 유지: `events`는 완벽이 목표가 아니고, `signals`가 금광.

---

## 3) 모듈 구조 (OCR 포함, 그래도 단순)

구현은 **Python 중심**으로 진행 가능. (OCR/영상처리 라이브러리 호환 우선)

```
src/
  decode/
    ffmpeg_decode.ts           # 프레임 추출(저fps) + 트리거 윈도우 고fps
  roi/
    profile_480p.json
    crop.ts
  detect/
    diff_trigger.ts            # roi changed: selection_panel, production_queue
  ocr/
    preprocess.ts              # upscale/denoise/adaptive threshold (+프레임 선명도 스코어)
    read_supply.ts             # "d{1,2}/d{1,2}" 전용 파서 (OCR+슬래시 앵커)
    read_selection.ts          # 선택 패널 텍스트(OCR)
    read_queue.ts              # 큐 텍스트(OCR)
  vision_optional/
    icon_queue.ts              # (옵션) 큐 아이콘 템플릿 매칭 Top-1
  pipeline.ts
  eval/
    eval.ts
  cli.ts
```

없앤 것 그대로: `infer/rules/DP` 같은 무거운 추론은 여전히 제거.

---

## 4) 트리거 (그대로 2개, 단 최선 프레임 선택을 OCR에 맞춤)

### supply_changed(t)

- supply ROI를 `2fps`로 읽되,
- 읽을 때마다 한 프레임만 OCR하지 말고
- `t` 근처 `0.25초`에서 `7프레임` 샘플
- preprocess 후 OCR `conf`가 최대인 프레임을 채택
- 값이 바뀌면 기록 + evidence 저장

### roi_changed(t, roi=selected_panel|production_queue)

- diff가 임계값 넘으면
- `0.5초`에서 프레임 `10장` 저장
- preprocess 후 텍스트 선명도(에지/라플라시안) + OCR conf로 `1장` 선택
- 그 1장에 OCR 수행, 결과와 함께 evidence 저장

---

## 5) OCR 엔진/전처리(필수로 명시)

### 5.1 OCR 엔진 선택(우선순위)

MVP에서 중요한 건 빨리 붙고, 숫자에 강한 것:

1. PaddleOCR: 실전에서 강함(설치/의존이 다소 무거울 수 있음)
2. EasyOCR: 붙이기 쉬움, 다국어도 가능
3. Tesseract: 가벼우나, 게임 UI 작은 글씨에선 전처리/튜닝이 중요

어느 걸 쓰든, MVP에선 **영어만(SCV/Marine 등) + 숫자 중심**.

### 5.2 전처리(고정 ROI에 최적화)

- ROI crop
- 3x 업스케일(필수)
- denoise(약하게)
- adaptive threshold(필수)
- morphology close/open(상황 따라)
- (옵션) ROI 미세 정렬: / 또는 UI 테두리로 10px 보정

### 5.3 서플라이 전용 파서(중요)

일반 OCR로 `4/9`를 그냥 읽게 두지 말고:

- `/` 위치를 템플릿/shape로 찾고
- 좌/우를 분리해서 digit OCR
- 결과 조립 + sanity check(`used<=total`, 범위 제한)

이게 필수입니다.

---

## 6) 템플릿 구성 (OCR 필수로 바뀌었으니 최소화)

- digits 템플릿: OCR 실패시 fallback
- queue icons 템플릿: 선택 사항
- 텍스트가 없는 UI일 때만 Top-1 fallback

즉, 템플릿은 주력이 아니라 보험.

---

## 7) 평가 방법 (동일, + OCR 품질 로그)

- GT: 수동 라벨로 충분
- 매칭: `id 동일 + |Δt| <= 3s`
- 지표: Precision/Recall/F1 + mean |Δt|

추가로 OCR 디버깅용 통계:

- supply OCR 성공률(파싱 성공 비율)
- selection OCR non-empty 비율
- queue OCR non-empty 비율
- conf 분포

이걸 보면 어디서부터 무너지는지 바로 보입니다.

---

## 8) 구현 순서 (OCR 포함 버전)

1. 비디오 디코딩 + ROI crop + evidence 저장
2. preprocess 파이프라인(업스케일+adaptive threshold) + 선명도 스코어
3. supply 전용 OCR 파서(d{1,2}/d{1,2}) 완성
4. diff trigger(선택/큐) + 최선 프레임 선택
5. selection OCR(이름 라인만) / queue OCR(가능하면)
6. signals JSON 출력
7. events는 signals 기반으로 최소 생성(예: `queue_text==Marine` → `marine_started`)

---

## 9) ROI 캘리브레이션 (선택/큐)

선택 패널과 생산 큐 ROI는 자동으로 추론하지 않고, 한 번만 수동 캘리브레이션합니다.

```
python src/roi/calibrate.py yt_480p.mp4 --time 10 --out src/roi/profile_480p.json
```

위 스크립트는:

- 공급(supply)은 템플릿 매칭으로 자동 탐지
- selection_panel / production_queue는 드래그로 선택
- `profile_480p.json`을 갱신
```
