# 딥러닝 실험실 · Easy! 딥러닝 인터랙티브 복습 사이트

혁펜하임의 **Easy! 딥러닝** 커리큘럼(1강~8강)을 직접 슬라이더로 조작하며 복습할 수 있는
인터랙티브 학습 사이트입니다. 빌드 도구 없이 순수 HTML/CSS/JS로만 만들어져 있어서
GitHub Pages에 바로 올릴 수 있습니다.

## 포함된 내용

- **1강** 딥러닝 개요 · 경사하강법 vs Newton's Method · 자기지도학습 · 강화학습
- **2강** 인공신경망 · 선형회귀 · 경사하강법(2D) · 웨이트 초기화
- **3강** 극한과 입실론-델타 · 미분 · 연쇄법칙 · 편미분과 그래디언트
- **4강** 랜덤변수와 확률분포 · 평균과 분산 · 균등/정규분포
- **5강** Linear activation의 통찰 · 역전파(Backpropagation)
- **6강** Sigmoid · 로지스틱회귀 · Cross-Entropy/MLE · 소프트맥스회귀
- **7강** Universal Approximation Theorem
- **8강** Vanishing Gradient/ReLU · BatchNorm/LayerNorm · Loss Landscape · 과적합/데이터증강 · Dropout · Regularization/MAP

## 로컬에서 미리보기

폴더 안에서 아무 정적 서버나 실행하면 됩니다. 예:

```bash
python3 -m http.server 8000
# 브라우저에서 http://localhost:8000 접속
```

(그냥 `index.html`을 더블클릭해서 열어도 대부분 동작하지만, 일부 브라우저는 로컬 파일 접근 정책 때문에
간단한 서버로 여는 걸 권장합니다.)

## GitHub Pages로 배포하기

1. 새 GitHub 레포지토리를 만들고 이 폴더의 파일들(`index.html`, `style.css`, `app.js`)을 그대로 올립니다.
2. 레포 → **Settings → Pages** 로 이동합니다.
3. **Source**를 `Deploy from a branch`로, 브랜치는 `main` (또는 `master`), 폴더는 `/ (root)`로 선택 후 저장합니다.
4. 몇 분 뒤 `https://<사용자명>.github.io/<레포이름>/` 주소로 접속하면 사이트가 열립니다.

레포 루트가 아니라 `docs/` 폴더에 두고 싶다면, 파일들을 `docs/`로 옮기고 Pages 설정에서 폴더를
`/docs`로 지정하면 됩니다.

## 구조

```
index.html   페이지 골격 + 오실로스코프 헤더 + 사이드바
style.css    디자인 시스템 (신호처리 실험실 테마)
app.js       모든 개념 데이터 + 인터랙티브 데모 로직 (강의별로 섹션 구분되어 있음)
```

## 콘텐츠 더 추가하고 싶을 때

`app.js` 안에 강의별로 `topicX_Y` 객체들이 있고, 각 객체는 `{id, title, desc, render(root)}` 형태입니다.
새 항목을 추가하려면 비슷한 패턴으로 객체를 만들고 해당 `CHAPTER_N.topics` 배열에 추가하면 사이드바에
자동으로 반영됩니다.
