# 🎵 Grooni — Discord Bot Project

## 📌 소개
Grooni는 **Discord 서버에서 AI를 활용한 학습 피드백** 기능을 제공하기 위해 만든 학습용 봇입니다.  
동아대학교 재학 중 Python 공부와 Discord API, 머신러닝 실습을 위해 제작했습니다.

---

## 🛠 기술 스택
- **언어**: Python 3.10+
- **라이브러리**: discord.py (또는 py-cord), json, csv
- **기타**: Git, VS Code

---

## 🚀 설치 & 실행 방법
1. 저장소 클론
```bash
git clone https://github.com/sopo9880/Grooni.git
cd Grooni
```

2. 가상환경 생성 및 활성화

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. 패키지 설치

```bash
pip install -r requirements.txt
```

4. 환경변수 파일 작성

* 루트 경로에 `.env` 파일 생성

```env
DISCORD_BOT_TOKEN=YOUR_TOKEN_HERE
```

5. 실행

```bash
python discord_study_feedback_bot.py
```

---

## 📂 주요 파일 구조

```
Grooni/
 ├─ discord_study_feedback_bot.py   # 봇 메인 코드
 ├─ test.py                         # 테스트 코드
 ├─ user_dataset.csv                # 사용자 데이터셋 (학습 피드백용)
 ├─ user_profiles.json              # 사용자 프로필 예시
 └─ 설계도.txt                       # 프로젝트 설계 기록
```

---

## ✨ 주요 기능

* 음악 관련 명령어 (재생, 정지, 건너뛰기 등 — 확장 예정)
* 학습 데이터 기반 간단한 피드백
* CSV/JSON 파일을 활용한 사용자 데이터 관리

---

## 📖 배운 점

* Discord API 기초와 봇 구조 이해
* JSON/CSV 데이터 처리
* Python 비동기 프로그래밍의 기본

---

## 📝 앞으로의 계획 (To-Do)

* 에러 핸들링 및 예외 처리 강화
* GUI 또는 웹 대시보드 연동
* 데이터 분석 기능 확장

---

## 📜 라이선스

이 프로젝트는 **MIT License**를 따릅니다.
