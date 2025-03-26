# API 테스트 자동화 도구

이 프로젝트는 API 엔드포인트를 테스트하고 결과를 비교하는 자동화 도구입니다.

## 주요 기능

- 단일 질문 테스트
  - 단일 데이터셋 테스트
  - 여러 데이터셋 비교 테스트
  - API 직접 테스트 (curl 명령어 제공)

- 배치 테스트
  - CSV/Excel 파일에서 질문 일괄 처리
  - 단일/다중 데이터셋 비교
  - 실시간 결과 모니터링
  - 중간 결과 저장 및 다운로드

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/bottomup32/qa_testing.git
cd qa_testing
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
source venv/bin/activate     # Linux/Mac
```

3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

## 실행 방법

### Windows
```bash
run_app.bat
```

### Linux/Mac
```bash
streamlit run app.py
```

## 프로젝트 구조

```
qa_testing/
├── app.py              # 메인 애플리케이션 코드
├── run_app.bat         # Windows 실행 스크립트
├── requirements.txt    # Python 패키지 의존성
├── .streamlit/        # Streamlit 설정
│   └── config.toml    # Streamlit 설정 파일
└── README.md          # 프로젝트 문서
```

## 사용 방법

1. 단일 테스트
   - "단일 테스트" 탭 선택
   - 테스트 모드 선택 (단일/비교)
   - 질문 입력
   - 추가 API 파라미터 설정 (선택사항)
   - 테스트 실행

2. 배치 테스트
   - "배치 테스트" 탭 선택
   - CSV/Excel 파일 업로드
   - 테스트 모드 및 데이터셋 선택
   - 배치 크기 및 타임아웃 설정
   - 테스트 실행

## API 설정

- API 엔드포인트, 인증 토큰, 배포 이름을 사이드바에서 설정 가능
- "Other" 데이터셋의 경우 사용자 정의 인덱스/메타데이터 파일 경로 지정 가능

## 결과 출력

- 실시간 테스트 결과 표시
- Excel 형식으로 결과 다운로드
- 오류 로그 확인

## 라이선스

MIT License 