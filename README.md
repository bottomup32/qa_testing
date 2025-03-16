# API 테스트 자동화 도구

이 애플리케이션은 CSV 또는 Excel 파일에 포함된 질문들을 API에 배치 호출하여 응답에서 특정 정보를 추출 후 엑셀 파일로 저장합니다.

## 주요 기능

- CSV 또는 Excel 파일에서 질문 목록 로드
- 여러 데이터셋 옵션 지원 (Hans, Nam, Yeom)
- 각 질문에 대한 API 호출 및 응답 처리
- "Embedding search top 10"과 "LLM Top 3" 정보 추출
- 여러 데이터셋 간 비교 기능
- 결과를 엑셀 파일로 다운로드
- 타임아웃 및 오류 복구 기능 지원
- 중간 결과 저장 및 다운로드
- 단일 질문 테스트 및 결과 비교 기능

## 설치 방법

```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
streamlit run app.py
```

## 사용 방법

### 배치 테스트

1. 웹 인터페이스에서 '배치 테스트' 탭 선택
2. 단일 데이터셋 또는 데이터셋 비교 모드 선택
3. 테스트할 데이터셋 선택
4. `question` 컬럼이 포함된 CSV 또는 Excel 파일 업로드
5. 배치 크기, 타임아웃, 재시도 횟수 설정
6. 'API 테스트 실행' 버튼 클릭
7. 테스트 완료 후 결과 엑셀 파일 다운로드

### 단일 테스트

1. 웹 인터페이스에서 '단일 테스트' 탭 선택
2. 단일 데이터셋 또는 여러 데이터셋 비교 모드 선택
3. 테스트할 데이터셋 선택
4. 질문 입력
5. '테스트 실행' 버튼 클릭
6. 결과 확인

## 입력 파일 요구사항

- CSV 또는 Excel 파일 형식
- `question` 컬럼 포함

## 결과 파일 형식

- Excel 파일 (xlsx)
- 질문, 상태, Embedding Search Top 10, LLM Top 3 정보 포함
- 데이터셋 비교 시 각 데이터셋별 결과 시트 포함

## 필요한 데이터

애플리케이션을 실행하기 위해서는 다음 경로에 FAISS 인덱스 및 메타데이터 파일이 필요합니다:

```
faiss_index/embedding dataset-0314-Hans-0313_v20250314_222213.index
faiss_index/embedding dataset-0314-Hans-0313_v20250314_222213.csv
faiss_index/embedding dataset-0314-Nam-0313_v20250315_002250.index
faiss_index/embedding dataset-0314-Nam-0313_v20250315_002250.csv
faiss_index/embedding dataset-0314-Yeom-0313_v20250315_002545.index
faiss_index/embedding dataset-0314-Yeom-0313_v20250315_002545.csv
```

## 오류 처리

- API 타임아웃 발생 시 자동 재시도 (최대 재시도 횟수 설정 가능)
- 오류 발생 시 로그 기록 및 표시
- 배치 처리 중 중간 결과 저장

## 주의사항

- 대량의 질문을 처리할 경우 API 서버의 부하와 속도 제한을 고려하여 배치 크기와 딜레이를 적절히 설정하세요.
- 중요한 테스트의 경우 중간 결과를 주기적으로 다운로드하여 데이터 손실을 방지하세요. 