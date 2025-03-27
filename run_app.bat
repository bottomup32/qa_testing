@echo off
echo API 테스트 자동화 도구 실행 스크립트
echo ============================

REM Python 가상환경이 있는지 확인
if exist "venv" (
    echo 기존 가상환경 사용
) else (
    echo 새로운 가상환경 생성
    python -m venv venv
)

REM 가상환경 활성화
call venv\Scripts\activate

REM 필요한 패키지 설치
echo 필요한 패키지 설치 중...
pip install -r requirements.txt

REM Streamlit 앱 실행
echo Streamlit 앱 실행 중...
streamlit run app.py --server.address localhost --server.port 8501

pause 