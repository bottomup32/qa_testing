@echo off
echo ===================================
echo    Streamlit App Startup Script
echo ===================================
echo.
echo Local IP: 192.168.0.169
echo Access URLs:
echo   Local:     http://localhost:8501
echo   Network:   http://192.168.0.169:8501
echo.
echo Starting Streamlit App...
echo.
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
pause 