import streamlit as st
import pandas as pd
import requests
import json
import time
import os
from io import BytesIO
import tempfile
import xlsxwriter
from datetime import datetime

# 앱 제목 설정
st.set_page_config(page_title="API 테스트 자동화 도구", layout="wide")

# 앱 스타일 설정
st.markdown("""
<style>
    .main-header {
        font-size: 30px;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 20px;
        font-weight: bold;
        color: #004D40;
        margin-top: 20px;
    }
    .info-text {
        font-size: 16px;
        color: #37474F;
    }
    .success-text {
        color: #2E7D32;
        font-weight: bold;
    }
    .error-text {
        color: #D32F2F;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">API 테스트 자동화 도구</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-text">
CSV 또는 Excel 파일에서 질문을 로드하고 API에 배치로 요청하거나 단일 질문을 테스트할 수 있습니다.
</div>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'temp_results' not in st.session_state:
    st.session_state.temp_results = None
if 'temp_filename' not in st.session_state:
    st.session_state.temp_filename = None
# 중지 플래그 및 실시간 결과 초기화
if 'stop_processing' not in st.session_state:
    st.session_state.stop_processing = False
if 'live_results' not in st.session_state:
    st.session_state.live_results = None
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

# API 설정 데이터 (사이드바보다 먼저 정의)
API_DATASETS = {
    "Hans": {
        "index": "faiss_index/embedding dataset-0314-Hans-0313_v20250314_222213.index",
        "metadata": "faiss_index/embedding dataset-0314-Hans-0313_v20250314_222213.csv"
    },
    "Nam": {
        "index": "faiss_index/embedding dataset-0314-Nam-0313_v20250315_002250.index",
        "metadata": "faiss_index/embedding dataset-0314-Nam-0313_v20250315_002250.csv"
    },
    "Yeom": {
        "index": "faiss_index/embedding dataset-0314-Yeom-0313_v20250315_002545.index",
        "metadata": "faiss_index/embedding dataset-0314-Yeom-0313_v20250315_002545.csv"
    }
}

# API 엔드포인트와 인증 데이터 (제공된 curl 명령과 동일하게 업데이트)
API_ENDPOINT = "https://ai-human-chatbot-roasu.koreacentral.inference.ml.azure.com/score"
API_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer BV3RjJz66FNP82LSkw6SU1905E57UgOMizCuAe3y63atc5Wwj5HXJQQJ99BCAAAAAAAAAAAAINFRAZML1o4N",
    "azureml-model-deployment": "ai-human-chatbot-roasu-4"
}

# 사이드바 설정
with st.sidebar:
    st.markdown('<div class="sub-header">API 테스트 자동화 도구</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-text">
    이 도구는 다양한 API 설정으로 질문을 테스트하고 결과를 비교할 수 있습니다.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<div class="sub-header">API 정보</div>', unsafe_allow_html=True)
    st.code("""
엔드포인트: https://ai-human-chatbot-roasu.koreacentral.inference.ml.azure.com/score
헤더:
  - Content-Type: application/json
  - Authorization: Bearer BV3RjJz6...
  - azureml-model-deployment: ai-human-chatbot-roasu-4
    """)
    
    st.markdown("---")
    
    st.markdown('<div class="sub-header">데이터셋 정보</div>', unsafe_allow_html=True)
    for dataset_name, info in API_DATASETS.items():
        with st.expander(f"{dataset_name} 데이터셋"):
            st.markdown(f"**인덱스 파일:** `{info['index']}`")
            st.markdown(f"**메타데이터 파일:** `{info['metadata']}`")
    
    st.markdown("---")
    
    st.markdown('<div class="sub-header">사용 방법</div>', unsafe_allow_html=True)
    with st.expander("배치 테스트 방법"):
        st.markdown("""
        1. '배치 테스트' 탭 선택
        2. 테스트할 데이터셋 선택 또는 여러 데이터셋 비교 모드 선택
        3. CSV 또는 Excel 파일 업로드 (첫 번째 컬럼을 질문으로 사용)
        4. 배치 크기 및 타임아웃 설정
        5. API 테스트 실행 버튼 클릭
        6. 테스트 완료 후 결과 파일 다운로드
        """)
    
    with st.expander("단일 테스트 방법"):
        st.markdown("""
        1. '단일 테스트' 탭 선택
        2. 단일 데이터셋 또는 비교 모드 선택
        3. 질문 입력
        4. 테스트 실행 버튼 클릭
        5. 결과 확인
        """)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="info-text" style="font-size: 12px; text-align: center;">
    © 2025 API 테스트 자동화 도구<br>
    버전 1.0.0
    </div>
    """, unsafe_allow_html=True)

# API 요청 함수 개선
def call_api(query, dataset_name, timeout=10, max_retries=3, show_debug=False):
    """API 호출 및 응답 처리 함수"""
    dataset_info = API_DATASETS[dataset_name]
    
    # payload 구성 - curl 명령 형식에 맞춤
    payload = {
        "user_query": query,
        "top_k": "10",
        "index_path": dataset_info["index"],
        "metadata_path": dataset_info["metadata"]
    }
    
    # 디버깅용: 요청 페이로드와 헤더 출력 (show_debug가 True인 경우)
    if show_debug:
        st.write("API 요청 정보:")
        st.json({
            "endpoint": API_ENDPOINT,
            "headers": API_HEADERS,
            "payload": payload
        })
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            # 요청 전송
            response = requests.post(API_ENDPOINT, headers=API_HEADERS, json=payload, timeout=timeout)
            
            # 응답 코드 확인
            if show_debug:
                st.write(f"API 응답 상태 코드: {response.status_code}")
                st.write(f"응답 헤더: {dict(response.headers)}")
            
            if response.status_code == 200:
                try:
                    # 응답 파싱
                    json_response = response.json()
                    
                    # 디버깅: 응답 확인
                    if show_debug:
                        st.write("원본 API 응답 내용:")
                        st.json(json_response)
                    
                    return json_response
                except Exception as e:
                    if show_debug:
                        st.error(f"API 응답 파싱 오류: {str(e)}")
                        st.write("응답 텍스트:", response.text[:500])  # 처음 500자만 표시
                    return {"error": f"API 응답 파싱 오류: {str(e)}"}
            elif response.status_code == 429:  # Rate limit 오류
                retry_count += 1
                wait_time = min(2 ** retry_count, 30)  # 지수 백오프
                if show_debug:
                    st.warning(f"Rate limit 초과, {wait_time}초 후 재시도 ({retry_count}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                if show_debug:
                    st.error(f"API 요청 실패: {response.status_code}")
                    st.write("응답 텍스트:", response.text[:500])  # 처음 500자만 표시
                return {"error": f"API 요청 실패: {response.status_code} - {response.text[:200]}"}
        except requests.exceptions.Timeout:
            retry_count += 1
            if retry_count >= max_retries:
                return {"error": f"API 요청 타임아웃 (최대 재시도 횟수 초과)"}
            wait_time = min(2 ** retry_count, 30)
            if show_debug:
                st.warning(f"요청 타임아웃, {wait_time}초 후 재시도 ({retry_count}/{max_retries})")
            time.sleep(wait_time)
        except Exception as e:
            if show_debug:
                st.error(f"API 요청 중 예외 발생: {str(e)}")
            return {"error": f"API 요청 중 오류 발생: {str(e)}"}
    
    return {"error": "API 요청 실패: 최대 재시도 횟수 초과"}

# 응답에서 필요한 정보 추출 함수 개선
def extract_info(response, show_debug=False):
    """API 응답에서 필요한 정보 추출"""
    if "error" in response:
        return {"embedding_search_top10": response["error"], "embedding_search_top10_with_score": response["error"], "llm_top1": response["error"]}
    
    try:
        # intent_data 객체 확인 (중첩 구조 처리)
        if "intent_data" in response:
            intent_data = response["intent_data"]
        else:
            intent_data = response  # 중첩되지 않은 경우
            
        # 응답 키 확인 (디버깅)
        if show_debug:
            st.write("응답 키:", list(intent_data.keys()) if isinstance(intent_data, dict) else "응답이 dict 형식이 아님")
        
        # Top 10 Embedding Search Intent 추출
        matches = intent_data.get("matches", [])
        
        if show_debug:
            st.write(f"matches 개수: {len(matches)}")
            if matches:
                st.write("첫 번째 match 항목 구조:")
                st.json(matches[0])
        
        embedding_search_top10 = []
        embedding_search_top10_with_score = []
        
        for i, match in enumerate(matches[:10]):
            # intent와 score 필드 확인
            intent = match.get("intent", "")
            score = match.get("score", 0)
            
            if intent:  # 의도가 있는 경우
                embedding_search_top10.append(intent)
                embedding_search_top10_with_score.append(f"{i+1}. \"{intent}\" {score:.4f}")
        
        # LLM Top 1 추출
        reranked_matches = intent_data.get("reranked_matches", [])
        
        if show_debug:
            st.write(f"reranked_matches 개수: {len(reranked_matches)}")
            if reranked_matches:
                st.write("첫 번째 reranked_match 항목 구조:")
                st.json(reranked_matches[0])
        
        llm_top1 = ""
        
        if reranked_matches and len(reranked_matches) > 0:
            llm_top1 = reranked_matches[0].get("answer", "")
        
        # 결과 생성 - 줄바꿈을 HTML <br> 태그로 변환하여 표에서 줄바꿈이 보이도록 함
        result = {
            "embedding_search_top10": ", ".join(embedding_search_top10) if embedding_search_top10 else "결과 없음",
            "embedding_search_top10_with_score": "<br>".join(embedding_search_top10_with_score) if embedding_search_top10_with_score else "결과 없음",
            "llm_top1": llm_top1 if llm_top1 else "결과 없음"
        }
        
        if show_debug:
            st.write("추출된 결과:")
            st.json(result)
        
        return result
    except Exception as e:
        if show_debug:
            st.error(f"응답 데이터 처리 중 오류 발생: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        return {"embedding_search_top10": f"정보 추출 실패: {str(e)}", 
                "embedding_search_top10_with_score": f"정보 추출 실패: {str(e)}", 
                "llm_top1": f"정보 추출 실패: {str(e)}"}

# 탭 생성
tab1, tab2 = st.tabs(["배치 테스트", "단일 테스트"])

# 배치 테스트 탭
with tab1:
    st.markdown('<div class="sub-header">배치 테스트</div>', unsafe_allow_html=True)
    
    # 중간 결과 다운로드 버튼 표시
    if st.session_state.temp_results is not None:
        st.download_button(
            label="가장 최근 중간 결과 다운로드",
            data=st.session_state.temp_results,
            file_name=st.session_state.temp_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_temp"
        )
    
    # 배치 테스트 모드 선택
    batch_test_mode = st.radio(
        "배치 테스트 모드",
        ["단일 데이터셋", "데이터셋 비교"],
        index=0,
        key="batch_test_mode"
    )
    
    if batch_test_mode == "단일 데이터셋":
        # 데이터셋 선택
        dataset_name = st.selectbox(
            "데이터셋 선택",
            options=list(API_DATASETS.keys()),
            index=0
        )
    else:
        # 비교할 데이터셋 선택
        st.write("비교할 데이터셋 선택:")
        batch_selected_datasets = {}
        batch_cols = st.columns(len(API_DATASETS))
        for i, dataset in enumerate(API_DATASETS.keys()):
            with batch_cols[i]:
                batch_selected_datasets[dataset] = st.checkbox(dataset, value=True, key=f"batch_compare_{dataset}")
        
        batch_selected_dataset_names = [dataset for dataset, selected in batch_selected_datasets.items() if selected]
        if not batch_selected_dataset_names:
            st.error("최소한 하나의 데이터셋을 선택해주세요.")
    
    # 파일 업로드
    uploaded_file = st.file_uploader("CSV 또는 Excel 파일 업로드 (첫 번째 컬럼을 질문으로 사용)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # 파일 형식에 따라 로드
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, header=None)
            else:
                df = pd.read_excel(uploaded_file, header=None)
            
            # 첫 번째 컬럼을 질문으로 사용
            if df.shape[1] < 1:
                st.error("업로드된 파일에 최소 한 개의 컬럼이 필요합니다.")
            else:
                # 데이터프레임 컬럼 이름 설정
                df.columns = [f'Column{i+1}' for i in range(df.shape[1])]
                df = df.rename(columns={'Column1': 'question'})
                
                st.write(f"파일 로드 성공: {len(df)} 개의 질문")
                st.dataframe(df[['question']].head(5))
                
                # 배치 크기 및 타임아웃 설정
                col1, col2 = st.columns(2)
                with col1:
                    batch_size = st.number_input("배치 크기 (한 번에 처리할 질문 수)", min_value=1, max_value=50, value=10)
                with col2:
                    timeout_seconds = st.number_input("API 요청 타임아웃 (초)", min_value=1, max_value=60, value=10)
                
                max_retries = st.slider("최대 재시도 횟수", min_value=1, max_value=10, value=3)
                
                # API 테스트 실행 버튼
                if st.button("API 테스트 실행"):
                    if batch_test_mode == "데이터셋 비교" and not batch_selected_dataset_names:
                        st.error("최소한 하나의 데이터셋을 선택해주세요.")
                    else:
                        # 처리 시작 플래그 설정
                        st.session_state.is_processing = True
                        st.session_state.stop_processing = False
                        
                        # 정지 버튼 추가
                        stop_button_placeholder = st.empty()
                        if stop_button_placeholder.button("테스트 중지", key="stop_button"):
                            st.session_state.stop_processing = True
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        error_log = []
                        
                        # 실시간 결과 표시를 위한 컨테이너
                        live_results_container = st.container()
                        
                        if batch_test_mode == "단일 데이터셋":
                            # 단일 데이터셋 배치 처리
                            # 결과 저장용 컬럼 추가
                            df['embedding_search_top10'] = ""
                            df['embedding_search_top10_with_score'] = ""
                            df['llm_top1'] = ""
                            df['status'] = ""
                            
                            # 처리 결과 저장을 위한 데이터프레임 복사
                            results_df = df.copy()
                            
                            # 일괄 처리 시작
                            total_questions = len(df)
                            processed = 0
                            successful = 0
                            failed = 0
                            
                            # 실시간 결과 표시
                            with live_results_container:
                                st.subheader("실시간 테스트 결과")
                                live_results_table = st.empty()
                                result_df = pd.DataFrame(columns=['질문', 'Top 10 검색 결과 (intent & score)', 'Top 1 재순위화 결과 (answer)', '상태'])
                                live_results_table.dataframe(result_df)
                            
                            for i in range(0, total_questions, batch_size):
                                if st.session_state.stop_processing:
                                    status_text.warning("테스트가 사용자에 의해 중지되었습니다.")
                                    break
                                
                                batch = df.iloc[i:min(i+batch_size, total_questions)]
                                
                                for idx, row in batch.iterrows():
                                    if st.session_state.stop_processing:
                                        break
                                    
                                    query = row['question']
                                    status_text.write(f"처리 중: {processed+1}/{total_questions} - {query[:50]}...")
                                    
                                    try:
                                        # API 호출 - 배치 처리에서는 디버그 비활성화
                                        with st.expander(f"질문 {processed+1}/{total_questions} 처리 중", expanded=False):
                                            st.write(f"질문: {query}")
                                            response = call_api(query, dataset_name, timeout_seconds, max_retries, show_debug=False)
                                            
                                            if "error" in response:
                                                st.error(f"오류 발생: {response['error']}")
                                                df.at[idx, 'status'] = "실패"
                                                df.at[idx, 'embedding_search_top10'] = response["error"]
                                                df.at[idx, 'embedding_search_top10_with_score'] = response["error"]
                                                df.at[idx, 'llm_top1'] = response["error"]
                                                error_log.append(f"질문 '{query}': {response['error']}")
                                                failed += 1
                                            else:
                                                # 중요 키가 있는지 확인
                                                missing_keys = []
                                                if "matches" not in response:
                                                    missing_keys.append("matches")
                                                if "reranked_matches" not in response:
                                                    missing_keys.append("reranked_matches")
                                                
                                                if missing_keys:
                                                    st.warning(f"응답에 일부 키가 누락됨: {', '.join(missing_keys)}")
                                                
                                                # 응답 추출
                                                info = extract_info(response, show_debug=False)
                                                st.success("응답 추출 성공")
                                                st.write("Top 10 검색 결과:", info['embedding_search_top10'])
                                                st.write("Top 1 재순위화 결과:", info['llm_top1'])
                                                
                                                df.at[idx, 'embedding_search_top10'] = info['embedding_search_top10']
                                                df.at[idx, 'embedding_search_top10_with_score'] = info['embedding_search_top10_with_score']
                                                df.at[idx, 'llm_top1'] = info['llm_top1']
                                                df.at[idx, 'status'] = "성공"
                                    except Exception as e:
                                        with st.expander(f"오류 - 질문 {processed+1}/{total_questions}", expanded=True):
                                            st.error(f"처리 중 예외 발생: {str(e)}")
                                        df.at[idx, 'status'] = "오류"
                                        df.at[idx, 'embedding_search_top10'] = f"처리 중 오류: {str(e)}"
                                        df.at[idx, 'embedding_search_top10_with_score'] = f"처리 중 오류: {str(e)}"
                                        df.at[idx, 'llm_top1'] = f"처리 중 오류: {str(e)}"
                                        error_log.append(f"질문 '{query}': 처리 중 오류 - {str(e)}")
                                        failed += 1
                                    
                                    processed += 1
                                    progress_bar.progress(processed / total_questions)
                                    
                                    # 실시간 결과 업데이트
                                    new_row = {
                                        '질문': query,
                                        'Top 10 검색 결과 (intent & score)': df.at[idx, 'embedding_search_top10_with_score'],
                                        'Top 1 재순위화 결과 (answer)': df.at[idx, 'llm_top1'],
                                        '상태': df.at[idx, 'status']
                                    }
                                    result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
                                    live_results_table.dataframe(result_df)
                                    
                                    # 중간 저장
                                    if processed % 10 == 0 or processed == total_questions:
                                        temp_output = BytesIO()
                                        with pd.ExcelWriter(temp_output, engine='xlsxwriter') as writer:
                                            display_df = df[['question', 'embedding_search_top10_with_score', 'llm_top1', 'status']]
                                            display_df.columns = ['질문', 'Top 10 검색 결과 (intent & score)', 'Top 1 재순위화 결과 (answer)', '상태']
                                            display_df.to_excel(writer, index=False, sheet_name='API 테스트 결과')
                                        st.session_state.temp_results = temp_output.getvalue()
                                        st.session_state.temp_filename = f"api_test_partial_results_{processed}of{total_questions}.xlsx"
                                    
                                    # 과부하 방지를 위한 지연
                                    time.sleep(0.5)
                                
                                # 배치 완료 후 잠시 대기
                                time.sleep(1)
                            
                            # 실시간 결과 테이블 유지하고 최종 상태 표시
                            st.session_state.live_results = result_df
                            
                            if st.session_state.stop_processing:
                                status_text.warning(f"테스트 중지됨: {total_questions}개 중 {processed}개 처리됨 ({successful}개 성공, {failed}개 실패)")
                            else:
                                status_text.markdown(f'<div class="info-text">테스트 완료: {total_questions}개 질문 중 <span class="success-text">{successful}개 성공</span>, <span class="error-text">{failed}개 실패</span></div>', unsafe_allow_html=True)
                            
                            # 오류 로그 표시
                            if error_log:
                                with st.expander("오류 로그 보기"):
                                    for error in error_log:
                                        st.error(error)
                            
                            # 결과 저장 및 다운로드
                            now = datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_filename = f"api_test_results_{now}.xlsx"
                            
                            # 엑셀 파일 생성
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                result_df.to_excel(writer, index=False, sheet_name='API 테스트 결과')
                                # 열 너비 자동 조정
                                worksheet = writer.sheets['API 테스트 결과']
                                for i, col in enumerate(result_df.columns):
                                    max_len = max(result_df[col].astype(str).map(len).max(), len(col)) + 2
                                    worksheet.set_column(i, i, max_len)
                                    # embedding_search_top10_with_score 컬럼은 더 넓게 설정
                                    if i == 1:  # Top 10 검색 결과 컬럼
                                        worksheet.set_column(i, i, max_len * 2)
                            
                            # 다운로드 버튼
                            st.download_button(
                                label="결과 파일 다운로드",
                                data=output.getvalue(),
                                file_name=output_filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        else:
                            # 여러 데이터셋 비교 배치 처리
                            # 결과 데이터프레임 초기화
                            result_dfs = {}
                            for dataset in batch_selected_dataset_names:
                                result_dfs[dataset] = df.copy()
                                result_dfs[dataset][f'{dataset}_embedding_search_top10'] = ""
                                result_dfs[dataset][f'{dataset}_embedding_search_top10_with_score'] = ""
                                result_dfs[dataset][f'{dataset}_llm_top1'] = ""
                                result_dfs[dataset][f'{dataset}_status'] = ""
                            
                            # 합쳐진 결과 데이터프레임 초기화
                            merged_df = df.copy()
                            
                            # 실시간 결과 표시
                            with live_results_container:
                                st.subheader("실시간 테스트 결과")
                                tabs = st.tabs([f"{dataset}" for dataset in batch_selected_dataset_names] + ["통합 결과"])
                                
                                # 각 데이터셋별 결과 테이블 초기화
                                live_results_tables = {}
                                for i, dataset in enumerate(batch_selected_dataset_names):
                                    with tabs[i]:
                                        live_results_tables[dataset] = st.empty()
                                        live_results_tables[dataset].dataframe(pd.DataFrame(columns=['질문', 'Top 10 검색 결과 (intent & score)', 'Top 1 재순위화 결과 (answer)', '상태']))
                                
                                # 통합 결과 테이블 초기화
                                with tabs[-1]:
                                    merged_results_table = st.empty()
                                    merged_results_table.dataframe(pd.DataFrame(columns=['질문'] + [f"{dataset}_결과" for dataset in batch_selected_dataset_names]))
                            
                            # 일괄 처리 시작
                            total_tasks = len(df) * len(batch_selected_dataset_names)
                            processed = 0
                            successful = 0
                            failed = 0
                            
                            # 데이터셋별 실시간 결과 초기화
                            live_dataset_results = {}
                            for dataset in batch_selected_dataset_names:
                                live_dataset_results[dataset] = pd.DataFrame(columns=['질문', 'Top 10 검색 결과 (intent & score)', 'Top 1 재순위화 결과 (answer)', '상태'])
                            
                            # 통합 결과 초기화
                            live_merged_results = pd.DataFrame(columns=['질문'] + [f"{dataset}_embedding_search_top10" for dataset in batch_selected_dataset_names] + 
                                                                     [f"{dataset}_embedding_search_top10_with_score" for dataset in batch_selected_dataset_names] + 
                                                                     [f"{dataset}_llm_top1" for dataset in batch_selected_dataset_names])
                            
                            for i in range(0, len(df), batch_size):
                                if st.session_state.stop_processing:
                                    status_text.warning("테스트가 사용자에 의해 중지되었습니다.")
                                    break
                                
                                batch = df.iloc[i:min(i+batch_size, len(df))]
                                
                                for idx, row in batch.iterrows():
                                    if st.session_state.stop_processing:
                                        break
                                    
                                    query = row['question']
                                    merged_row = {'질문': query}
                                    
                                    for dataset in batch_selected_dataset_names:
                                        if st.session_state.stop_processing:
                                            break
                                            
                                        status_text.write(f"처리 중: 질문 {idx+1}/{len(df)}, 데이터셋 {dataset}")
                                        
                                        try:
                                            # API 호출 - 배치 처리에서는 디버그 비활성화
                                            with st.expander(f"질문 {idx+1}/{len(df)} - {dataset} 처리 중", expanded=False):
                                                st.write(f"질문: {query}")
                                                response = call_api(query, dataset, timeout_seconds, max_retries, show_debug=False)
                                                
                                                if "error" in response:
                                                    st.error(f"오류 발생: {response['error']}")
                                                    result_dfs[dataset].at[idx, f'{dataset}_status'] = "실패"
                                                    result_dfs[dataset].at[idx, f'{dataset}_embedding_search_top10'] = response["error"]
                                                    result_dfs[dataset].at[idx, f'{dataset}_embedding_search_top10_with_score'] = response["error"]
                                                    result_dfs[dataset].at[idx, f'{dataset}_llm_top1'] = response["error"]
                                                    error_log.append(f"데이터셋 {dataset}, 질문 '{query}': {response['error']}")
                                                    failed += 1
                                                    
                                                    # 실시간 결과 업데이트
                                                    status = "실패"
                                                    embedding_search = response["error"]
                                                    embedding_search_with_score = response["error"]
                                                    llm_result = response["error"]
                                                else:
                                                    # 중요 키가 있는지 확인
                                                    missing_keys = []
                                                    if "matches" not in response:
                                                        missing_keys.append("matches")
                                                    if "reranked_matches" not in response:
                                                        missing_keys.append("reranked_matches")
                                                    
                                                    if missing_keys:
                                                        st.warning(f"응답에 일부 키가 누락됨: {', '.join(missing_keys)}")
                                                    
                                                    # 응답 추출
                                                    info = extract_info(response, show_debug=False)
                                                    st.success("응답 추출 성공")
                                                    st.write("Top 10 검색 결과:", info['embedding_search_top10'])
                                                    st.write("Top 1 재순위화 결과:", info['llm_top1'])
                                                    
                                                    result_dfs[dataset].at[idx, f'{dataset}_embedding_search_top10'] = info['embedding_search_top10']
                                                    result_dfs[dataset].at[idx, f'{dataset}_embedding_search_top10_with_score'] = info['embedding_search_top10_with_score']
                                                    result_dfs[dataset].at[idx, f'{dataset}_llm_top1'] = info['llm_top1']
                                                    result_dfs[dataset].at[idx, f'{dataset}_status'] = "성공"
                                        except Exception as e:
                                            with st.expander(f"오류 - 질문 {idx+1}/{len(df)} - {dataset}", expanded=True):
                                                st.error(f"처리 중 예외 발생: {str(e)}")
                                            result_dfs[dataset].at[idx, f'{dataset}_status'] = "오류"
                                            result_dfs[dataset].at[idx, f'{dataset}_embedding_search_top10'] = f"처리 중 오류: {str(e)}"
                                            result_dfs[dataset].at[idx, f'{dataset}_embedding_search_top10_with_score'] = f"처리 중 오류: {str(e)}"
                                            result_dfs[dataset].at[idx, f'{dataset}_llm_top1'] = f"처리 중 오류: {str(e)}"
                                            error_log.append(f"데이터셋 {dataset}, 질문 '{query}': 처리 중 오류 - {str(e)}")
                                            failed += 1
                                            
                                            # 실시간 결과 업데이트
                                            status = "오류"
                                            embedding_search = f"처리 중 오류: {str(e)}"
                                            embedding_search_with_score = f"처리 중 오류: {str(e)}"
                                            llm_result = f"처리 중 오류: {str(e)}"
                                            
                                            # 합쳐진 결과 데이터프레임에도 추가
                                            merged_df.at[idx, f'{dataset}_embedding_search_top10'] = f"처리 중 오류: {str(e)}"
                                            merged_df.at[idx, f'{dataset}_embedding_search_top10_with_score'] = f"처리 중 오류: {str(e)}"
                                            merged_df.at[idx, f'{dataset}_llm_top1'] = f"처리 중 오류: {str(e)}"
                                            merged_df.at[idx, f'{dataset}_status'] = "오류"
                                        
                                    # 데이터셋별 실시간 결과 업데이트
                                    new_row = {
                                        '질문': query,
                                        'Top 10 검색 결과 (intent & score)': embedding_search_with_score,
                                        'Top 1 재순위화 결과 (answer)': llm_result,
                                        '상태': status
                                    }
                                    live_dataset_results[dataset] = pd.concat([live_dataset_results[dataset], pd.DataFrame([new_row])], ignore_index=True)
                                    live_results_tables[dataset].dataframe(live_dataset_results[dataset])
                                    
                                    # 통합 결과용 데이터 추가
                                    merged_row[f"{dataset}_embedding_search_top10"] = embedding_search
                                    merged_row[f"{dataset}_embedding_search_top10_with_score"] = embedding_search_with_score
                                    merged_row[f"{dataset}_llm_top1"] = llm_result
                                    
                                    processed += 1
                                    progress_bar.progress(processed / total_tasks)
                                    
                                    # 과부하 방지를 위한 지연
                                    time.sleep(0.5)
                                
                                # 통합 결과 업데이트
                                live_merged_results = pd.concat([live_merged_results, pd.DataFrame([merged_row])], ignore_index=True)
                                merged_results_table.dataframe(live_merged_results)
                                
                                # 중간 저장
                                if (idx + 1) % 5 == 0 or idx == len(df) - 1:
                                    temp_output = BytesIO()
                                    with pd.ExcelWriter(temp_output, engine='xlsxwriter') as writer:
                                        merged_df.to_excel(writer, index=False, sheet_name='API 테스트 결과')
                                    st.session_state.temp_results = temp_output.getvalue()
                                    st.session_state.temp_filename = f"api_comparison_partial_results_{idx+1}of{len(df)}.xlsx"
                            
                            # 실시간 결과 테이블 유지하고 최종 상태 표시
                            st.session_state.live_results = live_merged_results
                            
                            if st.session_state.stop_processing:
                                status_text.warning(f"테스트 중지됨: {total_tasks}개 중 {processed}개 처리됨 ({successful}개 성공, {failed}개 실패)")
                            else:
                                status_text.markdown(f'<div class="info-text">테스트 완료: {total_tasks}개 작업 중 <span class="success-text">{successful}개 성공</span>, <span class="error-text">{failed}개 실패</span></div>', unsafe_allow_html=True)
                            
                            # 오류 로그 표시
                            if error_log:
                                with st.expander("오류 로그 보기"):
                                    for error in error_log:
                                        st.error(error)
                            
                            # 결과 저장 및 다운로드
                            now = datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_filename = f"api_comparison_results_{now}.xlsx"
                            
                            # 엑셀 파일 생성
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                # 합쳐진 결과 시트
                                merged_df.to_excel(writer, index=False, sheet_name='통합 결과')
                                
                                # 각 데이터셋별 결과 시트
                                for dataset in batch_selected_dataset_names:
                                    display_df = result_dfs[dataset][['question', f'{dataset}_embedding_search_top10_with_score', f'{dataset}_llm_top1', f'{dataset}_status']]
                                    display_df.columns = ['질문', 'Top 10 검색 결과 (intent & score)', 'Top 1 재순위화 결과 (answer)', '상태']
                                    display_df.to_excel(writer, index=False, sheet_name=f'{dataset} 결과')
                            
                            # 다운로드 버튼
                            st.download_button(
                                label="비교 결과 다운로드",
                                data=output.getvalue(),
                                file_name=output_filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
        
        except Exception as e:
            st.error(f"파일 처리 중 오류 발생: {str(e)}")

# 단일 테스트 탭
with tab2:
    st.markdown('<div class="sub-header">단일 질문 테스트</div>', unsafe_allow_html=True)
    
    # 데이터셋 선택 (라디오 버튼으로 변경)
    test_mode = st.radio(
        "테스트 모드",
        ["단일 데이터셋", "모든 데이터셋 비교"],
        index=0,
        key="test_mode"
    )
    
    if test_mode == "단일 데이터셋":
        # 단일 데이터셋 선택
        single_dataset_name = st.selectbox(
            "데이터셋 선택",
            options=list(API_DATASETS.keys()),
            index=0,
            key="single_test_dataset"
        )
    else:
        # 모든 데이터셋 체크박스
        st.write("비교할 데이터셋 선택:")
        selected_datasets = {}
        cols = st.columns(len(API_DATASETS))
        for i, dataset in enumerate(API_DATASETS.keys()):
            with cols[i]:
                selected_datasets[dataset] = st.checkbox(dataset, value=True, key=f"compare_{dataset}")
    
    # 타임아웃 설정
    single_timeout = st.slider("API 요청 타임아웃 (초)", min_value=1, max_value=60, value=10)
    
    # 질문 입력
    query = st.text_area("질문 입력", height=100)
    
    # 결과 표시 영역 미리 확보 (최상단에 검색 결과 요약 표시)
    results_placeholder = st.empty()
    
    # API 직접 테스트 옵션
    show_direct_test = st.checkbox("API 직접 테스트 보기", value=False)
    if show_direct_test:
        st.markdown("### API 직접 테스트")
        st.markdown("curl 명령으로 API를 직접 테스트합니다.")
        
        if test_mode == "단일 데이터셋":
            dataset_info = API_DATASETS[single_dataset_name]
        else:
            # 선택된 데이터셋 목록 가져오기
            selected_dataset_names = [dataset for dataset, selected in selected_datasets.items() if selected]
            if selected_dataset_names:
                dataset_info = API_DATASETS[selected_dataset_names[0]]
            else:
                dataset_info = API_DATASETS[list(API_DATASETS.keys())[0]]
        
        # curl 명령 생성
        curl_command = f"""curl --location '{API_ENDPOINT}' \\
--header 'Content-Type: application/json' \\
--header 'Authorization: {API_HEADERS["Authorization"]}' \\
--header 'azureml-model-deployment: {API_HEADERS["azureml-model-deployment"]}' \\
--data '{{
    "user_query":"{query}",
    "top_k":"10",
    "index_path":"{dataset_info["index"]}",
    "metadata_path":"{dataset_info["metadata"]}"
}}'"""
        
        st.code(curl_command, language="bash")
        
        # 직접 테스트 버튼
        if st.button("직접 API 테스트 실행"):
            if not query:
                st.error("질문을 입력해주세요.")
            else:
                with st.spinner("API 직접 호출 중..."):
                    # 원시 API 호출
                    try:
                        payload = {
                            "user_query": query,
                            "top_k": "10",
                            "index_path": dataset_info["index"],
                            "metadata_path": dataset_info["metadata"]
                        }
                        
                        st.write("API 요청 정보:")
                        st.json({
                            "endpoint": API_ENDPOINT,
                            "headers": API_HEADERS,
                            "payload": payload
                        })
                        
                        response = requests.post(API_ENDPOINT, headers=API_HEADERS, json=payload, timeout=single_timeout)
                        
                        st.write(f"응답 상태 코드: {response.status_code}")
                        st.write(f"응답 헤더: {dict(response.headers)}")
                        
                        if response.status_code == 200:
                            try:
                                json_response = response.json()
                                st.success("API 응답 성공")
                                st.subheader("API 원본 응답:")
                                st.json(json_response)
                                
                                # intent_data 객체 확인 (중첩 구조 처리)
                                if "intent_data" in json_response:
                                    intent_data = json_response["intent_data"]
                                else:
                                    intent_data = json_response  # 중첩되지 않은 경우
                                
                                # matches에서 Top 10 결과 추출
                                matches = intent_data.get("matches", [])
                                top10_results = []
                                
                                for i, match in enumerate(matches[:10]):
                                    intent = match.get("intent", "")
                                    score = match.get("score", 0)
                                    if intent:
                                        top10_results.append(f"{i+1}. \"{intent}\" {score:.4f}")
                                
                                # 줄바꿈을 HTML <br> 태그로 변환하여 표에서 줄바꿈이 보이도록 함
                                top10_text = "<br>".join(top10_results) if top10_results else "결과 없음"
                                
                                # reranked_matches에서 Top 1 결과 추출
                                reranked_matches = intent_data.get("reranked_matches", [])
                                top1_answer = reranked_matches[0].get("answer", "결과 없음") if reranked_matches else "결과 없음"
                                
                                # 결과 테이블로 표시 (맨 위 placeholder에)
                                result_df = pd.DataFrame({
                                    "질문": [query],
                                    "Top 10 검색 결과 (intent & score)": [top10_text],
                                    "Top 1 재순위화 결과 (answer)": [top1_answer]
                                })
                                
                                # HTML 태그가 적용되도록 unsafe_allow_html=True 설정
                                results_placeholder.write(result_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"응답 파싱 오류: {str(e)}")
                                st.text(f"원본 응답: {response.text[:1000]}")  # 첫 1000자만 표시
                        else:
                            st.error(f"API 요청 실패: 상태 코드 {response.status_code}")
                            st.text(f"응답 내용: {response.text[:1000]}")
                    except Exception as e:
                        st.error(f"API 호출 중 오류 발생: {str(e)}")
    
    # 기존 테스트 버튼
    if st.button("테스트 실행"):
        if not query:
            st.error("질문을 입력해주세요.")
        else:
            if test_mode == "단일 데이터셋":
                # 단일 데이터셋 테스트
                with st.spinner(f"{single_dataset_name} 데이터셋으로 API 호출 중..."):
                    # API 호출
                    response = call_api(query, single_dataset_name, single_timeout, show_debug=False)
                    
                    if "error" in response:
                        results_placeholder.error(f"API 요청 오류: {response['error']}")
                    else:
                        # intent_data 객체 확인 (중첩 구조 처리)
                        if "intent_data" in response:
                            intent_data = response["intent_data"]
                        else:
                            intent_data = response  # 중첩되지 않은 경우
                        
                        # matches에서 Top 10 결과 추출
                        matches = intent_data.get("matches", [])
                        top10_results = []
                        
                        for i, match in enumerate(matches[:10]):
                            intent = match.get("intent", "")
                            score = match.get("score", 0)
                            if intent:
                                top10_results.append(f"{i+1}. \"{intent}\" {score:.4f}")
                        
                        # 줄바꿈을 HTML <br> 태그로 변환하여 표에서 줄바꿈이 보이도록 함
                        top10_text = "<br>".join(top10_results) if top10_results else "결과 없음"
                        
                        # reranked_matches에서 Top 1 결과 추출
                        reranked_matches = intent_data.get("reranked_matches", [])
                        top1_answer = reranked_matches[0].get("answer", "결과 없음") if reranked_matches else "결과 없음"
                        
                        # 결과 테이블로 표시 (맨 위 placeholder에)
                        result_df = pd.DataFrame({
                            "질문": [query],
                            "Top 10 검색 결과 (intent & score)": [top10_text],
                            "Top 1 재순위화 결과 (answer)": [top1_answer]
                        })
                        
                        # HTML 태그가 적용되도록 unsafe_allow_html=True 설정
                        results_placeholder.write(result_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                        
                    # 원본 응답 데이터 확인 (선택적으로 보이게)
                    with st.expander("원본 API 응답 보기"):
                        st.json(response)
                        
                        if "intent_data" in response:
                            intent_data = response["intent_data"]
                            st.write("API 응답이 intent_data 객체 안에 중첩되어 있습니다.")
                            
                            if "matches" in intent_data:
                                st.write(f"matches 개수: {len(intent_data['matches'])}")
                                st.write("첫 번째 matches 항목 예시:")
                                st.json(intent_data["matches"][0] if intent_data["matches"] else "매치 없음")
                            
                            if "reranked_matches" in intent_data:
                                st.write(f"reranked_matches 개수: {len(intent_data['reranked_matches'])}")
                                st.write("첫 번째 reranked_matches 항목 예시:")
                                st.json(intent_data["reranked_matches"][0] if intent_data["reranked_matches"] else "재순위화 결과 없음")
                        else:
                            if "matches" in response:
                                st.write("첫 번째 matches 항목 예시:")
                                st.json(response["matches"][0] if response["matches"] else "매치 없음")
                            
                            if "reranked_matches" in response:
                                st.write("첫 번째 reranked_matches 항목 예시:")
                                st.json(response["reranked_matches"][0] if response["reranked_matches"] else "재순위화 결과 없음")
            else:
                # 여러 데이터셋 비교 테스트
                selected_dataset_names = [dataset for dataset, selected in selected_datasets.items() if selected]
                
                if not selected_dataset_names:
                    st.error("최소한 하나의 데이터셋을 선택해주세요.")
                else:
                    # 각 데이터셋별 결과 저장용 테이블 준비
                    comparison_data = []
                    
                    for dataset_name in selected_dataset_names:
                        with st.spinner(f"{dataset_name} 데이터셋으로 API 호출 중..."):
                            # API 호출
                            response = call_api(query, dataset_name, single_timeout)
                            
                            if "error" in response:
                                row = {
                                    "데이터셋": dataset_name,
                                    "Top 10 검색 결과 (intent & score)": response["error"],
                                    "Top 1 재순위화 결과 (answer)": response["error"]
                                }
                            else:
                                # intent_data 객체 확인 (중첩 구조 처리)
                                if "intent_data" in response:
                                    intent_data = response["intent_data"]
                                else:
                                    intent_data = response  # 중첩되지 않은 경우
                                
                                # matches에서 Top 10 결과 추출
                                matches = intent_data.get("matches", [])
                                top10_results = []
                                
                                for i, match in enumerate(matches[:10]):
                                    intent = match.get("intent", "")
                                    score = match.get("score", 0)
                                    if intent:
                                        top10_results.append(f"{i+1}. \"{intent}\" {score:.4f}")
                                
                                # 줄바꿈을 HTML <br> 태그로 변환하여 표에서 줄바꿈이 보이도록 함
                                top10_text = "<br>".join(top10_results) if top10_results else "결과 없음"
                                
                                # reranked_matches에서 Top 1 결과 추출
                                reranked_matches = intent_data.get("reranked_matches", [])
                                top1_answer = reranked_matches[0].get("answer", "결과 없음") if reranked_matches else "결과 없음"
                                
                                row = {
                                    "데이터셋": dataset_name,
                                    "Top 10 검색 결과 (intent & score)": top10_text,
                                    "Top 1 재순위화 결과 (answer)": top1_answer
                                }
                            
                            comparison_data.append(row)
                            
                            # 전체 응답도 저장
                            st.session_state[f"response_{dataset_name}"] = response
                    
                    # 비교 테이블 표시 (맨 위 placeholder에) - HTML 태그가 적용되도록 
                    results_placeholder.subheader("데이터셋 비교 결과")
                    comparison_df = pd.DataFrame(comparison_data)
                    results_placeholder.write(comparison_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    
                    # 전체 응답 보기 옵션
                    st.subheader("전체 응답 보기")
                    selected_dataset_for_details = st.selectbox(
                        "상세 응답을 볼 데이터셋 선택:",
                        options=selected_dataset_names
                    )
                    
                    if selected_dataset_for_details:
                        with st.expander(f"{selected_dataset_for_details} 전체 응답"):
                            st.json(st.session_state[f"response_{selected_dataset_for_details}"])
                    
                    # 비교 결과 다운로드 - 엑셀에서는 <br> 대신 줄바꿈 문자 사용
                    comparison_excel_df = comparison_df.copy()
                    for i, row in comparison_excel_df.iterrows():
                        if isinstance(row["Top 10 검색 결과 (intent & score)"], str) and "<br>" in row["Top 10 검색 결과 (intent & score)"]:
                            comparison_excel_df.at[i, "Top 10 검색 결과 (intent & score)"] = row["Top 10 검색 결과 (intent & score)"].replace("<br>", "\n")
                    
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        comparison_excel_df.to_excel(writer, index=False, sheet_name='데이터셋 비교 결과')
                        # 셀 서식 지정 - 자동 줄바꿈 활성화
                        workbook = writer.book
                        worksheet = writer.sheets['데이터셋 비교 결과']
                        wrap_format = workbook.add_format({'text_wrap': True})
                        # Top 10 검색 결과 컬럼에 자동 줄바꿈 적용
                        worksheet.set_column('B:B', 60, wrap_format)
                    
                    st.download_button(
                        label="비교 결과 다운로드",
                        data=output.getvalue(),
                        file_name=f"dataset_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    ) 