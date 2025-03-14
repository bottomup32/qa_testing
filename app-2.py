import pandas as pd
import requests
import json
import textwrap
import streamlit as st
from io import BytesIO
import time
import os
import concurrent.futures
import pickle
from tqdm.auto import tqdm

# 기존 API 설정 (필요 시 기본값으로 사용)
API_CONFIGS = {
    "Default API": {
        "url": "https://ai-human-chatbot-roasu.koreacentral.inference.ml.azure.com/score",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer BV3RjJz66FNP82LSkw6SU1905E57UgOMizCuAe3y63atc5Wwj5HXJQQJ99BCAAAAAAAAAAAAINFRAZML1o4N",
            "azureml-model-deployment": "ai-human-chatbot-roasu-2"
        }
    },
    "Backup API": {
        "url": "https://ai-human-chatbot-roasu-backup.koreacentral.inference.ml.azure.com/score",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer BACKUP_TOKEN",
            "azureml-model-deployment": "ai-human-chatbot-roasu-2"
        }
    }
}

# 임시 파일 저장 디렉토리
TEMP_DIR = "temp_data"
os.makedirs(TEMP_DIR, exist_ok=True)

def query_api(api_name, question, timeout=10, custom_api_url=None):
    """API에 질의하고 응답을 반환 (타임아웃 설정 추가 및 사용자 지정 URL 지원)"""
    try:
        # API 설정 가져오기
        api_config = API_CONFIGS.get(api_name)
        
        # 선택한 API 설정이 없으면 Default API 사용
        if api_config is None:
            api_config = API_CONFIGS["Default API"]
            st.warning(f"'{api_name}' API 설정을 찾을 수 없어 Default API를 사용합니다.")
        
        # 사용자 지정 API URL이 입력된 경우, 해당 URL로 덮어쓰기
        if custom_api_url and custom_api_url.strip() != "":
            api_config["url"] = custom_api_url.strip()
        
        payload = json.dumps({"user_query": question})
        
        # API 요청 시도
        response = requests.post(
            api_config["url"], 
            headers=api_config["headers"], 
            data=payload,
            timeout=timeout  # 타임아웃 설정
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API 응답 오류 (상태 코드: {response.status_code}, 메시지: {response.text})"}
            
    except requests.exceptions.Timeout:
        return {"error": f"API 요청 타임아웃 ({timeout}초 초과)"}
    except requests.exceptions.RequestException as e:
        # 연결 오류가 발생하고 Default API가 아니면 Default API로 재시도
        if api_name != "Default API":
            st.warning(f"{api_name} API 연결 오류, Default API로 재시도합니다.")
            return query_api("Default API", question, timeout, custom_api_url)
        return {"error": f"API 요청 실패: {str(e)}"}
    except Exception as e:
        return {"error": f"예상치 못한 오류: {str(e)}"}

def process_response(response):
    """API 응답에서 필요한 정보 추출"""
    if not response:
        return "API 응답 없음", "응답을 받지 못했습니다."
    
    if "error" in response:
        return f"오류: {response['error']}", f"API 요청 중 오류가 발생했습니다: {response['error']}"
    
    if "intent_data" not in response:
        return "Intent 데이터 없음", "응답에 intent_data가 없습니다."
    
    try:
        # Embedding Search Top 10
        embedding_top_10 = "\n".join(
            [f"- {match['intent']} (Score: {match.get('score', 'N/A'):.4f})" 
             for match in response["intent_data"]["matches"][:10]]
        )
    
        # LLM Top 3
        llm_top_3 = "\n\n".join(
            [textwrap.fill(f"{i+1}. {match['answer']}", width=80) 
             for i, match in enumerate(response["intent_data"]["reranked_matches"][:3])]
        )
    
        return embedding_top_10, llm_top_3
    except Exception as e:
        return f"응답 처리 오류: {str(e)}", "응답 데이터 처리 중 오류가 발생했습니다."

def process_batch(batch_data, api_name, timeout=10, custom_api_url=None):
    """배치 데이터 처리 (병렬 처리를 위한 함수)"""
    results = []
    for idx, question in batch_data:
        response = query_api(api_name, str(question), timeout, custom_api_url)
        
        # 오류 발생 시 재시도 (최대 1회)
        if response and "error" in response and "API 요청 실패" in response["error"]:
            time.sleep(1)
            response = query_api("Default API", str(question), timeout, custom_api_url)
            
        embedding_top_10, llm_top_3 = process_response(response)
        results.append((idx, embedding_top_10, llm_top_3))
    return results

def get_checkpoint_filename(uploaded_file_name):
    """체크포인트 파일 이름 생성"""
    base_name = os.path.splitext(uploaded_file_name)[0]
    return os.path.join(TEMP_DIR, f"{base_name}_checkpoint.pkl")

def save_checkpoint(uploaded_file_name, df, processed_indices):
    """처리 진행 상황 저장"""
    checkpoint_data = {
        'df': df,
        'processed_indices': processed_indices,
        'timestamp': time.time()
    }
    
    filename = get_checkpoint_filename(uploaded_file_name)
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    return filename

def load_checkpoint(uploaded_file_name):
    """저장된 체크포인트 불러오기"""
    filename = get_checkpoint_filename(uploaded_file_name)
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                checkpoint_data = pickle.load(f)
            return checkpoint_data
        except Exception as e:
            st.warning(f"체크포인트 파일 로드 중 오류: {str(e)}")
    return None

def process_file(api_name, uploaded_file, result_placeholder, stats_placeholder, df_placeholder, question_column, 
                 workers=4, batch_size=10, timeout=10, start_from_checkpoint=False, custom_api_url=None):
    """파일을 읽고 API 요청을 실행하여 처리된 데이터 반환 (병렬 처리 및 체크포인트 지원)"""
    try:
        checkpoint_data = None
        processed_indices = set()
        
        if start_from_checkpoint:
            checkpoint_data = load_checkpoint(uploaded_file.name)
            if checkpoint_data:
                df = checkpoint_data['df']
                processed_indices = set(checkpoint_data['processed_indices'])
                checkpoint_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(checkpoint_data['timestamp']))
                stats_placeholder.info(f"체크포인트에서 재시작합니다. (저장 시간: {checkpoint_time})")
                stats_placeholder.info(f"이미 처리된 질문 수: {len(processed_indices)}/{len(df)}")
                df_placeholder.dataframe(df, use_container_width=True)
                progress_bar = st.progress(len(processed_indices) / len(df))
                time.sleep(2)
        
        if checkpoint_data is None:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("지원되지 않는 파일 형식입니다. CSV 또는 XLSX 파일을 업로드해주세요.")
                return None
            
            if question_column not in df.columns:
                stats_placeholder.error(f"선택한 열 '{question_column}'이 파일에 존재하지 않습니다. 올바른 열을 선택해주세요.")
                return None
            
            if "Embedding Search Top 10" not in df.columns:
                df["Embedding Search Top 10"] = ""
            if "LLM Top 3" not in df.columns:
                df["LLM Top 3"] = ""
        
        progress_bar = st.progress(len(processed_indices) / len(df))
        
        if 'processing' not in st.session_state:
            st.session_state.processing = True
        
        start_time = time.time()
        total_rows = len(df)
        remaining_indices = [i for i in range(total_rows) if i not in processed_indices]
        total_remaining = len(remaining_indices)
        
        stats_placeholder.info(f"총 {total_rows}개 질문 중 {total_remaining}개 처리 예정 (병렬 처리: {workers}개 스레드)")
        
        batches = []
        current_batch = []
        
        for idx in remaining_indices:
            question = df.iloc[idx][question_column]
            current_batch.append((idx, question))
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
        if current_batch:
            batches.append(current_batch)
        
        batch_progress = st.empty()
        batch_progress.text(f"0/{len(batches)} 배치 처리 중...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i in range(min(workers, len(batches))):
                if i < len(batches):
                    futures.append(executor.submit(process_batch, batches[i], api_name, timeout, custom_api_url))
            
            completed_batches = 0
            next_batch_idx = min(workers, len(batches))
            
            while futures:
                done, not_done = concurrent.futures.wait(
                    futures, 
                    timeout=0.1,
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                if not st.session_state.processing:
                    for future in futures:
                        future.cancel()
                    stats_placeholder.warning(f"처리가 중단되었습니다. {len(processed_indices)}/{total_rows} 질문 처리 완료.")
                    break
                
                for future in done:
                    results = future.result()
                    for idx, embedding_top_10, llm_top_3 in results:
                        df.at[idx, "Embedding Search Top 10"] = embedding_top_10
                        df.at[idx, "LLM Top 3"] = llm_top_3
                        processed_indices.add(idx)
                    
                    if next_batch_idx < len(batches):
                        futures.append(executor.submit(process_batch, batches[next_batch_idx], api_name, timeout, custom_api_url))
                        next_batch_idx += 1
                    
                    completed_batches += 1
                    batch_progress.text(f"{completed_batches}/{len(batches)} 배치 처리 중... ({len(processed_indices)}/{total_rows} 질문)")
                    progress_bar.progress(len(processed_indices) / total_rows)
                    
                    if completed_batches % 2 == 0 or completed_batches == len(batches):
                        display_df = df.copy()
                        df_placeholder.dataframe(display_df, use_container_width=True)
                    
                    if completed_batches % 5 == 0 or completed_batches == len(batches):
                        checkpoint_file = save_checkpoint(uploaded_file.name, df, list(processed_indices))
                        stats_placeholder.info(f"체크포인트 저장됨: {time.strftime('%H:%M:%S')}")
                
                futures = list(not_done)
                
                elapsed_time = time.time() - start_time
                if len(processed_indices) > 0:
                    avg_time_per_question = elapsed_time / len(processed_indices)
                    remaining_questions = total_rows - len(processed_indices)
                    estimated_time_remaining = avg_time_per_question * remaining_questions
                    est_minutes = int(estimated_time_remaining // 60)
                    est_seconds = int(estimated_time_remaining % 60)
                    
                    stats_text = (f"진행 상황: {len(processed_indices)}/{total_rows} 질문 ({len(processed_indices)/total_rows*100:.1f}%)\n"
                                  f"경과 시간: {int(elapsed_time//60)}분 {int(elapsed_time%60)}초\n"
                                  f"예상 남은 시간: {est_minutes}분 {est_seconds}초\n"
                                  f"처리 속도: {len(processed_indices)/elapsed_time:.2f} 질문/초")
                    
                    stats_placeholder.text(stats_text)
        
        if st.session_state.processing:
            save_checkpoint(uploaded_file.name, df, list(processed_indices))
        
        total_time = time.time() - start_time
        if st.session_state.processing or len(processed_indices) == total_rows:
            result_placeholder.success(f"처리가 완료되었습니다! 총 {total_rows}개 질문, 소요 시간: {int(total_time//60)}분 {int(total_time%60)}초")
        
        df_placeholder.dataframe(df, use_container_width=True)
        st.session_state.processing = False
    
        return df

    except Exception as e:
        try:
            if 'df' in locals() and 'processed_indices' in locals():
                save_checkpoint(uploaded_file.name, df, list(processed_indices))
                stats_placeholder.info("오류 발생 시점까지의 진행상황이 저장되었습니다.")
        except:
            pass
            
        stats_placeholder.error(f"처리 중 오류가 발생했습니다: {str(e)}")
        st.session_state.processing = False
        return None

def process_single_question(api_name, question, timeout=10, custom_api_url=None):
    """단일 질문을 처리하고 결과 반환"""
    with st.spinner("API 요청 처리 중..."):
        response = query_api(api_name, question, timeout, custom_api_url)
        if response:
            embedding_top_10, llm_top_3 = process_response(response)
            return embedding_top_10, llm_top_3
        else:
            return "API 응답 오류 발생", "응답을 받지 못했습니다."

# Streamlit 앱 메인 함수
st.title("API 테스트 자동화 툴")

# 세션 상태 초기화
if 'processing' not in st.session_state:
    st.session_state.processing = False

# 사이드바 설정
st.sidebar.header("API 설정")
api_name = st.sidebar.selectbox(
    "사용할 API 선택",
    list(API_CONFIGS.keys()),
    index=0
)

# 사용자 지정 API URL 입력 (비워두면 선택된 API 사용)
custom_api_url = st.sidebar.text_input(
    "사용자 지정 API URL (입력 시 선택된 API URL을 대체합니다)",
    value=""
)

# 선택된 API URL 정보 표시
if custom_api_url and custom_api_url.strip() != "":
    st.sidebar.info(f"사용자 지정 API URL: {custom_api_url.strip()}")
else:
    st.sidebar.info(f"선택된 API URL: {API_CONFIGS[api_name]['url']}")

# 고급 설정
st.sidebar.header("고급 설정")
timeout = st.sidebar.slider("API 타임아웃 (초)", min_value=5, max_value=60, value=10, step=5)
workers = st.sidebar.slider("병렬 처리 스레드 수", min_value=1, max_value=16, value=4, step=1)
batch_size = st.sidebar.slider("배치 크기", min_value=1, max_value=20, value=5, step=1)

tab1, tab2 = st.tabs(["단일 질문", "대량 질문 (Excel)"])

with tab1:
    st.subheader("단일 질문 테스트")
    st.write("질문을 입력하고 API 응답을 바로 확인하세요.")
    question = st.text_area("질문 입력", height=100, placeholder="여기에 질문을 입력하세요...")
    if st.button("질문 제출"):
        if question:
            embedding_top_10, llm_top_3 = process_single_question(api_name, question, timeout, custom_api_url)
            st.subheader("Embedding Search Top 10")
            st.markdown(embedding_top_10)
            st.subheader("LLM Top 3")
            st.markdown(llm_top_3)
            st.subheader("결과 요약")
            result_df = pd.DataFrame({
                "질문": [question],
                "Embedding Search Top 10": [embedding_top_10],
                "LLM Top 3": [llm_top_3]
            })
            st.dataframe(result_df)
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                result_df.to_excel(writer, index=False)
            excel_data = output.getvalue()
            st.download_button(
                label="결과 Excel 파일 다운로드",
                data=excel_data,
                file_name="single_question_result.xlsx",
                mime="application/vnd.ms-excel"
            )
    else:
        st.warning("질문을 입력해주세요.")

with tab2:
    st.subheader("대량 질문 처리")
    st.write("CSV 또는 XLSX 파일을 업로드하고 API를 선택하여 테스트를 실행하세요.")
    uploaded_file = st.file_uploader("CSV 또는 XLSX 파일 업로드", type=["csv", "xlsx"])
    if uploaded_file is not None:
        st.write("파일이 업로드되었습니다:", uploaded_file.name)
        checkpoint_exists = os.path.exists(get_checkpoint_filename(uploaded_file.name))
        try:
            if uploaded_file.name.endswith(".csv"):
                preview_df = pd.read_csv(uploaded_file)
            else:
                preview_df = pd.read_excel(uploaded_file)
            st.subheader("파일 미리보기")
            st.dataframe(preview_df.head(5))
            question_column = st.selectbox(
                "질문이 포함된 열을 선택하세요",
                options=preview_df.columns.tolist(),
                index=0
            )
            use_checkpoint = False
            if checkpoint_exists:
                use_checkpoint = st.checkbox("이전 진행 상황에서 계속하기", value=True)
                if use_checkpoint:
                    st.info("저장된 체크포인트에서 처리를 재개합니다. 이전에 처리된 질문은 건너뜁니다.")
            result_placeholder = st.empty()
            stats_placeholder = st.empty()
            col1, col2, col3 = st.columns(3)
            start_button = col1.button("테스트 시작", disabled=st.session_state.processing, key="start_button")
            stop_button = col2.button("처리 중단", key="stop_button")
            if checkpoint_exists:
                delete_checkpoint = col3.button("체크포인트 삭제", key="delete_button")
                if delete_checkpoint:
                    try:
                        os.remove(get_checkpoint_filename(uploaded_file.name))
                        st.success("체크포인트가 삭제되었습니다.")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"체크포인트 삭제 오류: {str(e)}")
            df_placeholder = st.empty()
            download_placeholder = st.empty()
            if start_button:
                st.session_state.processing = True
                uploaded_file.seek(0)
                df = process_file(
                    api_name, 
                    uploaded_file, 
                    result_placeholder, 
                    stats_placeholder, 
                    df_placeholder, 
                    question_column,
                    workers=workers,
                    batch_size=batch_size,
                    timeout=timeout,
                    start_from_checkpoint=use_checkpoint,
                    custom_api_url=custom_api_url
                )
                if df is not None:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False)
                    excel_data = output.getvalue()
                    download_placeholder.download_button(
                        label="결과 Excel 파일 다운로드",
                        data=excel_data,
                        file_name="test_results.xlsx",
                        mime="application/vnd.ms-excel",
                        key="download_button"
                    )
            if stop_button:
                if st.session_state.processing:
                    st.session_state.processing = False
                    result_placeholder.warning("사용자 요청으로 처리가 중단되었습니다. 진행 상황은 저장되었습니다.")
                else:
                    result_placeholder.info("현재 처리 중인 작업이 없습니다.")
            if checkpoint_exists and not st.session_state.processing:
                if st.button("저장된 체크포인트 결과 보기", key="view_checkpoint"):
                    checkpoint_data = load_checkpoint(uploaded_file.name)
                    if checkpoint_data:
                        df = checkpoint_data['df']
                        df_placeholder.dataframe(df, use_container_width=True)
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df.to_excel(writer, index=False)
                        excel_data = output.getvalue()
                        download_placeholder.download_button(
                            label="저장된 결과 Excel 파일 다운로드",
                            data=excel_data,
                            file_name="checkpoint_results.xlsx",
                            mime="application/vnd.ms-excel",
                            key="download_checkpoint_button"
                        )
        except Exception as e:
            st.error(f"파일 읽기 오류: {str(e)}")
