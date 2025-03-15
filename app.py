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
import logging
import traceback
import random
import threading

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler("api_test.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# 기본 API 설정 (변경 없이 사용)
API_CONFIGS = {
    "Default API": {
        "url": "https://ai-human-chatbot-roasu.koreacentral.inference.ml.azure.com/score",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer BV3RjJz66FNP82LSkw6SU1905E57UgOMizCuAe3y63atc5Wwj5HXJQQJ99BCAAAAAAAAAAAAINFRAZML1o4N",
            "azureml-model-deployment": "ai-human-chatbot-roasu-4"
        }
    },
    "Backup API": {
        "url": "https://ai-human-chatbot-roasu-backup.koreacentral.inference.ml.azure.com/score",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer BACKUP_TOKEN",
            "azureml-model-deployment": "ai-human-chatbot-roasu-4"
        }
    }
}

# 임시 파일 저장 디렉토리
TEMP_DIR = "temp_data"
os.makedirs(TEMP_DIR, exist_ok=True)

# 전역 변수 (스레드 안전)
class ProcessingState:
    def __init__(self):
        self.is_processing = False
        self._lock = threading.Lock()
    
    def start(self):
        with self._lock:
            self.is_processing = True
    
    def stop(self):
        with self._lock:
            self.is_processing = False
    
    def is_active(self):
        with self._lock:
            return self.is_processing

# 전역 상태 객체 생성
processing_state = ProcessingState()

def query_api(api_name, question, top_k, index_path, metadata_path, timeout=20, max_retries=3):
    """API에 질의하고 응답을 반환 (타임아웃 기본값 20초, 최대 재시도 횟수 추가)"""
    retries = 0
    while retries < max_retries:
        try:
            api_config = API_CONFIGS.get(api_name)
            if api_config is None:
                api_config = API_CONFIGS["Default API"]
                logger.warning(f"'{api_name}' API 설정을 찾을 수 없어 Default API를 사용합니다.")
            
            payload = json.dumps({
                "user_query": question,
                "top_k": top_k,
                "index_path": index_path,
                "metadata_path": metadata_path
            })
            
            response = requests.post(
                api_config["url"],
                headers=api_config["headers"],
                data=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Too Many Requests
                retry_after = random.uniform(1, 3) * (retries + 1)  # 지수 백오프 + 지터 적용
                logger.warning(f"API 속도 제한 감지. {retry_after:.1f}초 후 재시도 (시도 {retries+1}/{max_retries})")
                time.sleep(retry_after)
                retries += 1
                continue
            else:
                error_msg = f"API 응답 오류 (상태 코드: {response.status_code}, 메시지: {response.text})"
                logger.error(error_msg)
                if retries < max_retries - 1:
                    retry_after = random.uniform(1, 2) * (retries + 1)
                    logger.info(f"{retry_after:.1f}초 후 재시도 (시도 {retries+1}/{max_retries})")
                    time.sleep(retry_after)
                    retries += 1
                    continue
                return {"error": error_msg}
        
        except requests.exceptions.Timeout:
            error_msg = f"API 요청 타임아웃 ({timeout}초 초과)"
            logger.warning(error_msg)
            if retries < max_retries - 1:
                retry_after = random.uniform(1, 2) * (retries + 1)
                logger.info(f"{retry_after:.1f}초 후 재시도 (시도 {retries+1}/{max_retries})")
                time.sleep(retry_after)
                retries += 1
                continue
            return {"error": error_msg}
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API 요청 실패: {str(e)}"
            logger.error(error_msg)
            # 다른 API로 시도
            if api_name != "Default API" and retries == 0:
                logger.warning(f"{api_name} API 연결 오류, Default API로 재시도합니다.")
                return query_api("Default API", question, top_k, index_path, metadata_path, timeout, max_retries)
            if retries < max_retries - 1:
                retry_after = random.uniform(2, 4) * (retries + 1)
                logger.info(f"{retry_after:.1f}초 후 재시도 (시도 {retries+1}/{max_retries})")
                time.sleep(retry_after)
                retries += 1
                continue
            return {"error": error_msg}
            
        except Exception as e:
            error_msg = f"예상치 못한 오류: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            if retries < max_retries - 1:
                retry_after = random.uniform(2, 5) * (retries + 1)
                logger.info(f"{retry_after:.1f}초 후 재시도 (시도 {retries+1}/{max_retries})")
                time.sleep(retry_after)
                retries += 1
                continue
            return {"error": error_msg}

def process_response(response):
    """API 응답에서 필요한 정보 추출"""
    if not response:
        return "API 응답 없음", "응답을 받지 못했습니다."
    
    if "error" in response:
        return f"오류: {response['error']}", f"API 요청 중 오류가 발생했습니다: {response['error']}"
    
    if "intent_data" not in response:
        return "Intent 데이터 없음", "응답에 intent_data가 없습니다."
    
    try:
        embedding_top_10 = "\n".join(
            [f"- {match['intent']} (Score: {match.get('score', 'N/A'):.4f})" 
             for match in response["intent_data"]["matches"][:10]]
        )
    
        llm_top_3 = "\n\n".join(
            [textwrap.fill(f"{i+1}. {match['answer']}", width=80) 
             for i, match in enumerate(response["intent_data"]["reranked_matches"][:3])]
        )
    
        return embedding_top_10, llm_top_3
    except Exception as e:
        logger.error(f"응답 처리 오류: {str(e)}\n{traceback.format_exc()}")
        return f"응답 처리 오류: {str(e)}", "응답 데이터 처리 중 오류가 발생했습니다."

def process_batch(batch_data, api_name, top_k, index_path, metadata_path, timeout=20, max_retries=3):
    """배치 데이터 처리 (병렬 처리를 위한 함수)"""
    results = []
    for idx, question in batch_data:
        # session_state 대신 전역 상태 객체 사용
        if not processing_state.is_active():  # 중지 버튼 클릭 시 즉시 반환
            break
        try:
            response = query_api(api_name, str(question), top_k, index_path, metadata_path, timeout, max_retries)
            embedding_top_10, llm_top_3 = process_response(response)
            results.append((idx, embedding_top_10, llm_top_3))
            # 과도한 API 호출 방지를 위한 짧은 지연
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"질문 처리 중 오류 (인덱스 {idx}): {str(e)}\n{traceback.format_exc()}")
            results.append((idx, f"처리 오류: {str(e)}", "데이터 처리 중 오류가 발생했습니다."))
    return results

def get_checkpoint_filename(uploaded_file_name):
    base_name = os.path.splitext(uploaded_file_name)[0]
    return os.path.join(TEMP_DIR, f"{base_name}_checkpoint.pkl")

def save_checkpoint(uploaded_file_name, df, processed_indices):
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
    filename = get_checkpoint_filename(uploaded_file_name)
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                checkpoint_data = pickle.load(f)
            return checkpoint_data
        except Exception as e:
            logger.warning(f"체크포인트 파일 로드 중 오류: {str(e)}\n{traceback.format_exc()}")
            st.warning(f"체크포인트 파일 로드 중 오류: {str(e)}")
    return None

def process_file(api_name, uploaded_file, result_placeholder, stats_placeholder, df_placeholder, question_column, 
                 workers=4, batch_size=10, timeout=20, start_from_checkpoint=False, top_k="", index_path="", metadata_path=""):
    try:
        max_retries = 3  # API 호출 최대 재시도 횟수
        checkpoint_data = None
        processed_indices = set()
        
        # 전역 상태 객체 초기화
        processing_state.start()
        
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
                time.sleep(1)
        
        if checkpoint_data is None:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("지원되지 않는 파일 형식입니다. CSV 또는 XLSX 파일을 업로드해주세요.")
                processing_state.stop()
                return None
            if question_column not in df.columns:
                stats_placeholder.error(f"선택한 열 '{question_column}'이 파일에 존재하지 않습니다. 올바른 열을 선택해주세요.")
                processing_state.stop()
                return None
            if "Embedding Search Top 10" not in df.columns:
                df["Embedding Search Top 10"] = ""
            if "LLM Top 3" not in df.columns:
                df["LLM Top 3"] = ""
        
        progress_bar = st.progress(len(processed_indices) / len(df))
        
        # 처리 중지 버튼
        stop_button_col = st.container()
        stop_button = stop_button_col.button("처리 중지", type="primary", key="stop_processing")
        if stop_button:
            processing_state.stop()
            stats_placeholder.warning("처리 중지 요청됨... 현재 작업 완료 후 중지됩니다.")
        
        # 중지 버튼 상태 모니터링
        def check_stop_button():
            if not processing_state.is_active():
                return True
            return False
        
        start_time = time.time()
        total_rows = len(df)
        remaining_indices = [i for i in range(total_rows) if i not in processed_indices]
        total_remaining = len(remaining_indices)
        
        stats_placeholder.info(f"총 {total_rows}개 질문 중 {total_remaining}개 처리 예정 (병렬 처리: {workers}개 스레드)")
        logger.info(f"처리 시작: 총 {total_rows}개 질문, {total_remaining}개 처리 예정, {workers}개 스레드, 배치 크기 {batch_size}")
        
        # 배치 데이터 생성
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
        
        # 상태 표시 컨테이너
        status_container = st.container()
        batch_progress = status_container.empty()
        batch_progress.text(f"0/{len(batches)} 배치 처리 중...")
        
        # 최신 처리 결과 표시 컨테이너
        latest_results = status_container.empty()
        latest_results.text("아직 처리된 결과가 없습니다...")
        
        # 처리 실패한 질문 저장
        failed_questions = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i in range(min(workers, len(batches))):
                if i < len(batches):
                    futures.append(executor.submit(process_batch, batches[i], api_name, top_k, index_path, metadata_path, timeout, max_retries))
            
            completed_batches = 0
            next_batch_idx = min(workers, len(batches))
            
            # 처리 상태 업데이트 주기
            display_update_interval = 2  # 2개 배치마다 결과 표시 업데이트
            checkpoint_interval = 5      # 5개 배치마다 체크포인트 저장
            
            while futures and processing_state.is_active():
                # 0.1초마다 완료된 작업 확인
                done, not_done = concurrent.futures.wait(
                    futures, 
                    timeout=0.1,
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                # 중지 요청 확인
                if check_stop_button() or not processing_state.is_active():
                    logger.info("사용자가 처리 중지를 요청했습니다.")
                    for future in futures:
                        future.cancel()
                    stats_placeholder.warning(f"처리가 중단되었습니다. {len(processed_indices)}/{total_rows} 질문 처리 완료.")
                    break
                
                # 완료된 작업 처리
                for future in done:
                    try:
                        results = future.result()
                        latest_processed = []
                        
                        for idx, embedding_top_10, llm_top_3 in results:
                            df.at[idx, "Embedding Search Top 10"] = embedding_top_10
                            df.at[idx, "LLM Top 3"] = llm_top_3
                            processed_indices.add(idx)
                            
                            # 오류 발생 여부 확인
                            if "오류" in embedding_top_10 or "오류" in llm_top_3:
                                failed_questions.append((idx, df.iloc[idx][question_column], embedding_top_10))
                            
                            # 최근 처리된 5개 결과만 표시
                            if len(latest_processed) < 5:
                                latest_processed.append({
                                    "인덱스": idx,
                                    "질문": df.iloc[idx][question_column][:50] + "..." if len(df.iloc[idx][question_column]) > 50 else df.iloc[idx][question_column],
                                    "상태": "성공" if "오류" not in embedding_top_10 else "실패"
                                })
                        
                        # 최근 결과 표시
                        if latest_processed:
                            latest_results_df = pd.DataFrame(latest_processed)
                            latest_results.dataframe(latest_results_df, use_container_width=True)
                        
                        # 새로운 배치 시작
                        if next_batch_idx < len(batches) and processing_state.is_active():
                            futures.append(executor.submit(process_batch, batches[next_batch_idx], api_name, top_k, index_path, metadata_path, timeout, max_retries))
                            next_batch_idx += 1
                        
                        completed_batches += 1
                        batch_progress.text(f"{completed_batches}/{len(batches)} 배치 처리 중... ({len(processed_indices)}/{total_rows} 질문)")
                        progress_bar.progress(len(processed_indices) / total_rows)
                        
                        # 주기적으로 결과 표시 업데이트
                        if completed_batches % display_update_interval == 0 or completed_batches == len(batches):
                            display_df = df.copy()
                            df_placeholder.dataframe(display_df, use_container_width=True)
                        
                        # 주기적으로 체크포인트 저장
                        if completed_batches % checkpoint_interval == 0 or completed_batches == len(batches):
                            checkpoint_file = save_checkpoint(uploaded_file.name, df, list(processed_indices))
                            stats_placeholder.info(f"체크포인트 저장됨: {time.strftime('%H:%M:%S')}")
                    
                    except Exception as e:
                        logger.error(f"배치 처리 결과 처리 중 오류: {str(e)}\n{traceback.format_exc()}")
                        stats_placeholder.error(f"배치 처리 결과 처리 중 오류: {str(e)}")
                
                # 진행 중인 작업 목록 업데이트
                futures = list(not_done)
                
                # 진행 상황 및 통계 업데이트
                elapsed_time = time.time() - start_time
                if len(processed_indices) > 0:
                    avg_time_per_question = elapsed_time / len(processed_indices)
                    remaining_questions = total_rows - len(processed_indices)
                    estimated_time_remaining = avg_time_per_question * remaining_questions
                    est_minutes = int(estimated_time_remaining // 60)
                    est_seconds = int(estimated_time_remaining % 60)
                    
                    # 실패율 계산
                    failure_rate = len(failed_questions) / len(processed_indices) * 100 if processed_indices else 0
                    
                    stats_text = (f"진행 상황: {len(processed_indices)}/{total_rows} 질문 ({len(processed_indices)/total_rows*100:.1f}%)\n"
                                  f"경과 시간: {int(elapsed_time//60)}분 {int(elapsed_time%60)}초\n"
                                  f"예상 남은 시간: {est_minutes}분 {est_seconds}초\n"
                                  f"처리 속도: {len(processed_indices)/elapsed_time:.2f} 질문/초\n"
                                  f"오류 발생: {len(failed_questions)}개 ({failure_rate:.1f}%)")
                    stats_placeholder.text(stats_text)
        
        # 처리 완료 후 체크포인트 저장
        if len(processed_indices) > 0:
            save_checkpoint(uploaded_file.name, df, list(processed_indices))
        
        # 처리 결과 요약
        total_time = time.time() - start_time
        result_summary = f"처리가 완료되었습니다! 총 {total_rows}개 질문 중 {len(processed_indices)}개 처리 완료, 소요 시간: {int(total_time//60)}분 {int(total_time%60)}초"
        if failed_questions:
            result_summary += f"\n{len(failed_questions)}개 질문에서 오류가 발생했습니다."
        
        result_placeholder.success(result_summary)
        
        # 최종 결과 표시
        df_placeholder.dataframe(df, use_container_width=True)
        
        # 실패한 질문 목록 표시
        if failed_questions:
            st.subheader(f"오류 발생 질문 ({len(failed_questions)}개)")
            failed_df = pd.DataFrame([(idx, question, error) for idx, question, error in failed_questions], 
                                    columns=["인덱스", "질문", "오류 메시지"])
            st.dataframe(failed_df, use_container_width=True)
        
        processing_state.stop()
        return df
    
    except Exception as e:
        logger.error(f"처리 중 예외 발생: {str(e)}\n{traceback.format_exc()}")
        try:
            if 'df' in locals() and 'processed_indices' in locals():
                save_checkpoint(uploaded_file.name, df, list(processed_indices))
                stats_placeholder.info("오류 발생 시점까지의 진행상황이 저장되었습니다.")
        except Exception as save_error:
            logger.error(f"체크포인트 저장 중 추가 오류: {str(save_error)}")
        
        stats_placeholder.error(f"처리 중 오류가 발생했습니다: {str(e)}")
        processing_state.stop()
        return None

def process_single_question(api_name, question, top_k, index_path, metadata_path, timeout=20):
    """단일 질문을 처리하고 결과 반환 (타임아웃 기본값 20초)"""
    with st.spinner("API 요청 처리 중..."):
        response = query_api(api_name, question, top_k, index_path, metadata_path, timeout)
        if response:
            embedding_top_10, llm_top_3 = process_response(response)
            return embedding_top_10, llm_top_3
        else:
            return "API 응답 오류 발생", "응답을 받지 못했습니다."

# Streamlit 앱 메인 함수
st.title("API 테스트 자동화 툴")

# Streamlit 앱 초기화 - session_state 제거하고 전역 상태 사용

# 사이드바 설정: API 선택 (URI와 헤더는 고정)
st.sidebar.header("API 설정")
api_name = st.sidebar.selectbox(
    "사용할 API 선택",
    list(API_CONFIGS.keys()),
    index=0
)
st.sidebar.info(f"사용 중인 API URL: {API_CONFIGS[api_name]['url']}")

# 추가 파라미터 설정 (top_k, index_path, metadata_path)
st.sidebar.header("추가 파라미터 설정")
top_k = st.sidebar.text_input("top_k", value="10")
index_path = st.sidebar.text_input("index_path", value="faiss_index/embedding dataset-0314-Hans-0313_v20250314_222213.index")
metadata_path = st.sidebar.text_input("metadata_path", value="faiss_index/embedding dataset-0314-Hans-0313_v20250314_222213.csv")

# 고급 설정
st.sidebar.header("고급 설정")
timeout = st.sidebar.slider("API 타임아웃 (초)", min_value=5, max_value=60, value=20, step=5)
workers = st.sidebar.slider("병렬 처리 스레드 수", min_value=1, max_value=16, value=4, step=1)
batch_size = st.sidebar.slider("배치 크기", min_value=1, max_value=20, value=5, step=1)

# 병렬 처리 권장 설정 안내
if workers > 8:
    st.sidebar.warning("병렬 처리 스레드가 너무 많으면 API 제한이나 성능 저하가 발생할 수 있습니다. 4-8개 사이를 권장합니다.")
    
if batch_size > 10:
    st.sidebar.warning("배치 크기가 클수록 중간 결과 저장 빈도가 줄어들어 오류 발생 시 더 많은 작업이 손실될 수 있습니다.")

# 탭 생성: 단일 질문과 대량 질문 처리
tab1, tab2 = st.tabs(["단일 질문", "대량 질문 (Excel)"])

with tab1:
    st.subheader("단일 질문 테스트")
    st.write("질문을 입력하고 API 응답을 바로 확인하세요.")
    question = st.text_area("질문 입력", height=100, placeholder="여기에 질문을 입력하세요...")
    if st.button("질문 제출"):
        if question:
            embedding_top_10, llm_top_3 = process_single_question(api_name, question, top_k, index_path, metadata_path, timeout)
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
    st.write("CSV 또는 XLSX 파일을 업로드하여 여러 질문을 한번에 처리하세요.")
    
    # 파일 업로드 섹션
    uploaded_file = st.file_uploader("CSV 또는 XLSX 파일 업로드", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        st.write("파일이 업로드되었습니다:", uploaded_file.name)
        checkpoint_exists = os.path.exists(get_checkpoint_filename(uploaded_file.name))
        
        try:
            # 파일 미리보기
            if uploaded_file.name.endswith(".csv"):
                preview_df = pd.read_csv(uploaded_file)
            else:
                preview_df = pd.read_excel(uploaded_file)
                
            st.subheader("파일 미리보기")
            st.dataframe(preview_df.head(5))
            
            # 질문 열 선택
            question_column = st.selectbox(
                "질문이 포함된 열을 선택하세요",
                options=preview_df.columns.tolist(),
                index=0
            )
            
            # 체크포인트 사용 여부
            use_checkpoint = False
            if checkpoint_exists:
                use_checkpoint = st.checkbox("이전 진행 상황에서 계속하기", value=True)
                if use_checkpoint:
                    st.info("저장된 체크포인트에서 처리를 재개합니다. 이전에 처리된 질문은 건너뜁니다.")
            
            # 결과 컨테이너 준비
            result_placeholder = st.empty()
            stats_placeholder = st.empty()
            df_placeholder = st.empty()
            download_placeholder = st.empty()
            
            # 처리 시작 버튼
            if st.button("처리 시작", type="primary"):
                # 처리 시작 - 전역 상태 객체 사용
                uploaded_file.seek(0)
                
                # 파일 처리 함수 호출
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
                    top_k=top_k,
                    index_path=index_path,
                    metadata_path=metadata_path
                )
                
                # 결과 저장 및 다운로드 옵션
                if df is not None:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False)
                    excel_data = output.getvalue()
                    
                    download_placeholder.download_button(
                        label="결과 Excel 파일 다운로드",
                        data=excel_data,
                        file_name="test_results.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                    
                    # 추가 통계 정보
                    success_count = sum(1 for col in df["Embedding Search Top 10"] if "오류" not in str(col))
                    error_count = len(df) - success_count
                    
                    st.subheader("처리 통계")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("총 질문 수", len(df))
                    col2.metric("성공", success_count)
                    col3.metric("실패", error_count)
                    
        except Exception as e:
            logger.error(f"파일 읽기 오류: {str(e)}\n{traceback.format_exc()}")
            st.error(f"파일 읽기 오류: {str(e)}")
