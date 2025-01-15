import os
import platform
import openai
import chromadb
import langchain
import tiktoken

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import TextLoader

os.environ["OPENAI_API_KEY"] = "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"

from langchain.document_loaders import PyPDFLoader

# 파일명: app.py
import streamlit as st

# 제목 및 설명 추가
st.title("이온엠 스터디 챗봇")

left_side, right_side = st.columns(2)
st.markdown("""
    <style>
    .stMainBlockContainer {
        max-width: 1350px; /* 원하는 너비 값으로 변경 */
    }
    h1 {
        display: flex;
        justify-content: center;
        align-items: center; 
    }
    .stColumn {
        border: 2px solid black;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)
left_side.write("입력")
right_side.write("결과")

# 입력받기
path_title = left_side.write("1. PDF 파일 경로")
# 파일 업로드 위젯
uploaded_file = left_side.file_uploader("파일을 선택하세요!", type=["pdf"])
pdf_file_name = ''
# 파일이 업로드되었는지 확인
if uploaded_file is not None:
    pdf_file_name = uploaded_file.name
    # 파일의 내용을 읽어 처리
    file_content = uploaded_file.read()

rcts_title = left_side.write("2. ReculsiveCharacterTextSplitter param 적용")
_1st, _2nd = left_side.columns(2)
rcts_sz = _1st.number_input("chunk_size", min_value=1000, max_value=3000)
rcts_ovlp = _2nd.number_input("chunk_overlap", min_value=0, max_value=500)

coa_title = left_side.write("3. ChatOpenAI param 적용")
_1st, _2nd = left_side.columns(2)
coa_temp = _1st.number_input("temperature", min_value=0, max_value=1)
coa_m_tk = _1st.number_input("max_token", min_value=1, max_value=3000)
coa_md_nm = _2nd.selectbox("model_name", ["gpt-3.5-turbo", "gpt-4"])

cvdb_title = left_side.write("4. ChatVectorDBChain param 적용")
cvdb_return = left_side.selectbox("원본문서 반환여부", [True, False])

querys_title = left_side.write("5. 쿼리 입력")
querys_cnts = left_side.text_input("질문")

# 버튼 생성
if left_side.button('클릭하세요'):
    # PDF 파일 경로
    pdf_file_path = "C:\\Users\\NTH\\PycharmProjects\\pythonProject5\\.venv\\이온엠_전체메뉴얼\\이온엠_전체메뉴얼\\"+pdf_file_name

    # PDFLoader를 사용하여 PDF 파일 로드
    loader = PyPDFLoader(pdf_file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=rcts_sz, chunk_overlap=rcts_ovlp)
    doc = text_splitter.split_documents(data)

    persist_directory = "/content/drive/My Drive/Colab Notebooks/chroma/romeo"

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(data, embeddings, persist_directory=persist_directory)
    vectordb.persist()

    model = ChatOpenAI(temperature=coa_temp, max_tokens=coa_m_tk, model_name=coa_md_nm)
    chain = ChatVectorDBChain.from_llm(model, vectordb, return_source_documents=cvdb_return)

    query = querys_cnts
    result = chain({"question": query, "chat_history": []})

    # 텍스트 영역 생성
    if cvdb_return:
        right_side.text_area("질의 결과:", result["answer"], height=250)
        right_side.text_area("참조 페이지:", result["source_documents"][0], height=600)
    else:
        right_side.text_area("질의 결과:", result["answer"], height=850)

