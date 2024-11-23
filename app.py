import streamlit as st
from data_loader import load_documents
from vectorizer import vectorize_questions, vectorize_documents, build_vector_db, get_vector_model
from retriever import retrieve_documents
from answer_generator import generate_final_answer
import os
from dotenv import load_dotenv
#import numpy as np
import faiss
import pickle
#import time
from functools import lru_cache
import hashlib
from rank_bm25 import BM25Okapi

# 加载环境变量
load_dotenv()

# 显示当前工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))

print(f"AZURE_OPENAI_ENDPOINT in app.py: {os.getenv('AZURE_OPENAI_ENDPOINT')}")

# 定义索引文件路径
INDEX_FILE_QUESTIONS = "vector_index_questions.faiss"
INDEX_FILE_DOCUMENTS = "vector_index_documents.faiss"
MAPPING_FILE = "question_to_doc_mapping.pkl"
PREVIOUS_QUESTIONS_FILE = "previous_questions.pkl"

@st.cache_resource
def get_generate_questions():
    """使用 Streamlit 缓存机制加载 generate_questions 函数"""
    from question_generator import generate_questions
    return generate_questions

@st.cache_data
def process_documents_with_questions(documents):
    """
    处理文档并生成问题，使用 Streamlit 缓存避免重复处理
    """
    print("开始处理文档和生成问题...")
    print(f"文档数量: {len(documents)}")
    
    generate_questions = get_generate_questions()
    questions = []
    question_to_doc_mapping = {}
    
    for doc in documents:
        generated_questions = generate_questions([doc])
        questions.extend(generated_questions)
        question_to_doc_mapping.update({q: doc["id"] for q in generated_questions})
    
    print("文档处理完成")
    return questions, question_to_doc_mapping

def verify_data_consistency():
    """��证数据一致性"""
    files_to_check = {
        "questions.pkl": (list, "问题列表"),
        "documents.pkl": (list, "文档列表"),
        "question_to_doc_mapping.pkl": (dict, "问题文档映射"),
        "vector_index_questions.faiss": (None, "问题向量索引"),
        "vector_index_documents.faiss": (None, "文档向量索引")
    }
    
    for filename, (expected_type, desc) in files_to_check.items():
        filepath = os.path.join(current_dir, filename)
        if not os.path.exists(filepath):
            print(f"警告: {desc} 文件不存在: {filepath}")
            continue
            
        if expected_type:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                if not isinstance(data, expected_type):
                    print(f"警告: {desc} 类型不正确")
                print(f"{desc} 大小: {len(data)}")

def generate_file_hash(file_path):
    """生成文件的SHA256哈希值。"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    # 添加一个清理缓存的按钮
    if st.sidebar.button("清理缓存"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.pop('vectorizer_params', None)  # 清除存储的 vectorizer_params
        st.success("缓存已清理")

    verify_data_consistency()
    st.title("文档问答系统")

    # 创建选项卡
    tabs = st.tabs(["构建向量库", "查询", "调试"])

    with tabs[0]:
        st.header("构建向量库")

        # 上传文档
        uploaded_files = st.file_uploader("上传文档 (CSV, PDF, TXT)", type=["csv", "pdf", "txt"], accept_multiple_files=True)
        
        # 添加 chunk_sizes_input 的输入框
        chunk_sizes_input = st.text_input("输入 chunk_size 值，用逗号分隔", value="500,1000")
    
        try:
            chunk_sizes = [int(size.strip()) for size in chunk_sizes_input.split(",") if size.strip().isdigit()]
            if not chunk_sizes:
                st.error("请至少输入一个有效的 chunk_size 值。")
        except ValueError:
            st.error("请确保所有 chunk_size 值都是整数。")
        
        # 添加模型选择选项
        model_type = st.selectbox("选择向量化模型类型", options=["local", "azure_openai"], index=0)
        
        if model_type == 'local':
            model_name = st.text_input("输入本地模型名称", value="all-mpnet-base-v2")
            device = st.selectbox("选择向量化设备", options=["cpu", "cuda"], index=0)
            vectorizer_params = {
                "model_type": model_type,
                "model_name": model_name,
                "device": device
            }
        elif model_type == 'azure_openai':
            azure_api_key = st.text_input("输入 Azure OpenAI API 密钥", type="password")
            azure_endpoint = st.text_input("输入 Azure OpenAI 端点")
            azure_api_version = st.text_input("输入 Azure OpenAI API 版本", value="2023-05-15")
            azure_embedding_model = st.text_input("输入 Azure OpenAI 嵌入模型名称", value="text-embedding-ada-002")
            vectorizer_params = {
                "model_type": model_type,
                "api_key": azure_api_key,
                "api_base": azure_endpoint,
                "api_version": azure_api_version,
                "embedding_model": azure_embedding_model
            }
        else:
            st.error("请选择有效的模型类型。")
            vectorizer_params = {}
        
        # 检查是否切换了模型类型，若切换则清除相关缓存
        if 'vectorizer_params' in st.session_state:
            if st.session_state['vectorizer_params'].get("model_type") != vectorizer_params.get("model_type"):
                st.cache_resource.clear()
                st.cache_data.clear()
                st.session_state.pop('vectorizer_params', None)
                st.success("模型类型已更改，相关缓存已清理。")
        
        # 存储 vectorizer_params 到 session state
        if model_type and (vectorizer_params.get("api_key") or model_type == 'local'):
            st.session_state['vectorizer_params'] = vectorizer_params
        
        # 添加一个复选框，让用户选择是否在构建向量库时构建 BM25 模型
        use_bm25 = st.checkbox("构�� BM25 模型", value=False)
        
        if uploaded_files and chunk_sizes and model_type:
            # 保存上传的文件到 'documents' 目录
            documents_dir = os.path.join(current_dir, "documents")
            os.makedirs(documents_dir, exist_ok=True)

            # 加载已处理的文件名和哈希列表
            processed_files_path = os.path.join(documents_dir, "processed_files.txt")
            processed_hashes_path = os.path.join(documents_dir, "processed_hashes.txt")

            if os.path.exists(processed_files_path):
                with open(processed_files_path, "r") as pf:
                    processed_files = set(line.strip() for line in pf.readlines())
            else:
                processed_files = set()

            if os.path.exists(processed_hashes_path):
                with open(processed_hashes_path, "r") as ph:
                    processed_hashes = set(line.strip() for line in ph.readlines())
            else:
                processed_hashes = set()

            new_uploaded_files = []
            duplicate_files = []
            missing_files = []

            # 定义支持的文件类型
            supported_extensions = {".csv", ".pdf", ".txt"}

            for uploaded_file in uploaded_files:
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                if file_ext not in supported_extensions:
                    st.warning(f"跳过不支持的文件类型: {uploaded_file.name}")
                    continue  # 跳过不支持的文件类型

                file_path = os.path.join(documents_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                file_hash = generate_file_hash(file_path)

                if file_hash in processed_hashes:
                    if not os.path.exists(file_path):
                        missing_files.append(uploaded_file.name)
                        os.remove(file_path)  # 删除缺失的文件（如果存在）
                    else:
                        duplicate_files.append((uploaded_file.name, file_hash))
                        # 不立即删除文件，等待用户决定是否重新处理
                else:
                    new_uploaded_files.append((uploaded_file.name, file_hash))

            if duplicate_files:
                st.warning("以下文件已被处理过：")
                for fname, _ in duplicate_files:
                    st.write(f"- {fname}")
                
                # 添加复选框让用户选择要重新处理的文件
                files_to_reprocess = []
                for fname, fhash in duplicate_files:
                    if st.checkbox(f"是否重新处理文件: {fname}", key=f"reprocess_{fname}"):
                        files_to_reprocess.append((fname, fhash))
                
                if st.button("重新处理选中的重复文件"):
                    for fname, fhash in files_to_reprocess:
                        file_path = os.path.join(documents_dir, fname)
                        # 重新生成文件哈希
                        if os.path.exists(file_path):
                            new_hash = generate_file_hash(file_path)
                            new_uploaded_files.append((fname, new_hash))
                        else:
                            st.error(f"文件不存在，无法重新处理: {fname}")
                    # 移除被重新处理的文件从 duplicate_files
                    duplicate_files = [item for item in duplicate_files if item not in files_to_reprocess]
                
                # 删除未重新处理的重复文件
                for fname, _ in duplicate_files:
                    try:
                        os.remove(os.path.join(documents_dir, fname))
                        st.write(f"已删除文件: {fname}")
                    except Exception as e:
                        st.error(f"无法删除文件 {fname}: {e}")

            if missing_files:
                st.error("以下文件记录显示已处理，但实际文件不存在：")
                for fname in missing_files:
                    st.write(f"- {fname}")
                
                if st.button("移除缺失文件的记录"):
                    for fname in missing_files:
                        # 移除缺失文���的哈希记录
                        to_remove = [hash for name, hash in processed_files if name == fname]
                        for h in to_remove:
                            processed_hashes.discard(h)
                            processed_files.discard(fname)
                    st.success("缺失文件的记录已移除。")

            if new_uploaded_files:
                for fname, fhash in new_uploaded_files:
                    file_path = os.path.join(documents_dir, fname)
                    processed_files.add(fname)
                    processed_hashes.add(fhash)
                    st.write(f"保存文件: {file_path}")
                st.success("新文档上传成功！")

                # 更新已处理的文件列表和哈希列表
                with open(processed_files_path, "w") as pf:
                    for fname in processed_files:
                        pf.write(f"{fname}\n")
                with open(processed_hashes_path, "w") as ph:
                    for fhash in processed_hashes:
                        ph.write(f"{fhash}\n")

            try:
                # 加载已有的文档、问题和映射
                questions_file_path = os.path.join(current_dir, "questions.pkl")
                documents_file_path = os.path.join(current_dir, "documents.pkl")
                mapping_file_path = os.path.join(current_dir, MAPPING_FILE)

                if os.path.exists(questions_file_path):
                    with open(questions_file_path, "rb") as f:
                        existing_questions = pickle.load(f)
                else:
                    existing_questions = []

                if os.path.exists(documents_file_path):
                    with open(documents_file_path, "rb") as f:
                        existing_documents = pickle.load(f)
                else:
                    existing_documents = []

                if os.path.exists(mapping_file_path):
                    with open(mapping_file_path, "rb") as f:
                        existing_mapping = pickle.load(f)
                else:
                    existing_mapping = {}

                # 加载新文档并传递多个 chunk_size
                new_documents = load_documents(documents_dir, chunk_sizes=chunk_sizes)
                if not new_documents:
                    st.info("没有新的文档需要处理。")
                    return  # 如果没有新文档，直接返回

                # 合并新旧文档
                all_documents = existing_documents + new_documents

                # 处理新文档并生成问题
                new_questions, new_question_to_doc_mapping = process_documents_with_questions(new_documents)
                if not new_questions:
                    st.info("没有新的问题需要向量化。")
                    return  # 如果没有新问题，直接返回

                # 合并新旧问题和映射
                all_questions = existing_questions + new_questions
                all_question_to_doc_mapping = {**existing_mapping, **new_question_to_doc_mapping}

                # 向量化所有问题和文档
                print("开始向量化所有问题...")
                question_vectors = vectorize_questions(all_questions, **vectorizer_params)
                if question_vectors.size == 0:
                    st.error("没有可向量化的问题。")
                    return  # 如果问题向量为空，避免后续错误

                print("开始向量化所有文档...")
                document_vectors = vectorize_documents(all_documents, **vectorizer_params)
                if document_vectors.size == 0:
                    st.error("没���可向量化的文档。")
                    return  # 如果文档向量为空，避免后续��误

                # 验证向量维度
                if question_vectors.shape[1] != document_vectors.shape[1]:
                    st.error("问题向量和文档向量的维度不一致。")
                    return

                # 构建向量数据库
                indices = build_vector_db(question_vectors, document_vectors)
                st.success("向量数据库构建成功！")

                # 保存索引
                index_file_path_q = os.path.join(current_dir, INDEX_FILE_QUESTIONS)
                index_file_path_d = os.path.join(current_dir, INDEX_FILE_DOCUMENTS)
                faiss.write_index(indices['questions'], index_file_path_q)
                faiss.write_index(indices['documents'], index_file_path_d)

                # 保存映射
                mapping_file_path = os.path.join(current_dir, MAPPING_FILE)
                with open(mapping_file_path, "wb") as f:
                    pickle.dump(all_question_to_doc_mapping, f)

                # 保存问题和文档
                questions_file_path = os.path.join(current_dir, "questions.pkl")
                documents_file_path = os.path.join(current_dir, "documents.pkl")
                with open(questions_file_path, "wb") as f:
                    pickle.dump(all_questions, f)
                with open(documents_file_path, "wb") as f:
                    pickle.dump(all_documents, f)

                # 保存之前的问题列表
                previous_questions_file_path = os.path.join(current_dir, PREVIOUS_QUESTIONS_FILE)
                with open(previous_questions_file_path, "wb") as f:
                    pickle.dump(all_questions, f)

                # 构建或更新 BM25 模型
                bm25_model_file_path = os.path.join(current_dir, 'bm25_model.pkl')
                if use_bm25:
                    print("构建 BM25 模型...")
                    tokenized_corpus = [doc['text'].split() for doc in all_documents]
                    bm25_model = BM25Okapi(tokenized_corpus)
                    with open(bm25_model_file_path, 'wb') as f:
                        pickle.dump(bm25_model, f)
                    st.success("BM25模型已构建并保存。")
                else:
                    # 如果存在旧的 BM25 模型文件，则删除
                    if os.path.exists(bm25_model_file_path):
                        os.remove(bm25_model_file_path)
                        print("BM25 模型文件已删除。")

                # 删除上传的文件以防重复处理
                for uploaded_file in uploaded_files:
                    try:
                        os.remove(os.path.join(documents_dir, uploaded_file.name))
                    except Exception as e:
                        st.error(f"无法删除文件 {uploaded_file.name}: {e}")

                # 显示统计信息
                st.write(f"总问题数: {len(all_questions)}")
                st.write(f"FAISS 问题索引总数: {indices['questions'].ntotal}")
                st.write(f"FAISS 文档索引总数: {indices['documents'].ntotal}")

            except Exception as e:
                st.error(f"处理文档和生成向量时发生错误: {e}")

    with tabs[1]:
        st.header("查询")

        # 检索 vectorizer_params 从 session state
        if 'vectorizer_params' not in st.session_state:
            st.error("请先在 '构建向量库' 页签中配置向量化参数。")
            return
        vectorizer_params = st.session_state['vectorizer_params']

        # 添加一个复选框，让用户选择在检索中是否使用 BM25
        use_bm25 = st.checkbox("在检索中使用 BM25", value=False)

        # 文件路径
        index_file_path_q = os.path.join(current_dir, INDEX_FILE_QUESTIONS)
        index_file_path_d = os.path.join(current_dir, INDEX_FILE_DOCUMENTS)
        mapping_file_path = os.path.join(current_dir, MAPPING_FILE)
        questions_file_path = os.path.join(current_dir, "questions.pkl")
        documents_file_path = os.path.join(current_dir, "documents.pkl")
        bm25_model_file_path = os.path.join(current_dir, 'bm25_model.pkl')

        # 用户输入
        user_input = st.text_input("请输入您的问题：")

        # 检查必要文件是否存在
        missing_files = []
        required_files = [
            index_file_path_q,
            index_file_path_d,
            mapping_file_path,
            questions_file_path,
            documents_file_path
        ]
        if use_bm25:
            required_files.append(bm25_model_file_path)

        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)

        if missing_files:
            st.warning("缺失以下文件，无法进行查询：")
            for file in missing_files:
                st.write(f"- {file}")
            st.info("请在 '构建向量库' 页签重新构建向量库并确保 BM25 模型已构建（如果需要）。")
            # 不返回，继续显示用户输入框和按钮

        else:
            # 加载索引和映射
            try:
                index_questions = faiss.read_index(index_file_path_q)
                index_documents = faiss.read_index(index_file_path_d)
                with open(mapping_file_path, "rb") as f:
                    question_to_doc_mapping = pickle.load(f)
                with open(questions_file_path, "rb") as f:
                    questions = pickle.load(f)
                with open(documents_file_path, "rb") as f:
                    documents = pickle.load(f)

                # 仅当使用 BM25 时加载 BM25 模型
                if use_bm25:
                    with open(bm25_model_file_path, 'rb') as f:
                        bm25_model = pickle.load(f)
                    st.write("BM25模型已加载。")
                else:
                    bm25_model = None

                st.write(f"加载了 {len(questions)} 个问题。")
                st.write(f"加载了 {len(documents)} 个文档块。")
                st.write(f"FAISS 问题索引总数: {index_questions.ntotal}")
                st.write(f"FAISS 文档索引总数: {index_documents.ntotal}")

                if index_questions.ntotal != len(questions):
                    st.error("加载的 FAISS 问题索引总数与问题列表长度不匹配。")
                if index_documents.ntotal != len(documents):
                    st.error("加载的 FAISS 文档索引总数与文档列表长度不匹配。")

            except Exception as e:
                st.error(f"加载文件时发生错误: {e}")
                return

            if st.button("获取答案") and user_input:
                with st.spinner("正在检索相关文档..."):
                    try:
                        results = retrieve_documents(
                            user_query=user_input,
                            index_questions=index_questions,
                            index_documents=index_documents,
                            questions=questions,
                            question_to_doc_mapping=question_to_doc_mapping,
                            documents=documents,
                            vectorizer_params=vectorizer_params,
                            bm25_model=bm25_model
                        )
                    except Exception as e:
                        st.error(f"检索过程中发生错误: {e}")
                        return

                if not results["related_questions"] and not results["related_documents"]:
                    st.info("未找到相关的答案。请尝试其他问题。")
                else:
                    try:
                        relevant_docs = [res["text"] for res in results["related_documents"]]
                        context = "\n\n".join(relevant_docs)
                        answer = generate_final_answer(context, user_input)
                        st.write(f"**最终答案：** {answer}")
                    except Exception as e:
                        st.error(f"生成答案失败: {e}")

    with tabs[2]:
        st.header("调试")
        # 仅保留显示之前生成的问题列表
        previous_questions_file_path = os.path.join(current_dir, PREVIOUS_QUESTIONS_FILE)
        if os.path.exists(previous_questions_file_path):
            with open(previous_questions_file_path, "rb") as f:
                previous_questions = pickle.load(f)
            st.write("**之前生成的问题列表:**")
            for q in previous_questions:
                st.write(f"- {q}")
        else:
            st.write("没有之前的问题列表可供显示。")

if __name__ == "__main__":
    main()