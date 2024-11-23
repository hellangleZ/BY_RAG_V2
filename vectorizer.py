from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from functools import lru_cache
import openai  # 导入OpenAI库
from openai import AzureOpenAI  # 添加 AzureOpenAI 导入
import os  # 导入 os 模块用于获取环境变量

def get_vector_model(model_type='local', model_name='all-mpnet-base-v2', device='cpu', api_base=None, api_version='2023-05-15', api_key=None, embedding_model='text-embedding-ada-002'):
    """
    根据指定的模型类型初始化并返回向量化模型。
    
    参数:
        model_type (str): 模型类型，'local' 或 'azure_openai'。
        model_name (str): 本地模型名称。
        device (str): 设备类型，'cpu' 或 'cuda'。
        api_base (str): Azure OpenAI API 基础URL。
        api_version (str): Azure OpenAI API 版本。
        api_key (str): Azure OpenAI API 密钥。
        embedding_model (str): Azure OpenAI 嵌入模型名称。
    
    返回:
        object: 初始化后的模型对象。
    """
    if model_type == 'local':
        return SentenceTransformer(model_name, device=device)
    elif model_type == 'azure_openai':
        openai.api_type = "azure"
        openai.api_base = api_base
        openai.api_version = api_version
        openai.api_key = api_key
        return embedding_model
    else:
        raise ValueError("Unsupported model_type. Choose 'local' or 'azure_openai'.")

def get_azure_client(azure_api_key, azure_endpoint, azure_api_version):
    """
    初始化 AzureOpenAI 客户端。
    
    参数:
        azure_api_key (str): Azure OpenAI API 密钥。
        azure_endpoint (str): Azure OpenAI 端点。
        azure_api_version (str): Azure OpenAI API 版本。
    
    返回:
        AzureOpenAI: 初始化的 Azure OpenAI 客户端。
    """
    return AzureOpenAI(
        api_key=azure_api_key,
        api_version=azure_api_version,
        azure_endpoint=azure_endpoint
    )

def generate_embeddings(text, model, azure_api_key, azure_endpoint, azure_api_version):
    """
    使用 AzureOpenAI 客户端生成嵌入向量。
    
    参数:
        text (str): 要生成嵌入的文本。
        model (str): 使用的模型名称（部署名称）。
        azure_api_key (str): Azure OpenAI API 密钥。
        azure_endpoint (str): Azure OpenAI 端点。
        azure_api_version (str): Azure OpenAI API 版本。
    
    返回:
        list: 嵌入向量。
    """
    client = get_azure_client(azure_api_key, azure_endpoint, azure_api_version)
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def vectorize_questions(questions, model_type='local', **kwargs):
    """
    向量化问题列表。
    
    参数:
        questions (list of str): 要向量化的问题列表。
        model_type (str): 模型类型，'local' 或 'azure_openai'。
        **kwargs: 其他模型参数，包括 Azure 配置。
    
    返回:
        numpy.ndarray: 问题的向量表示。
    """
    if not questions:
        return np.empty((0, 768), dtype='float32')  # 768是默认维度，需根据实际模型调整
    
    if model_type == 'local':
        vector_model = get_vector_model(model_type, **kwargs)
        vectors = vector_model.encode(questions)
    elif model_type == 'azure_openai':
        azure_api_key = kwargs.get('api_key')
        azure_endpoint = kwargs.get('api_base')
        azure_api_version = kwargs.get('api_version', '2023-05-15')
        embedding_model = kwargs.get('embedding_model', 'text-embedding-ada-002')
        vectors = [
            generate_embeddings(q, model=embedding_model, azure_api_key=azure_api_key, 
                               azure_endpoint=azure_endpoint, azure_api_version=azure_api_version) 
            for q in questions
        ]
        vectors = np.array(vectors)
    else:
        raise ValueError("Unsupported model_type. Choose 'local' or 'azure_openai'.")
    
    print(f"问题向量维度: {vectors.shape}")  # 添加维度信息日志
    
    if isinstance(vectors, np.ndarray):
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)  # 修复单个向量的维度问题
    else:
        raise TypeError("向量化结果不是 numpy.ndarray")
    
    return vectors.astype('float32')  # 确保向量类型为 float32，FAISS 更兼容

def vectorize_documents(documents, model_type='local', **kwargs):
    """
    向量化文档块列表。
    
    参数:
        documents (list of dict): 包含文档文本的列表。
        model_type (str): 模型类型，'local' 或 'azure_openai'。
        **kwargs: 其他模型参数，包括 Azure 配置。
    
    返回:
        numpy.ndarray: 文档的向量表示。
    """
    texts = [doc["text"] for doc in documents]
    
    if not texts:
        return np.empty((0, 768), dtype='float32')  # 768是默认维度，需根据实际模型调整
    
    if model_type == 'local':
        vector_model = get_vector_model(model_type, **kwargs)
        vectors = vector_model.encode(texts)
    elif model_type == 'azure_openai':
        azure_api_key = kwargs.get('api_key')
        azure_endpoint = kwargs.get('api_base')
        azure_api_version = kwargs.get('api_version', '2023-05-15')
        embedding_model = kwargs.get('embedding_model', 'text-embedding-ada-002')
        vectors = [
            generate_embeddings(text, model=embedding_model, azure_api_key=azure_api_key, 
                               azure_endpoint=azure_endpoint, azure_api_version=azure_api_version) 
            for text in texts
        ]
        vectors = np.array(vectors)
    else:
        raise ValueError("Unsupported model_type. Choose 'local' or 'azure_openai'.")
    
    print(f"文档向量维度: {vectors.shape}")  # 添加维度信息日志
    
    if isinstance(vectors, np.ndarray):
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)  # 修复单个向量的维度问题
    else:
        raise TypeError("向量化结果不是 numpy.ndarray")
    
    return vectors.astype('float32')  # 确保向量类型为 float32，FAISS 更兼容

def build_vector_db(question_vectors, document_vectors):
    """
    构建问题和文档的向量数据库。

    参数:
        question_vectors (numpy.ndarray): 问题的向量表示。
        document_vectors (numpy.ndarray): 文档的向量表示。

    返回:
        dict: 包含问题索引和文档索引的字典。
    """
    if not isinstance(question_vectors, np.ndarray) or not isinstance(document_vectors, np.ndarray):
        raise TypeError("所有向量必须是 numpy.ndarray")
    
    indices = {}
    
    # 构建问题索引
    if question_vectors.size > 0:
        embedding_dim_q = question_vectors.shape[1]
        index_q = faiss.IndexFlatL2(embedding_dim_q)
        
        # 在添加向量之前打印维度信息
        print(f"问题向量维度: {question_vectors.shape}")
        print(f"FAISS 问题索引维度: {embedding_dim_q}")
        
        index_q.add(question_vectors)
        
        # 验证添加是否成功
        if index_q.ntotal != question_vectors.shape[0]:
            raise ValueError(f"FAISS 问题索引的向量数量 ({index_q.ntotal}) 与输入的向量数量 ({question_vectors.shape[0]}) 不匹配")
        
        indices['questions'] = index_q
    else:
        raise ValueError("没有可添加到FAISS索引的问题向量")
    
    # 构建文档索引
    if document_vectors.size > 0:
        embedding_dim_d = document_vectors.shape[1]
        index_d = faiss.IndexFlatL2(embedding_dim_d)
        
        # 在添加向量之前打印维度信息
        print(f"文档向量维度: {document_vectors.shape}")
        print(f"FAISS 文档索引维度: {embedding_dim_d}")
        
        index_d.add(document_vectors)
        
        # 验证添加是否成功
        if index_d.ntotal != document_vectors.shape[0]:
            raise ValueError("文档向量添加到 FAISS 索引失败")
        
        indices['documents'] = index_d
    else:
        raise ValueError("没有可添加到FAISS索引的文档向量")
    
    # 打印最终的索引信息
    print(f"构建完成 - 问题索引总数: {indices['questions'].ntotal}")
    print(f"构建完成 - 文档索引总数: {indices['documents'].ntotal}")
    
    return indices

def verify_vector_consistency(vectors, expected_count, name="向量"):
    """
    验证向量的一致性。

    参数:
        vectors (numpy.ndarray): 要验证的向量。
        expected_count (int): 期望的向量数量。
        name (str): 向量的名称，用于日志输出。
    
    """
    if not isinstance(vectors, np.ndarray):
        raise TypeError(f"{name}必须是 numpy.ndarray 类型")
    
    if vectors.ndim != 2:
        raise ValueError(f"{name}必须是二维数组，当前维度: {vectors.ndim}")
    
    if vectors.shape[0] != expected_count:
        raise ValueError(f"{name}数量 ({vectors.shape[0]}) 与期望数量 ({expected_count}) 不匹配")
    
    if not np.isfinite(vectors).all():
        raise ValueError(f"{name}包含无效值 (NaN 或 Inf)")
    
    print(f"{name}验证通过 - 形状: {vectors.shape}")
