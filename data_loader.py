import os  # 导入os模块用于文件操作
import csv  # 导入csv模块用于处理CSV文件
import PyPDF2  # 导入PyPDF2用于PDF文件处理

def load_documents_from_csv(file_path):
    """
    从CSV文件加载文档。

    参数:
        file_path (str): CSV文件的路径。

    返回:
        list of dict: 包含文档ID和文本的字典列表。
    """
    documents = []  # 初始化文档列表
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)  # 创建CSV字典读取器
        for row in reader:
            documents.append({"id": row["id"], "text": row["text"]})  # 添加每行数据到文档列表
    return documents  # 返回加载的文档

def load_documents_from_pdf(file_path):
    """
    从PDF文件加载文档。

    参数:
        file_path (str): PDF文档的路径。

    返回:
        list of dict: 包含文档ID和文本的字典列表。
    """
    documents = []  # 初始化文档列表
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)  # 创建PDF阅读器
        text = ""
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text  # 提取每页的文本
        documents.append({"id": os.path.basename(file_path), "text": text})  # 添加文档到列表
    return documents  # 返回加载的文档

def load_documents_from_txt(file_path):
    """
    从TXT文件加载文档。

    参数:
        file_path (str): TXT文件的路径。

    返回:
        list of dict: 包含文档ID和文本的字典列表。
    """
    documents = []  # 初始化文档列表
    with open(file_path, "r", encoding="utf-8") as txt_file:
        text = txt_file.read()  # 读取整个文本文件
        documents.append({"id": os.path.basename(file_path), "text": text})  # 添加文档到列表
    return documents  # 返回加载的文档

def chunk_text(text, chunk_size=2000):
    """
    将文本切分为多个块。

    参数:
        text (str): 要切分的文本。
        chunk_size (int): 每个块的最大字符数。

    返回:
        list of str: 切分后的文本块列表。
    """
    chunks = []  # 初始化文本块列表
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])  # 切分文本并添加到列表
    return chunks  # 返回文本块列表

def load_documents(directory, chunk_sizes=[500]):
    """
    载入指定目录的文档，并根据多个 chunk_size 切分为多个块。

    参数:
        directory (str): 包含文档的目录路径。
        chunk_sizes (list of int): 每个块的最大字符数列表。

    返回:
        list of dict: 切分后的文档块列表，每个块包含ID、chunk_size和文本。
    """
    documents = []  # 初始化文档列表
    meta_files = {"processed_files.txt", "processed_hashes.txt"}  # 定义要排除的元文件

    for filename in os.listdir(directory):
        if filename in meta_files:
            continue  # 跳过元文件

        file_path = os.path.join(directory, filename)  # 获取文件完整路径
        if filename.endswith(".csv"):
            docs = load_documents_from_csv(file_path)  # 加载CSV文件
        elif filename.endswith(".pdf"):
            docs = load_documents_from_pdf(file_path)  # 加载PDF文件
        elif filename.endswith(".txt"):
            docs = load_documents_from_txt(file_path)  # 加载TXT文件
        else:
            continue  # 跳过不支持的文件类型
        
        for doc in docs:
            for chunk_size in chunk_sizes:
                chunks = chunk_text(doc["text"], chunk_size)  # 根据不同 chunk_size 切分文档文本
                for idx, chunk in enumerate(chunks):
                    documents.append({
                        "id": f"{doc['id']}_size_{chunk_size}_chunk_{idx}",
                        "chunk_size": chunk_size,
                        "text": chunk
                    })  # 添加每个文本块及其 chunk_size 到文档列表
    return documents  # 返回所有加载和切分的文档
