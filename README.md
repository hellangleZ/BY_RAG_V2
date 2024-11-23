

# 基于 Streamlit 的文档问答系统

本项目是一个基于 Streamlit 的文档问答系统。用户可以上传多种格式的文档，系统通过向量化和 BM 检索算法，从中提取相关答案。支持多种向量化模型（如本地和 Azure OpenAI），并结合 FAISS 和 BM25 提供高效的检索能力。

## 项目特点

本项目最大的不同是可以支持 document2question2question 的方式，来预先生成可能潜在的问题，因为对于查询问题的 L2 距离来讲，潜在的近似问题的 L2 距离肯定是要短于文本块 chunk 的 L2 距离，所以用这个方法来更加匹配查询的可能性。

默认支持：
- 3 个问题最近距离的文档
- 3 个文本最近距离的文档
- 3 个 BM25 最近距离的文档

可以在代码里根据需要进行修改。

## 功能特点

- **文档上传**：支持 .csv、.pdf、.txt 文件。
- **问题生成**：基于文档内容自动生成问题。
- **向量化检索**：支持基于 FAISS 和 BM25 的高效检索。
- **答案生成**：利用 Azure OpenAI API 提供精准回答。

## 安装指南

### 环境准备

安装 Python（>= 3.10）

### 克隆项目代码

```bash
git clone <repository_url>
cd <repository_name>
```

### 创建并激活虚拟环境

```bash
python -m venv venv
source venv/bin/activate # Linux/macOS
venv\Scripts\activate # Windows
```

或者使用 Conda

### 安装依赖

```bash
pip install -r requirements.txt
```

> **注意**：不推荐直接安装 `requirements.txt` 里的依赖，那是我的环境，建议看 `suggestion.txt` 里需要的包来自行安装

### 环境变量配置

创建 `.env` 文件并填写以下内容：

```plaintext
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
DEPLOYMENT_NAME=
```

## 使用指南

### 本地启动

运行应用：

```bash
streamlit run app.py
```

打开浏览器访问 [http://localhost:8501](http://localhost:8501)

### 上传文档

支持格式：.csv、.pdf、.txt。

上传后，系统会根据指定的 chunk_size 自动分块处理。

### 构建向量库

可选择本地模型或 Azure OpenAI 模型进行向量化。支持多种配置，如向量化设备、模型名称等。

### 提问与检索

输入问题，系统会检索相关文档并生成最终答案。检索方式可选：仅使用向量化、结合 BM25。

## 开发指南

### 模块说明

#### `answer_generator.py`

**功能**：从检索到的文档生成最终答案。

**主要函数**：

```python
def generate_final_answer(context, user_question):
    """根据检索到的文档上下文生成最终答案。"""
```

#### `data_loader.py`

**功能**：支持从 CSV、PDF、TXT 文件加载文档并分块处理。

**主要函数**：

```python
def load_documents_from_csv(file_path)
def load_documents(directory, chunk_sizes)
```

#### `question_generator.py`

**功能**：基于文档内容生成问题。

**主要函数**：

```python
def generate_questions(document_chunks)
```

#### `retriever.py`

**功能**：检索与用户问题相关的文档和问题。

**主要函数**：

```python
def retrieve_documents(user_query, index_questions, index_documents, ...)
```

#### `vectorizer.py`

**功能**：支持向量化问题和文档块。

**主要函数**：

```python
def vectorize_questions(questions, model_type, **kwargs)
def build_vector_db(question_vectors, document_vectors)
```

## 项目文档

### 概述

本项目是一个基于 Streamlit 的文档问答系统。用户可以上传多种格式的文档，系统通过向量化和检索算法，从中提取相关答案。支持多种向量化模型（如本地和 Azure OpenAI），并结合 FAISS 和 BM25 提供高效的检索能力。

### 目录结构

```bash
├── app.py                # 主应用入口
├── answer_generator.py   # 答案生成模块
├── data_loader.py        # 数据加载和预处理
├── question_generator.py # 问题生成模块
├── retriever.py          # 文档检索模块
├── vectorizer.py         # 向量化模块
├── requirements.txt      # 项目依赖
├── .env                  # 环境变量配置文件
```

## 功能特点

- **文档上传**：支持 .csv、.pdf、.txt 文件。
- **问题生成**：基于文档内容自动生成问题。
- **向量化检索**：支持基于 FAISS 和 BM25 的高效检索。
- **答案生成**：利用 Azure OpenAI API 提供精准回答。

### 安装指南

#### 环境准备

安装 Python（>= 3.8）

#### 克隆项目代码

```bash
git clone <repository_url>
cd <repository_name>
```

#### 创建并激活虚拟环境

```bash
python -m venv venv
source venv/bin/activate # Linux/macOS
venv\Scripts\activate # Windows
```

#### 安装依赖

```bash
pip install -r requirements.txt
```

#### 环境变量配置

创建 `.env` 文件并填写以下内容：

```plaintext
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
DEPLOYMENT_NAME=
```

## 使用指南

### 本地启动

运行应用：

```bash
streamlit run app.py
```

打开浏览器访问 [http://localhost:8501](http://localhost:8501)

### 上传文档

支持格式：.csv、.pdf、.txt。

上传后，系统会根据指定的 chunk_size 自动分块处理。

### 构建向量库

可选择本地模型或 Azure OpenAI 模型进行向量化。支持多种配置，如向量化设备、模型名称等。

### 提问与检索

输入问题，系统会检索相关文档并生成最终答案。检索方式可选：仅使用向量化、结合 BM25。

## 部署指南

### 本地部署

按照上述安装和启动步骤即可。

### 云端部署（示例：Heroku）

#### 创建 Heroku 应用

```bash
heroku create <app_name>
```

#### 部署代码

```bash
git push heroku main
```

#### 设置环境变量

```bash
heroku config:set AZURE_OPENAI_ENDPOINT=
heroku config:set AZURE_OPENAI_API_KEY=
```

访问 Heroku 提供的应用 URL。

## 开发指南

### 代码模块说明

#### 核心模块

##### `app.py`

系统主入口，基于 Streamlit 构建交互界面。核心功能包括：
- 上传与处理文档
- 构建向量库
- 提问与生成答案

##### `answer_generator.py`

核心功能：调用 Azure OpenAI 模型生成最终答案。

示例函数：

```python
def generate_final_answer(context, user_question):
    """ 根据检索到的文档上下文生成最终答案。 """
```

##### `data_loader.py`

核心功能：加载 CSV、PDF、TXT 文件并进行分块。

示例函数：

```python
def load_documents(directory, chunk_sizes=[500]):
    """ 从指定目录加载文档，并切分为多个小块。 """
```

##### `retriever.py`

核心功能：基于向量化和 BM25 的文档检索。

示例函数：

```python
def retrieve_documents(user_query, index_questions, index_documents, ...):
    """ 检索与用户问题相关的文档和问题。 """
```

##### `vectorizer.py`

核心功能：使用 Azure OpenAI 生成文档和问题的向量表示。

示例函数：

```python
def vectorize_questions(questions, model_type='azure_openai', **kwargs):
    """ 向量化问题列表，支持 Azure OpenAI 模型。 """
```

### 示例数据

#### CSV 文件：

```csv
id,text
1,这是第一段文本内容。
2,这是第二段文本内容。
```

#### PDF 文件

支持多页 PDF，自动提取每页内容。

#### TXT 文件

纯
