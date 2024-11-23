


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

- 文档上传：支持 .csv、.pdf、.txt 文件。
- 问题生成：基于文档内容自动生成问题。
- 向量化检索：支持基于 FAISS 和 BM25 的高效检索。
- 答案生成：利用 Azure OpenAI API 提供精准回答。

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

支持格式：.csv、.pdf、.txt。上传后，系统会根据指定的 chunk_size 自动分块处理。

### 构建向量库

可选择本地模型或 Azure OpenAI 模型进行向量化。支持多种配置，如向量化设备、模型名称等。

### 提问与检索

输入问题，系统会检索相关文档并生成最终答案。检索方式可选：仅使用向量化、结合 BM25。

## 开发指南

### 模块说明

#### `answer_generator.py`

功能：从检索到的文档生成最终答案。

主要函数：

```python
def generate_final_answer(context, user_question):
    """根据检索到的文档上下文生成最终答案。"""
```

#### `data_loader.py`

功能：支持从 CSV、PDF、TXT 文件加载文档并分块处理。

主要函数：

```python
def load_documents_from_csv(file_path)
def load_documents(directory, chunk_sizes)
```

#### `question_generator.py`

功能：基于文档内容生成问题。

主要函数：

```python
def generate_questions(document_chunks)
```

#### `retriever.py`

功能：检索与用户问题相关的文档和问题。

主要函数：

```python
def retrieve_documents(user_query, index_questions, index_documents, ...)
```

#### `vectorizer.py`

功能：支持向量化问题和文档块。

主要函数：

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

- 文档上传：支持 .csv、.pdf、.txt 文件。
- 问题生成：基于文档内容自动生成问题。
- 向量化检索：支持基于 FAISS 和 BM25 的高效检索。
- 答案生成：利用 Azure OpenAI API 提供精准回答。

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

支持格式：.csv、.pdf、.txt。上传后，系统会根据指定的 chunk_size 自动分块处理。

### 构建向量库

可选择本地模型或 Azure OpenAI 模型进行向量化。支持多种配置，如向量化设备、模型名称等。

### 提问与检索

输入问题，系统会检索相关文档并生成最终答案。检索方式可选：仅使用向量化、结合 BM25。

### 部署指南

#### 本地部署

按照上述安装和启动步骤即可。

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

纯文本文件，按指定块大小切分。

## 常见问题

### 如何解决 Azure OpenAI API 调用失败？

- 检查 `.env` 文件中的配置。
- 确认 API 密钥有效并且有足够的配额。

### 如何清理缓存？

在 Streamlit 应用侧边栏点击“清理缓存”按钮。

### 支持哪些模型？

- 本地：all-mpnet-base-v2 等任何 sentence-transformers 支持的模型。
- Azure OpenAI：如 text-embedding-ada-002。

## 构建向量库操作指南

本文档描述了文档问答系统中 “构建向量库” 界面的使用方法。该功能是整个系统的核心部分，主要负责上传文档、配置向量化模型以及构建检索所需的向量数据库。

### 功能概述

构建向量库界面主要包括以下功能：

- 文档上传：支持上传多种格式的文档（CSV、PDF、TXT）。
- 分块配置：将文档内容切分为小块，支持用户自定义 chunk_size 值。
- 向量化模型选择：支持本地模型和 Azure OpenAI 模型（推荐）。
- 向量数据库构建：将文档向量化后保存到向量索引中，用于后续的高效检索。

### 操作步骤

![image](https://github.com/user-attachments/assets/feb2a132-a4e9-4d20-81a8-dac4a4d0bdfe)
![image](https://github.com/user-attachments/assets/737b117c-b489-44ca-b42b-6be363ca534e)


#### 文档上传

- 点击 "Browse files" 按钮或直接将文件拖拽至界面中的上传区域。
- 支持的文件格式包括：CSV 文件（如：文档内容以结构化表格存储）、PDF 文件（支持多页 PDF 自动提取内容）、TXT 文件（支持任意纯文本内容）。
- 每个文件大小限制为 200MB。

#### 配置文档分块（chunk_size）

- 在 "输入 chunk_size 值" 输入框中，输入多个块大小值，以英文逗号分隔（如 500,1000）。
- 系统会根据不同的 chunk_size 将文档内容切分为多个小块，以便更高效地向量化和检索。

#### 选择向量化模型类型

- 选项 1：本地模型
  - 适用于没有使用云服务的情况。
  - 配置项：
    - 本地模型名称：如 all-mpnet-base-v2。
    - 设备类型：选择 CPU 或 CUDA（如果有 GPU 支持）。

- 选项 2：Azure OpenAI 模型（推荐）
  - 适用于已配置 Azure OpenAI 服务的用户。
  - 配置项：
    - API 密钥：Azure OpenAI 服务的访问密钥。
    - API 端点：Azure OpenAI 服务的 URL。
    - 嵌入模型名称：如 text-embedding-ada-002。

> 注意：切换模型类型后，系统会自动清除缓存，以确保配置生效。

#### 构建向量库

- 配置完成后，点击页面底部的 "构建向量库" 按钮。
- 如果勾选了 "构建 BM25 模型" 选项，系统还会同时构建基于 BM25 的关键词检索模型。

### 注意事项

- 上传文档后，系统会跳过重复文件，但可以选择是否重新处理这些文档。
- 在使用 Azure OpenAI 时，请确保 `.env` 文件中的配置项与界面中输入的内容一致。
- 文档切分大小（chunk_size）会显著影响性能：
  - 较小的块：适合精细检索，但可能增加处理时间。
  - 较大的块：适合简单问题的快速检索。

## 构建向量库结果界面说明

当用户完成向量库的构建后，系统会显示以下关键信息，帮助用户确认操作是否成功并了解当前数据状态。

### 功能概述

构建完成后，系统会提供：
- 文件上传信息：显示已处理的文档文件及其保存路径。
- 构建状态信息：确认向量库和检索模型是否成功构建。
- 统计信息：提供问题总数、FAISS 索引状态和 BM25 模型状态。

### 界面信息说明

![image](https://github.com/user-attachments/assets/540a830b-9a23-4f76-af74-1a9027bc4fcf)


#### 文件上传信息

在界面中显示上传的文件信息，例如：

- 文件名称：graphRAG.pdf
- 文件大小：6.5MB
- 文件保存路径：/aml/by-rag/documents/graphRAG.pdf

用户可通过确认文件保存路径，确保文档已正确存储到指定目录。

#### 构建状态信息

系统会实时反馈以下构建状态：

- 文档上传状态：显示新文档上传的结果，如 “新文档上传成功！”。
- 向量数据库构建状态：提示向量数据库的构建是否成功，例如 “向量数据库构建成功！”。
- BM25 模型构建状态（如果勾选了相关选项）：提示 BM25 检索模型是否构建完成，例如 “BM25模型已构建并保存。”。

#### 统计信息

- 总问题数：表示系统从文档中生成的问题总数，例如：23。
- FAISS 问题索引总数：表示当前 FAISS 数据库中问题索引的总量，例如：23。
- FAISS 文档索引总数：表示当前 FAISS 数据库中文档索引的总量，例如：6。

通过这些统计信息，用户可以快速了解向量库的构建规模和当前系统状态。

### 使用建议

#### 检查构建状态：

确保所有状态反馈均为“成功”，如向量库和 BM25 模型均提示已构建完成。

#### 确认统计信息：

如果统计信息显示的索引数量与文档内容或问题数量不匹配，可能需要检查文档切分参数（如 chunk_size）。

#### 后续操作：

构建完成后，用户可前往“查询”模块，输入问题以验证检索与回答功能。

## 查询界面操作指南

本文档介绍了文档问答系统中 “查询” 界面的使用方法。用户可以在该界面输入问题并获取基于文档库的精准答案。

### 功能概述

查询界面 提供以下主要功能：

- 用户输入自然语言问题。
- 系统通过向量检索和 BM25 检索（可选）从文档库中获取相关内容。
- 基于相关文档，利用 Azure OpenAI 生成最终答案。

### 操作步骤

#### 配置检索方式：

- 启用 BM25（可选项）：勾选 “在检索中使用 BM25” 复选框后，系统会结合 BM25 模型进行关键词检索。如果未勾选，系统将仅使用向量检索（FAISS）。

#### 输入问题

![image](https://github.com/user-attachments/assets/214436a1-c4bb-45fe-b3cf-950cdff8be5d)


- 在输入框中输入自然语言问题，例如：什么是 Graph-RAG？。

#### 查看统计信息

系统会显示以下统计信息：

- 加载的总问题数：表示当前系统中已生成的问题总量。
- 加载的文档块数：表示从文档中切分出的块总量。
- FAISS 问题索引总数：表示系统中构建的 FAISS 问题向量总数。
- FAISS 文档索引总数：表示系统中构建的 FAISS 文档向量总数。

#### 获取答案

- 点击 “获取答案” 按钮，系统将：
  - 使用向量检索匹配相关问题和文档。
  - 如果启用 BM25，结合关键词检索结果。
  - 利用 Azure OpenAI 生成最终答案，并展示在界面下方。

### 示例界面

以下为查询界面的截图：

- 输入问题：用户输入的问题示例为 什么是 Graph-RAG？。
- 统计信息：
  - 总问题数：23
  - 文档块总数：6
  - FAISS 问题索引：23
  - FAISS 文档索引：6
- 最终答案：系统生成了详细的答案解释。

### 注意事项

#### 确保向量库已构建：

在查询前，请确保已在 “构建向量库” 界面完成文档上传和向量库构建。界面统计信息应与构建的文档规模一致。

#### 启用 BM25 的条件：

如果勾选了 BM25，请确保在构建向量库时已启用 BM25 模型构建，否则可能无效。

#### 输入问题的清晰度：

输入清晰的自然语言问题，以提高检索和答案生成的准确性。

## 常见问题

### 无法生成答案

- 检查是否已正确构建向量库。
- 确保 `.env` 文件中 Azure OpenAI 配置正确。

### 检索结果不准确

- 调整文档分块大小（chunk_size），优化向量化结果。
- 检查输入问题的表述是否与文档内容一致。

完成查询后，您可以根据生成的答案进一步优化输入问题，或返回 "构建向量库" 模块调整文档处理参数。

> 如果出现 UI Bug，刷新可以解决！
```

