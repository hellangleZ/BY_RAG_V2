import os  # 导入os模块用于获取环境变量
from dotenv import load_dotenv  # 导入dotenv模块
from openai import AzureOpenAI  # 导入官方AzureOpenAI客户端
import time  # 导入time模块用于计时

load_dotenv()  # 加载环境变量

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # 从环境变量获取Azure OpenAI端点
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")  # 从环境变量获取Azure OpenAI API密钥
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")  # 从环境变量获取部署名称
API_VERSION = "2024-10-21"  # 使用有效的API版本

# 初始化AzureOpenAI客户端
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=API_VERSION
)

print(f"Using Azure OpenAI Endpoint in question_generator.py: {AZURE_OPENAI_ENDPOINT}")  # 添加调试信息

def generate_questions(document_chunks):
    """
    调用 Azure OpenAI API 生成问题。

    参数:
        document_chunks (list of dict): 包含文档文本的块，每个块为一个字典。

    返回:
        list of str: 生成的问题列表。
    """
    questions = []  # 初始化问题列表
    for chunk in document_chunks:
        try:
            start_time = time.time()  # 记录开始时间
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,  # 使用部署名称作为模型
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"根据以下内容生成3个相关问题：\n{chunk['text']}"}
                ],
                max_tokens=100,
                temperature=0.7,
                n=1,
            )
            end_time = time.time()  # 记录结束时间
            print(f"API 调用耗时: {end_time - start_time} 秒")  # 打印API调用时间

            choices = response.choices
            if choices:
                # 假设生成的问题以换行符分隔
                chunk_questions = choices[0].message.content.strip().split("\n")
                # 清理和过滤生成的问题
                cleaned_questions = [q.strip() for q in chunk_questions if q.strip()]
                questions.extend(cleaned_questions)
                print(f"生成了 {len(cleaned_questions)} 个问题。")  # 添加调试信息
            else:
                raise ValueError("API 响应中没有找到 'choices' 字段")
        except Exception as e:
            print(f"生成问题失败: {e}")  # 添加调试信息
            raise
    return questions  # 返回生成的问题列表
