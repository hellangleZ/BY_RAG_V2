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

print(f"Using Azure OpenAI Endpoint in answer_generator.py: {AZURE_OPENAI_ENDPOINT}")  # 添加调试信息

def generate_final_answer(context, user_question):
    """
    根据检索结果生成答案。

    参数:
        context (str): 相关文档的拼接文本。
        user_question (str): 用户提出的问题。

    返回:
        str: 生成的答案。
    """
    try:
        start_time = time.time()  # 记录开始时间
        if not context or not user_question:
            raise ValueError("Context and user question must be provided")
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,  # 使用部署名称作为模型
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"根据以下内容回答用户的问题。\n\n用户问题：{user_question}\n\n相关文档：\n{context}\n\n答案："}
            ],
            max_tokens=300,
            temperature=0.7,
            n=1,
        )
        end_time = time.time()  # 记录结束时间
        print(f"API 调用耗时: {end_time - start_time} 秒")  # 打印API调用时间

        choices = response.choices
        if choices:
            answer = choices[0].message.content.strip()
            return answer  # 返回生成的答案
        else:
            raise ValueError("API 响应中没有找到 'choices' 字段")
    except Exception as e:
        print(f"生成答案失败: {e}")  # 添加调试信息
        raise
