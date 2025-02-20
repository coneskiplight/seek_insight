import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
import psycopg2

# 定义执行数据库查询的函数
def query_db(sql_query):
    try:
        # 建立数据库连接，需要根据实际情况修改数据库连接信息
        conn = psycopg2.connect(
            database="your_database",
            user="your_user",
            password="your_password",
            host="your_host",
            port="your_port"
        )
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()
        return f"查询结果：{columns}, {results}"
    except Exception as e:
        return f"查询数据库时出错: {str(e)}"

# 定义工具
tools = [
    Tool(
        name="query_db",
        func=query_db,
        description="Fetch data from postgres database",
        parameters={
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": "complete and correct sql query to fulfil user request."
                }
            },
            "required": ["sql_query"]
        }
    )
]

# 初始化聊天模型
chat = ChatOpenAI(model_name="gpt-3.5-turbo")

# 初始化代理
agent = initialize_agent(
    tools,
    chat,
    agent=AgentType.OPENAI_FUNCTION_CALLING,
    verbose=True
)

@cl.on_chat_start
async def start():
    # 当聊天开始时，可以在这里进行一些初始化操作
    pass

@cl.on_message
async def main(message: str):
    # 调用代理处理用户消息
    response = await cl.make_async(agent.run)(message)
    # 将代理的响应发送给用户
    await cl.Message(content=response).send()