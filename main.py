import chainlit as cl
import os
import sqlite3
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool


# 生成表信息
def generate_table_info():
    db_path = os.path.join(os.path.dirname(__file__), './data/mydatabase.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_info = ""
    for table in tables:
        table_name = table[0]
        table_info += f"Table: {table_name}\n"
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        for column in columns:
            column_name = column[1]
            column_type = column[2]
            table_info += f"  Column: {column_name}, Type: {column_type}\n"
    conn.close()
    return table_info


# 定义执行数据库查询的函数
def query_db(sql_query):
    try:
        db_path = os.path.join(os.path.dirname(__file__), './data/mydatabase.db')
        conn = sqlite3.connect(db_path)
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
        description="Fetch data from sqlite database",
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
    # 生成表信息
    table_info = generate_table_info()
    system_message = f"""You are an expert in data analysis. You will provided valuable insights for business user based on their request.
    Before responding, You will make sure that user ask pertains to data analysis on provided schema, else decline.
    If user request some data, you will build sql query based on the user request for sqlite db from the provided schema/table details and call query_db tools to fetch data from database with the correct/relevant query that gives correct result.
    You have access to tool to execute database query and get results and to plot the query results.
    One you have provided the data, you will do reflection to see if you have provided correct data or not. because you don't know the data beforehand but only the schema so you might discover some new insights while reflecting.

    Follow this Guidelines
    - It is very important that if you need certain inputs to proceed or are not sure about anything, you may ask question, but try to use your intelligence to understand user intention and also let user know if you make assumptions.
    - In the response message do not provide technical details like sql, table or column details, the response will be read by business user not technical person.
    - provide rich markdown response - if it is table data show it in markdown table format
    - In case you get a database error, you will reflect and try to call the correct sql query
    - Limit top N queries to 5 and let the user know that you have limited results
    - Limit number of columns to 5 - 8. Wisely Choose top columns to query in SQL queries based on the user request
    - when user asks for all records - limit results to 10 and tell them they you are limiting records
    - in SQL queries to fetch data, you must cast date and numeric columns into readable form(easy to read in string format)
    - Design robust sql queries that takes care of uppercase, lowercase or some variations because you don't know the complete data or list of enumerable values in columns.
    - Pay careful attention to the schema and table details I have provided below. Only use columns and tables mentioned in the schema details

    Here are complete schema details with column details:
    {table_info}"""
    agent.agent.llm_chain.prompt.messages[0].content = system_message


@cl.on_message
async def main(message: str):
    # 调用代理处理用户消息
    response = await cl.make_async(agent.run)(message)
    # 将代理的响应发送给用户
    await cl.Message(content=response).send()