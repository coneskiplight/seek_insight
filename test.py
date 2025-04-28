import os
import chainlit as cl
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# 配置 Azure OpenAI 信息
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "your_azure_openai_api_base"
os.environ["OPENAI_API_KEY"] = "your_azure_openai_api_key"
os.environ["OPENAI_API_VERSION"] = "your_azure_openai_api_version"

# 定义部署名称
embeddings_deployment_name = "your_embeddings_deployment_name"
llm_deployment_name = "your_llm_deployment_name"

# 向量数据库持久化目录
persist_directory = 'your_persist_directory'

# 初始化嵌入和向量数据库
embeddings = AzureOpenAIEmbeddings(
    deployment=embeddings_deployment_name,
    openai_api_version=os.getenv("OPENAI_API_VERSION")
)
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# 初始化大语言模型
llm = AzureChatOpenAI(
    deployment_name=llm_deployment_name,
    openai_api_version=os.getenv("OPENAI_API_VERSION")
)

# 定义自定义提示模板
prompt_template = """You are an expert in HSBC product marketing, proficient in generating various types of marketing content, including emails, SMS messages, and mobile push notifications.
Please generate corresponding marketing content based on the provided document content and the user's requirements. The generated content should be presented in {language}.
If there is no relevant information in the document, please create the content based on your professional knowledge and experience.

User requirement: {question}
Document content: {context}

{content_type} content:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["question", "context", "language", "content_type"]
)

# 创建包含自定义提示的问答链
chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)


@cl.on_chat_start
async def start():
    # 创建消息对象并发送
    msg = cl.Message(content="You can start submitting requirements for HSBC product marketing content generation!")
    # 添加语言选择按钮
    await msg.add_buttons([
        cl.Button(name="English", value="English", label="English"),
        cl.Button(name="Mandarin", value="Mandarin Chinese", label="Mandarin"),
        cl.Button(name="Cantonese", value="Cantonese", label="Cantonese")
    ])
    # 添加内容类型选择按钮
    await msg.add_buttons([
        cl.Button(name="Email", value="Email", label="Email"),
        cl.Button(name="SMS", value="SMS", label="SMS"),
        cl.Button(name="Mobile Push", value="Mobile push", label="Mobile Push")
    ])
    await msg.send()


@cl.on_button_click
async def on_button_click(button):
    if button.name in ["English", "Mandarin", "Cantonese"]:
        cl.user_session.set("language", button.value)
    elif button.name in ["Email", "SMS", "Mobile Push"]:
        cl.user_session.set("content_type", button.value)
    # 检查是否两个选项都已设置
    language = cl.user_session.get("language")
    content_type = cl.user_session.get("content_type")
    if language and content_type:
        await cl.Message(content=f"Selected language: {language}, Selected content type: {content_type}. Now you can enter your marketing requirement.").send()


@cl.on_message
async def main(message: str):
    language = cl.user_session.get("language")
    content_type = cl.user_session.get("content_type")
    if not language or not content_type:
        await cl.Message(content="Please select both language and content type first.").send()
        return
    # 根据用户问题进行相似度搜索
    similar_docs = vectorstore.similarity_search(message)

    # 把文档内容拼接成一个字符串作为上下文
    context = "\n".join([doc.page_content for doc in similar_docs])

    # 使用问答链生成回答
    result = chain.invoke({"input_documents": similar_docs, "question": message, "context": context, "language": language, "content_type": content_type})
    answer = result["output_text"]

    # 整理相似度搜索结果的来源信息
    sources = []
    for doc in similar_docs:
        if 'source' in doc.metadata:
            sources.append(doc.metadata['source'])

    # 如果有来源信息，将其添加到回答中
    if sources:
        source_info = "\n\nSources:\n" + "\n".join(sources)
        answer_with_sources = answer + source_info
    else:
        answer_with_sources = answer

    # 创建消息对象并发送包含来源信息的回答
    msg = cl.Message(content=answer_with_sources)
    await msg.send()