from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from PyPDF2 import PdfReader
import os

# 定义类型化的工作流状态
class WorkflowState(TypedDict):
    raw_text: str
    processed_chunks: Annotated[list[str], lambda x, y: x.extend(y)]
    extracted_data: dict

# 初始化模型
llm = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_kwargs={"response_format": {"type": "json_object"}}
)

# 设计结构化提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "您是HSBC的商业数据提取专家"),
    ("user", """请从以下文本中提取结构化数据：
      - company_name: 公司法定名称
      - fiscal_year: 财报年份
      - total_assets: 总资产（货币单位）
      - risk_factors: 列出Top3风险因素
      
      {text}
      
      严格按照JSON格式输出""")
])

# 创建数据处理链
analysis_chain = prompt | llm | JsonOutputParser()

def parse_pdf_node(state: WorkflowState):
    """PDF解析节点"""
    print("📄 Extracting text from PDF...")
    reader = PdfReader("financial_report.pdf")
    return {"raw_text": "".join(page.extract_text() for page in reader.pages)}

def chunking_node(state: WorkflowState):
    """文本分块处理节点"""
    chunk_size = 3000
    text = state["raw_text"]
    return {"processed_chunks": [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]}

def assistant(state: WorkflowState):
    """AI数据分析节点"""
    consolidated_data = {}
    
    for index, chunk in enumerate(state["processed_chunks"]):
        print(f"🔍 Analyzing chunk {index+1}/{len(state['processed_chunks'])}")
        
        try:
            chunk_result = analysis_chain.invoke({"text": chunk})
            
            # 数据合并策略
            for key in ["company_name", "fiscal_year"]:
                if chunk_result.get(key) and not consolidated_data.get(key):
                    consolidated_data[key] = chunk_result[key]
                    
            if isinstance(chunk_result.get("total_assets"), (int, float)):
                consolidated_data["total_assets"] = max(
                    consolidated_data.get("total_assets", 0),
                    chunk_result["total_assets"]
                )
                
            consolidated_data.setdefault("risk_factors", []).extend(
                chunk_result.get("risk_factors", [])[:3]
                if isinstance(chunk_result.get("risk_factors"), list)
                else []
            )
            
        except Exception as e:
            print(f"⚠️ Error processing chunk {index+1}: {str(e)}")
    
    # 去重风险因素
    if "risk_factors" in consolidated_data:
        consolidated_data["risk_factors"] = list({v: None for v in consolidated_data["risk_factors"]}.keys())[:3]
    
    return {"extracted_data": consolidated_data}

# 构建工作流
workflow = StateGraph(WorkflowState)
workflow.add_node("parse_pdf", parse_pdf_node)
workflow.add_node("chunk_data", chunking_node)
workflow.add_node("ai_analyst", assistant)

workflow.set_entry_point("parse_pdf")
workflow.add_edge("parse_pdf", "chunk_data")
workflow.add_edge("chunk_data", "ai_analyst")
workflow.add_edge("ai_analyst", END)

app = workflow.compile()

# 测试执行
if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "您的密钥"
    result = app.invoke({
        "raw_text": "",
        "processed_chunks": [],
        "extracted_data": {}
    })
    import json
    print(json.dumps(result["extracted_data"], indent=2, ensure_ascii=False))
