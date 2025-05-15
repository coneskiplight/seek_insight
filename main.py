"""
HSBC Commercial Banking - Batch PDF Processor
Version: 1.2.0
Author: HSBC AI Engineering Team
"""
import os
import glob
import json
import logging
from typing import TypedDict, Annotated, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from PyPDF2 import PdfReader
# ----------------- Configuration -----------------
MAX_WORKERS = 8  # Recommended: 2x CPU cores
PROCESS_TIMEOUT = 300  # 5 minutes per PDF
LOG_LEVEL = logging.INFO
# Initialize logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=LOG_LEVEL
)
logger = logging.getLogger("HSBC_PDF_Processor")
# ----------------- Sub Workflow Definition -----------------
class SubWorkflowState(TypedDict):
    file_path: str
    raw_text: Optional[str]
    processed_chunks: Annotated[list[str], lambda x, y: x + y]
    extracted_data: dict
def create_sub_workflow() -> Any:
    """创建处理单个PDF的子工作流"""
    workflow = StateGraph(SubWorkflowState)
    llm = ChatOpenAI(
        model="gpt-4-turbo",
        temperature=0,
        openai_api_key=os.getenv("HSBC_OPENAI_API_KEY"),
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    # ----------------- Node Definitions -----------------
    def parse_pdf(state: SubWorkflowState):
        """PDF文本提取"""
        logger.info(f"Processing {state['file_path']}")
        try:
            with open(state["file_path"], "rb") as f:
                reader = PdfReader(f)
                return {
                    "raw_text": "".join(
                        page.extract_text() for page in reader.pages
                    )
                }
        except Exception as e:
            logger.error(f"Parse failed: {state['file_path']} - {str(e)}")
            return {"raw_text": None}
    def chunk_text(state: SubWorkflowState):
        """文本分块（示例逻辑）"""
        if not state["raw_text"]:
            return {"processed_chunks": []}
        CHUNK_SIZE = 1000
        text = state["raw_text"]
        return {
            "processed_chunks": [
                text[i:i+CHUNK_SIZE]
                for i in range(0, len(text), CHUNK_SIZE)
            ]
        }
    def analyze_content(state: SubWorkflowState):
        """AI内容分析"""
        template = """You are a HSBC financial analyst. Analyze the document and extract:
        - company_name
        - fiscal_year
        - total_revenue
        - net_profit
        - key_risks (array)
        
        {content}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm
        
        try:
            analysis = chain.invoke({
                "content": state["raw_text"][:10000]  # First 10k chars
            })
            return {
                "extracted_data": json.loads(analysis.content)
            }
        except Exception as e:
            logger.error(f"Analysis failed: {state['file_path']} - {str(e)}")
            return {"extracted_data": {}}
    # ----------------- Workflow Construction -----------------
    workflow.add_node("parse_pdf", parse_pdf)
    workflow.add_node("chunk_text", chunk_text)
    workflow.add_node("analyze", analyze_content)
    
    workflow.set_entry_point("parse_pdf")
    workflow.add_edge("parse_pdf", "chunk_text")
    workflow.add_edge("chunk_text", "analyze")
    workflow.add_edge("analyze", END)
    
    return workflow.compile()
# ----------------- Main Workflow Definition -----------------
class MainWorkflowState(TypedDict):
    pending_files: Annotated[list[str], lambda x, y: x + y]
    completed_files: Annotated[list[dict], lambda x, y: x + y]
def process_pdfs(folder_path: str) -> list[dict]:
    """
    批量处理PDF主入口
    
    Args:
        folder_path: 包含PDF文件的文件夹路径
        
    Returns:
        [
            {
                "file_path": "path/to/file.pdf",
                "status": "success"|"error",
                "data": {},  # AI分析结果
                "error": "error_message"  # 如果失败
            },
            ...
        ]
    """
    # 输入验证
    if not os.path.isdir(folder_path):
        raise ValueError(f"Invalid directory: {folder_path}")
    
    # 初始化工作流
    main_workflow = StateGraph(MainWorkflowState)
    
    # ----------------- Main Nodes -----------------
    def initialize_state(state: MainWorkflowState):
        """初始化状态"""
        return {
            "pending_files": glob.glob(str(Path(folder_path)/"*.pdf")),
            "completed_files": []
        }
    
    def process_file(state: MainWorkflowState):
        """处理单个文件的包装函数"""
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_single_pdf, file): file
                for file in state["pending_files"]
            }
            
            results = []
            for future in as_completed(futures, timeout=PROCESS_TIMEOUT*2):
                file = futures[future]
                try:
                    result = future.result(timeout=PROCESS_TIMEOUT)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed processing {file}: {str(e)}")
                    results.append({
                        "file_path": file,
                        "status": "error",
                        "error": str(e)
                    })
        
        return {"completed_files": results}
    # ----------------- Workflow Construction -----------------
    main_workflow.add_node("init", initialize_state)
    main_workflow.add_node("process", process_file)
    
    main_workflow.set_entry_point("init")
    main_workflow.add_edge("init", "process")
    main_workflow.add_edge("process", END)
    
    # ----------------- Execution -----------------
    result = main_workflow.compile().invoke({
        "pending_files": [],
        "completed_files": []
    })
    
    return sorted(
        result["completed_files"], 
        key=lambda x: x.get("file_path", "")
    )
def process_single_pdf(file_path: str) -> dict:
    """处理单个文件并包装结果"""
    start_time = datetime.now()
    result = {
        "file_path": file_path,
        "status": "success",
        "processing_time": None,
        "data": None,
        "error": None
    }
    
    try:
        # 验证文件
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"PDF not found: {file_path}")
        
        if Path(file_path).stat().st_size > 100*1024*1024:  # 100MB限制
            raise ValueError("File size exceeds 100MB limit")
        
        # 执行子工作流
        sub_app = create_sub_workflow()
        output = sub_app.invoke({
            "file_path": file_path,
            "raw_text": None,
            "processed_chunks": [],
            "extracted_data": {}
        })
        
        result["data"] = output.get("extracted_data", {})
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        result.update({
            "status": "error",
            "error": str(e)
        })
    finally:
        result["processing_time"] = str(datetime.now() - start_time)
    
    return result
# ----------------- Main Execution -----------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python pdf_processor.py </path/to/pdf/folder>")
        sys.exit(1)
    
    try:
        print(f"Starting HSBC PDF Processor for: {sys.argv[1]}")
        start = datetime.now()
        
        results = process_pdfs(sys.argv[1])
        
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = len(results) - success_count
        
        print(f"\nProcessing completed in {datetime.now() - start}")
        print(f"Results: {success_count} succeeded, {error_count} errors")
        
        # 生成报告
        report_path = Path(sys.argv[1])/"processing_report.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Report saved to: {report_path}")
        
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        sys.exit(2)