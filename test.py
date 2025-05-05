from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage

@tool
def add(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers together."""
    return a * b

@tool
def divide(a: int, b: int) -> float:
    """Divides two numbers."""
    return a / b

tools = [add, multiply, divide]
tool_map = {t.name: t for t in tools}
def generic_bind_tools(llm, tools):
    """Generic tool binding solution for LangGraph"""
    def wrapped_llm(messages):
        # Generate tool descriptions
        tool_descs = "\n".join([
            f"{t.name}: {t.description}\nArgs: {json.dumps(t.args)}" 
            for t in tools
        ])
        
        # Get LLM response
        response = llm.invoke([
            SystemMessage(content=f"""You have access to these tools:
{tool_descs}

Respond with either:
1. Direct answer (normal message), OR
2. Tool call in JSON format:
{{"tool": "name", "args": {{...}}}}""")
        ] + messages)
        
        # Parse tool call if present
        try:
            data = json.loads(response.content)
            if "tool" in data:
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": data["tool"],
                        "args": data["args"],
                        "id": f"call_{hash(str(data))}"
                    }]
                )
        except json.JSONDecodeError:
            pass
            
        return response
        
    return wrapped_llm

llm_with_tools_3 = generic_bind_tools(llm, tools)
sys_msg = SystemMessage(content="You are a helpful assistant.")
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools_3(state["messages"])]}
