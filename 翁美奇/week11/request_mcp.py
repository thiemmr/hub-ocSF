"""
    麦当劳mcp调用
"""
from openai import OpenAI
import os
from dotenv import load_dotenv

import requests
import json
from pathlib import Path

# 假设服务器端点是 /mcp，可根据实际情况调整
BASE_URL = "https://mcp.mcd.cn"
OUT_DIR = Path(__file__).parent / 'out'

def send_request(method, params=None, session_id=None):
    """发送 JSON-RPC 请求到 MCP 服务器"""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,  # 简单的递增 ID，实际可维护一个计数器
        "method": method,
        "params": params or {}
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id

    # 如果有 Authorization 需求，可以加上
    headers["Authorization"] = "Bearer "

    resp = requests.post(BASE_URL, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()

def build_client():
    """构建并返回 OpenAI 客户端，使用环境变量 OPENAI_API_KEY 作为凭证。"""
    load_dotenv()  # 放在 os.getenv 之前，它会自动去找 .env 文件
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY 环境变量未设置")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def run(tools):
    formated_tools = [{
        "type": "function", "function": tool } for tool in tools]
    
    client = build_client()
    messages = [{"role": "user", "content": "麦当劳目前有哪些券可以领?查看我的优惠券"}]
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=formated_tools,
    )
    msg = response.choices[0].message
    
    messages.append(msg)
    if msg.tool_calls:
        for i, tool_call in enumerate(msg.tool_calls):
            if i > 5:
                return  # 限制最多调用 5 次工具，避免无限循环
            tool_id = tool_call.id
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            print(f"调用工具: {tool_name}, 参数: {tool_args}")
            # 调用工具
            tool_response = send_request("tools/call", {
                "name": tool_name,
                "arguments": tool_args
            })
            result = "\n".join(b.text for b in tool_response['result']['content'] if hasattr(b, "text"))
            messages.append({"role": "tool","tool_call_id": tool_id, "content": result})
            
        # 将工具响应作为新的消息发送给模型
        msg = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=formated_tools,
        ).choices[0].message   
        
    answer = msg.content
    print("模型回答:", answer)
    
def main():
    # 1. 初始化
    init_result = send_request("initialize", {
        "protocolVersion": "0.1.0",
        "clientInfo": {"name": "my-client", "version": "1.0"}
    })
    print("初始化成功")

    # 提取 session id（可能返回在 result 中或响应头）
    session_id = init_result.get("result", {}).get("sessionId")
    # 如果服务器将 session id 放在响应头，可从 resp.headers 获取
    # 但这里假设它放在 result 中

    # 2. 列出工具
    tools_result = send_request("tools/list", session_id=session_id)
    with open(OUT_DIR/ 'request_tools.json', "w", encoding="utf-8") as f:
        json.dump(tools_result, f, indent=2, ensure_ascii=False)
        
    print("可用工具列表：request_tools")

    
    tools = tools_result.get("result", {}).get("tools", [])
    # 3. 调用一个实际工具
    run(tools)

if __name__ == "__main__":
    main()
