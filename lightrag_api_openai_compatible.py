import asyncio
import os
import secrets
import shutil
import time
import json

from datetime import datetime
from typing import List, Optional, Union

import httpx
import nest_asyncio
import numpy as np
import pandas as pd
# import textract
# from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from LightRAG.lightrag import LightRAG, QueryParam
# from lightrag.llm import openai_compatible_complete_if_cache, \
#     openai_compatible_embedding
# from lightrag.utils import EmbeddingFunc

# from Gradio.graph_visual_with_html import KnowledgeGraphVisualizer as kgHTML

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

DEFAULT_RAG_DIR = "index_default"
app = FastAPI(title="LightRAG API", description="API for RAG operations")

# load_dotenv()

API_port = os.getenv("API_port",8020)
print(f"API_port: {API_port}")

# Configure working directory
WORKING_DIR = os.getenv("RAG_DIR",None)
print(f"WORKING_DIR: {WORKING_DIR}")
LLM_MODEL = os.getenv("LLM_MODEL",None)
print(f"LLM_MODEL: {LLM_MODEL}")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL",None)
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
EMBEDDING_MAX_TOKEN_SIZE = int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE",1024))
print(f"EMBEDDING_MAX_TOKEN_SIZE: {EMBEDDING_MAX_TOKEN_SIZE}")

BASE_URL=os.getenv("OPENAI_BASE_URL",None)
API_KEY = os.getenv("OPENAI_API_KEY",None)

available_models = []

# if not os.path.exists(WORKING_DIR):
#     os.mkdir(WORKING_DIR)


# LLM model function
"""
在本项目中，llm_model_func与embedding function默认使用openai_compatible的相关代码，如有需要可在/lightrag/llm.py寻找到你想要使用对应的代码，以配合你使用的llm
如果真的需要修改的话，请在/lightrag/lightrag.py中的llm_model_func和embedding function进行相对应的修改
"""

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    load_dotenv(override=True)
    return await openai_compatible_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        **kwargs,
    )


# Embedding function


# async def embedding_func(texts: list[str]) -> np.ndarray:
#     load_dotenv(override=True)
#     return await openai_compatible_embedding(
#         texts,
#         EMBEDDING_MODEL,
#         api_key=os.getenv("OPENAI_API_KEY"),
#         base_url=os.getenv("OPENAI_BASE_URL"),
#     )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    #print(f"{embedding_dim=}")
    return embedding_dim

from loguru import logger

async def set_openai_rag(rag_type=''):
    
    '''
    rag_type =   openai or local
    '''
    from src.config import config_file
    from LightRAG.lightrag.utils import EmbeddingFunc
    from LightRAG.lightrag.kg.postgres_impl import PostgreSQLDB
    from LightRAG.lightrag.llm import openai_complete, openai_embedding
    
    config = {}
    if os.path.exists(config_file):
        with open(config_file, "r", encoding='utf-8') as f:
            config = json.load(f)

    openai_api_key = config.get("openai_api_key", "") or os.environ.get("openai_api_key",None)
    openai_api_base = config.get("openai_api_base", "") or os.environ.get("openai_api_base",None)

    if openai_api_base and openai_api_base:
        pass
    else:
        
        logger.info(f"get_rag_instance err")
        return


    class RoundRobinEmbeddingFunc:
        def __init__(self, embedding_func, urls: list):
            self.embedding_func = embedding_func
            self.urls = urls
            import itertools
            self.iterator = itertools.cycle(urls)  # 无限轮询

        async def __call__(self, texts: list[str]) -> np.ndarray:
            # 获取当前的服务 URL
            url = next(self.iterator)
            # 使用原来的 embedding_func，传递选中的 URL
            return await self.embedding_func(texts, base_url=url)

    
    async def embedding_func(texts: list[str], base_url: str) -> np.ndarray:
        
        return await openai_embedding(
            texts,
            # model="text-embedding-bge-m3",
            model="text-embedding-bge-m3@f16",
            api_key="empty",
            base_url=base_url  # 使用传入的 base_url
        )


    def configure_neo4j():
        # Configuration with validation
        try:
            uri = "bolt://localhost:7687"  # Default Neo4j port
            # or use bolt protocol
        
            os.environ["NEO4J_URI"] = uri
            os.environ["NEO4J_USERNAME"] = "neo4j"
            os.environ["NEO4J_PASSWORD"] = "123456123"

            # print(f"Neo4j URI set to: {os.environ.get('NEO4J_URI')}")
            # print(f"Neo4j configuration complete")
            
        except Exception as e:
            print(f"Neo4j configuration error: {e}")
            raise



    
    
    async def configure_lightrag(working_dir,round_robin_embedding_func, embedding_dimension,history_messages=[]):
        from LightRAG.lightrag.llm import gpt_4o_mini_complete
        # api_key = "empty"
        return LightRAG(
            working_dir=working_dir,
            addon_params={"insert_batch_size": 100, "max_concurrent_tasks": 5},
            llm_model_func=gpt_4o_mini_complete,
            # llm_model_name="qwen2.5-7b-instruct",
            llm_model_max_async=30,
            # llm_model_max_token_size=32768,
            llm_model_kwargs={"base_url": openai_api_base, "api_key": openai_api_key},
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dimension,
                max_token_size=2048,
                func=round_robin_embedding_func if round_robin_embedding_func else embedding_func
                # func=embedding_func
            ),
            kv_storage="PGKVStorage",
            doc_status_storage="PGDocStatusStorage",
            graph_storage="Neo4JStorage",
            # graph_storage="PGGraphStorage",
            vector_storage="PGVectorStorage",
        )
    # LightRAG configuration
    async def configure_local_lightrag(working_dir,round_robin_embedding_func, embedding_dimension,history_messages=[]):
        api_key = "empty"
        return LightRAG(
            working_dir=working_dir,
            addon_params={"insert_batch_size": 20,"history_messages":history_messages},
            llm_model_func=openai_complete,
            llm_model_name="qwen2.5-7b-instruct",
            llm_model_max_async=4,
            llm_model_max_token_size=32768,
            llm_model_kwargs={"base_url": "http://127.0.0.1:12345/v1", "api_key": api_key},
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dimension,
                max_token_size=2048,
                func=round_robin_embedding_func if round_robin_embedding_func else embedding_func
                # func=embedding_func
            ),
            kv_storage="PGKVStorage",
            doc_status_storage="PGDocStatusStorage",
            graph_storage="Neo4JStorage",
            # graph_storage="PGGraphStorage",
            vector_storage="PGVectorStorage",
        )

    postgres_db = PostgreSQLDB(
        config={
            "host": "192.168.8.231",
            "port": 5432,
            "user": "postgres",
            "password": "098lkj.",
            "database": "rag", # rag2 code_rag
        }
    )
    configure_neo4j()
    await postgres_db.initdb()
    await postgres_db.check_tables()
    
    
    urls = [
        "http://127.0.0.1:12345/v1",
        "http://127.0.0.1:12345/v1"

    ]

    # 使用 RoundRobinEmbeddingFunc 包装原 embedding_func
    round_robin_embedding_func = RoundRobinEmbeddingFunc(embedding_func, urls)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    WORKING_DIR = os.path.join(ROOT_DIR, "init_dir1")
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    if rag_type == 'openai':

        rag = await configure_lightrag(WORKING_DIR, round_robin_embedding_func, 1024)
    elif rag_type == 'local':
        rag = await configure_local_lightrag(WORKING_DIR, round_robin_embedding_func, 1024)

    rag.doc_status.db = postgres_db
    rag.full_docs.db = postgres_db
    rag.text_chunks.db = postgres_db
    rag.llm_response_cache.db = postgres_db
    rag.key_string_value_json_storage_cls.db = postgres_db
    rag.chunks_vdb.db = postgres_db
    rag.relationships_vdb.db = postgres_db
    rag.entities_vdb.db = postgres_db
    rag.chunk_entity_relation_graph.embedding_func = rag.embedding_func

    return rag

# Initialize RAG instance
async def get_rag_instance(witch_rag_ins=None,history_messages: List = [],clea_history=False):
    '''
    实例化rag方法，想要获取该实例请使用rag = get_rag_instance()语句.
    '''
    # load_dotenv(override=True)  # 动态加载环境变量


   
    history_messages.append(history_messages)
    
    if clea_history:
        history_messages = []

    if witch_rag_ins == 'openai':
        return await set_openai_rag(rag_type='openai')
    elif witch_rag_ins == 'local': 
        return await set_openai_rag(rag_type='local')

    else:
        logger.info(f"test....get_rag_instance else")
        a = '1'
        return a 
    #     rag_dir = os.getenv("RAG_DIR")
    #     embeding_max_tokens = int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE"))
    #     print(f"RAG_DIR is set to: {rag_dir}")
    #     rag = LightRAG(
    #     working_dir=rag_dir,
    #     llm_model_func=llm_model_func(history_messages=[]),
    #     embedding_func=EmbeddingFunc(
    #         embedding_dim=asyncio.run(get_embedding_dim()),
    #         max_token_size=embeding_max_tokens,
    #         func=embedding_func,
    #     ),
    # )

    #     return rag


# Data models(LightRAG standard)


class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    only_need_context: bool = False


class InsertRequest(BaseModel):
    text: str


class Response(BaseModel):
    status: str
    data: Optional[str] = None
    message: Optional[str] = None


# Data models(OpenAI standard)

# 消息模型
class Message(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str

# 请求模型
class ChatRequest(BaseModel):
    model: str  # 模型名称
    messages: List[Message]  # 消息历史
    temperature: Optional[float] = 1.0  # 可选，生成的随机性
    top_p: Optional[float] = 1.0  # 可选，nucleus 采样
    n: Optional[int] = 1  # 可选，返回生成结果的数量
    stream: Optional[bool] = False  # 是否以流式传输返回
    stop: Optional[Union[str, List[str]]] = None  # 停止生成的标记
    max_tokens: Optional[int] = None  # 生成的最大 token 数量
    presence_penalty: Optional[float] = 0.0  # 可选，基于 token 出现的惩罚系数
    frequency_penalty: Optional[float] = 0.0  # 可选，基于 token 频率的惩罚系数
    user: Optional[str] = None  # 可选，用户标识

# 选项模型
class Choice(BaseModel):
    index: int  # 结果索引
    message: Message  # 每个结果的消息
    finish_reason: Optional[str]  # 生成结束的原因，例如 "stop"

# 使用统计模型
class Usage(BaseModel):
    prompt_tokens: int  # 提示词 token 数
    completion_tokens: int  # 生成的 token 数
    total_tokens: int  # 总 token 数

# 响应模型
class ChatCompletionResponse(BaseModel):
    id: str  # 响应唯一 ID
    object: str  # 响应类型，例如 "chat.completion"
    created: int  # 响应创建的时间戳
    model: str  # 使用的模型名称
    choices: List[Choice]  # 生成的结果列表
    usage: Optional[Usage]  # 可选，使用统计信息

# 单个文件响应模型（暂时保留）
class FileResponse(BaseModel):
    object: str = "file"
    id: str
    filename: str
    purpose: str
    status: str = "processed"

class ConnectResponse(BaseModel):
    connective: bool

# 多文件响应模型
class FilesResponse(BaseModel):
    object: str = "file"
    filename: str
    file_path: str
    purpose: str
    status: str = "processed"
    message: str = "Success"

# 多文件请求模型
class FilesRequest(BaseModel):
    files: dict[str, str]  # 文件名 -> 文件路径
    purpose: str = "knowledge_graph_frontend"

# 支持的文件类型
SUPPORTED_FILE_TYPES = ["txt", "pdf", "doc", "docx", "ppt", "pptx", "csv"]

# 文件保存路径
BASE_DIR = "./text"

# 历史消息的处理策略
def process_messages(
    user_message: str,
    system_prompt: Optional[str],
    history_messages: list[dict],
    prefill: Optional[str],
    strategy: str = "full_context",
) -> str:
    """
    处理消息的方法，用于生成最终需要传递给 RAG 的输入。

    Args:
        user_message (str): 当前用户消息。
        system_prompt (Optional[str]): 系统提示。
        history_messages (list[dict]): 多轮对话历史记录。
        prefill (Optional[str]): 预填充的消息。
        strategy (str): 消息处理策略，默认为 "full_context"。

    Returns:
        str: 处理后的完整输入消息。
    """
    if strategy == "current_only":
        # 仅处理当前用户输入，不添加上下文
        return user_message

    elif strategy == "recent_context":
        # 仅保留最近几轮上下文
        recent_messages = history_messages[-3:]  # 最近 3 条对话
        full_context = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in recent_messages
        )
        message = f"{full_context}\nUser: {user_message}"
        if prefill:
            message += f"\nAssistant: {prefill}"
        return message

    elif strategy == "full_context":
        # 完整上下文处理
        full_context = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in history_messages
        )
        message = f"System: {system_prompt}\n{full_context}\nUser: {user_message}" if system_prompt else f"{full_context}\nUser: {user_message}"
        if prefill:
            message += f"\nAssistant: {prefill}"
        return message

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def append_random_hex_to_list(user_messages: list, hex_length: int = 8) -> list:
    """
    在用户消息列表的每一项末尾添加一个随机的十六进制数。

    Args:
        user_messages (list): 原始用户消息的列表。
        hex_length (int): 生成的十六进制数长度，默认为8。

    Returns:
        list: 添加了随机十六进制数后的用户消息列表。
    """
    modified_messages = []
    for message in user_messages:
        if isinstance(message, str):  # 确保每个项是字符串
            random_hex = secrets.token_hex(nbytes=hex_length // 2)
            modified_messages.append(f"{message}\n\n\n---The following strings are for markup only and are not relevant to the above, so please ignore them---\n\n\n[#{random_hex}]")
        else:
            modified_messages.append(message)  # 非字符串项原样返回
    return modified_messages


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions_endpoint(request: ChatRequest):
    """
    QueryParam:
        mode:处理模式，共四种，请查看官方文档以便选择
        only_need_context：在本项目中，即使为true也不影响上下文策略
    """
    load_dotenv()   #动态加载环境变量
    rag = get_rag_instance()
    try:
        asyncio.run(get_model_info())

        # 确认使用何种模型
        if request.model not in available_models:
            raise HTTPException(status_code=400, detail="Selected model is not available.")
        else:
            llm_model = request.model

        global LLM_MODEL

        # Extract user query from messages
        user_message = []
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = [msg.content]
                break

        # 添加字符串以绕过hash碰撞
        appended_message = append_random_hex_to_list(user_message,8)

        user_messages = "\n".join(appended_message)
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found.")


        # Extract system prompt from messages before first user/assistant message
        system_prompts = []
        for msg in request.messages:
            if msg.role == "system" and not any(m.role in ["user", "assistant"] for m in request.messages[:request.messages.index(msg)]):
                system_prompts.append(msg.content)
            elif msg.role in ["user", "assistant"]:
                break
        system_prompt = "\n".join(system_prompts)

        # Get history messages starting after initial system messages
        history_messages = []
        start_processing = False
        for i, msg in enumerate(request.messages):  # Process all messages
            # Skip initial system messages until we find first user/assistant message
            if not start_processing and msg.role in ['user', 'assistant']:
                start_processing = True

            if start_processing:
                # Skip if it's the last message and it's a user message
                if i == len(request.messages) - 1 and msg.role == 'user':
                    continue
                # Skip if it's the last user message followed by an assistant message (which would be prefill)
                if msg.role == 'user' and i < len(request.messages) - 1 and request.messages[i + 1].role == 'assistant' and i + 1 == len(request.messages) - 1:
                    continue

                history_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        # Check if last message is from assistant (prefill)
        prefill = None
        if history_messages and history_messages[-1]["role"] == "assistant":
            prefill = history_messages[-1]["content"]
            # Remove the prefill from history
            history_messages = history_messages[:-1]

        processed_message = process_messages(
            user_message=user_messages,
            system_prompt=system_prompt,
            history_messages=history_messages,
            prefill=prefill,
            strategy="full_context",  # 默认使用完整上下文策略
        )

        # Store the original LLM_MODEL
        original_llm_model = LLM_MODEL

        async def update_llm_model():
            await asyncio.sleep(3)
            global LLM_MODEL
            LLM_MODEL = llm_model

        # Simulate RAG query result（不再使用）
        async def simulate_rag_query(query, system_prompt, history):
            # Simulated result, replace with actual rag.query call
            await rag.query(
                processed_message,
                param=QueryParam(mode="hybrid", only_need_context=False),
                #addon_params={"language": "Simplified Chinese"},
                #system_prompt_from_frontend=system_prompt,  # 添加 system_prompt
                #history_messages=history_messages,  # 添加 history_messages
            )
            return f"Simulated response to '{query}'"

        # Stream generator
        async def stream_generator():
            # 启动异步任务
            update_task = asyncio.create_task(update_llm_model())

            try:
                # 执行 RAG 查询
                result = rag.query(
                    processed_message,
                    param=QueryParam(mode="mix", only_need_context=False),
                    system_prompt_from_frontend=system_prompt,  # 添加 system_prompt
                )

                # 确保 update_task 被正确取消
                if not update_task.done():
                    update_task.cancel()
                    try:
                        await update_task
                    except asyncio.CancelledError:
                        pass

                # 恢复 LLM_MODEL
                LLM_MODEL = original_llm_model

                # 按换行符和空格生成块
                current_chunk = ""
                for char in result:
                    current_chunk += char
                    # 每次遇到换行符或积累到一定长度时发送
                    if char in ("\n", " ") or len(current_chunk) >= 50:
                        yield (
                            f'data: {json.dumps({"id": "chunk", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": current_chunk}}]})}\n\n'
                        )
                        await asyncio.sleep(0.1)  # 模拟流式传输的间隔
                        current_chunk = ""

                # 发送剩余内容
                if current_chunk:
                    yield (
                        f'data: {json.dumps({"id": "chunk", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": current_chunk}}]})}\n\n'
                    )

                # 发送结束信号
                yield 'data: {"id": "done", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}\n\n'

            except Exception as e:
                # 处理异常（记录日志或返回错误流）
                yield (
                    f'data: {json.dumps({"id": "error", "object": "chat.error", "message": str(e)})}\n\n'
                )
            finally:
                # 确保 update_task 被正确取消
                if not update_task.done():
                    update_task.cancel()
                    try:
                        await update_task
                    except asyncio.CancelledError:
                        pass
        # Stream or standard response
        if request.stream:
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            # Start the task to update LLM_MODEL after 1 second
            update_task = asyncio.create_task(update_llm_model())

            # Perform the rag.query operation
            result = rag.query(
                processed_message,
                param=QueryParam(mode="mix", only_need_context=False),
                system_prompt_from_frontend=system_prompt,  # 添加 system_prompt
                )

            # Ensure that LLM_MODEL is reverted back after rag.query completes
            if not update_task.done():
                update_task.cancel()
                try:
                    await update_task
                except asyncio.CancelledError:
                    pass
            LLM_MODEL = original_llm_model

            created_time = int(time.time())
            """
            print("\n\n\n")
            print(f"prompt_tokens: {len(system_prompt.split())}")
            print(f"user_message: {len(processed_message.split())}")
            """
            return ChatCompletionResponse(
                id="completion",
                object="chat.completion",
                created=created_time,
                model=llm_model,
                choices=[
                    Choice(
                        index=0,
                        message=Message(role="assistant", content=result),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=len(system_prompt.split()),
                    completion_tokens=len(result.split()),
                    total_tokens=len(system_prompt.split()) + len(result.split()),
                ),
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def get_model_info():
    load_dotenv()  # 动态加载环境变量
    global available_models
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    async with httpx.AsyncClient() as client:
        try:
            # 请求第三方 API
            response = await client.get(BASE_URL + f"/models", headers=headers)
            response.raise_for_status()  # 如果状态码非 2xx 会抛出异常

            # 获取响应数据并提取模型名称
            response_data = response.json()
            available_models = [model["id"] for model in response_data.get("data", [])]

            # 将 JSON 数据直接中转返回
            return JSONResponse(
                content=response.json(),  # 使用解析后的 JSON 数据
                status_code=response.status_code  # 保留原始响应的状态码
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail="API request failed")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# 单个文件的构造（暂时保留）
@app.post("/v1/file", response_model=FileResponse)
async def upload_file_to_build(file: UploadFile = File(...), purpose: str = "knowledge_graph_build"):
    load_dotenv()  # 动态加载环境变量
    rag = get_rag_instance()
    try:
        # 验证文件类型
        file_ext = file.filename.split(".")[-1].lower()
        if file_ext not in SUPPORTED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported types: {', '.join(SUPPORTED_FILE_TYPES)}",
            )

        # 创建以文件名和时间命名的子文件夹
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        folder_name = f"{file.filename.rsplit('.', 1)[0]}_{current_time}"
        save_dir = os.path.join(BASE_DIR, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        # 保存文件到子文件夹
        file_path = os.path.join(save_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 读取文件内容（按块读取以处理大型文件）
        if file_ext == "csv":
            # 处理 CSV 文件
            df = pd.read_csv(file_path)
            content = df.to_string()
        else:
            # 使用 textract 提取其他文件内容
            content = textract.process(file_path).decode("utf-8")

        # 插入内容到 RAG
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: rag.insert(content))

        # 返回响应
        return FileResponse(
            id=folder_name,  # 使用子文件夹名作为 ID
            filename=file.filename,
            purpose=purpose,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

# 多个文件的构造与插入（目前仅适用于管理前端）
@app.post("/v1/files", response_model=FilesResponse)
async def upload_files_to_build_frontend(request: FilesRequest):
    """
    接收文件字典，直接从路径读取文件内容。
    """
    load_dotenv(override=True)  # 动态加载环境变量
    rag = get_rag_instance()
    responses = []  # 存储每个文件的响应

    for file_name, file_path in request.files.items():
        try:
            # 验证文件是否存在
            if not os.path.exists(file_path):
                responses.append(FilesResponse(
                    filename=file_name,
                    file_path=file_path,
                    purpose=request.purpose,
                    status="failed",
                    message="File path does not exist"
                ))
                continue

            # 验证文件类型
            file_ext = file_name.split(".")[-1].lower()
            if file_ext not in SUPPORTED_FILE_TYPES:
                responses.append(FilesResponse(
                    filename=file_name,
                    file_path=file_path,
                    purpose=request.purpose,
                    status="failed",
                    message=f"Unsupported file type: {file_ext}. Supported types: {', '.join(SUPPORTED_FILE_TYPES)}"
                ))
                continue

            # 读取文件内容（按块读取以处理大型文件）
            if file_ext == "csv":
                # 处理 CSV 文件
                df = pd.read_csv(file_path)
                content = df.to_string()
            else:
                # 使用 textract 提取其他文件内容
                content = textract.process(file_path).decode("utf-8")

            # 插入内容
            async def async_insert(rag, content):
                await asyncio.to_thread(rag.insert, content)

            await async_insert(rag, content)

            # 成功响应
            responses.append(FilesResponse(
                filename=file_name,
                file_path=file_path,
                purpose=request.purpose,
                status="processed",
                message="File processed successfully"
            ))
        except Exception as e:
            # 异常处理
            responses.append(FilesResponse(
                filename=file_name,
                file_path=file_path,
                purpose=request.purpose,
                status="failed",
                message=f"Failed to process file: {str(e)}"
            ))
    # 完成后自动构建HTML以展示结果
    visualizer = kgHTML(os.getenv("RAG_DIR") + f"/graph_chunk_entity_relation.graphml")
    html_file_path = visualizer.generate_graph()
    # 返回所有文件的响应
    return JSONResponse(content=[response.dict() for response in responses])

@app.post("/v1/connect",response_model=ConnectResponse)
async def checkconnection():
    return ConnectResponse(
        connective = True
    )

#test API

# class QueryRequest(BaseModel):
#     query: str
#     mode: str = "mix"
#     query_stream: str = ''
#     use_method: str = 'local'
#     only_need_context: bool = False

# # 临时的流式生成器
# async def temporary_stream():
#     # 预定义的段落
#     paragraph = (
#         "这是一个用于测试的临时流式返回示例。"
#         "我们将这段文字分成多个部分，以模拟流式传输的数据。"
#         "每一部分之间会有短暂的延迟，模拟真实的流式数据传输。"
#         "希望这个示例能帮助你验证客户端和服务器之间的流式通信是否正常工作。"
#     )
#     # 将段落分成句子
#     sentences = paragraph.split("。")
#     for sentence in sentences:
#         if sentence:
#             yield sentence + "。\n"
#             await asyncio.sleep(2)  # 模拟延迟

# @app.post("/query")
# async def query_endpoint(request: QueryRequest):
#     # print("access")
#     try:
#         if not request.query_stream:
#             return {"status": "success", "data": "流式返回未启用。"}

#         async def event_generator():
#             async for chunk in temporary_stream():
#                 yield chunk
#         await asyncio.sleep(10)
#         return StreamingResponse(event_generator(), media_type="text/plain")

#     except Exception as e:
#         logger.error(f"Error processing query: {e}")
#         raise HTTPException(status_code=500, detail=str(e))



class QueryRequest(BaseModel):
    query: str
    mode: str = "mix"
    query_stream: str = ''
    use_method: str = 'local'
    only_need_context: bool = False
rag_cache = {}
cache_lock = asyncio.Lock()  # 异步锁，确保并发安全

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    处理 /query 端点的请求，根据 use_method 缓存并重用 RAG 实例。
    """
    query_mode = request.mode if request.mode else "mix"
    query_stream = True if request.query_stream else False

    if not request.use_method:
        logger.info("Cannot get request.use_method, returning")
        raise HTTPException(status_code=400, detail="use_method is required")

    use_method = request.use_method

    # 使用异步锁确保线程安全
    async with cache_lock:
        if use_method in rag_cache:
            rag = rag_cache[use_method]
            logger.info(f"使用缓存的 RAG 实例: {use_method}")
        else:
            logger.info(f"初始化新的 RAG 实例: {use_method}")
            try:
                rag = await set_openai_rag(rag_type=use_method) #get_rag_instance(witch_rag_ins=use_method)
                rag_cache[use_method] = rag
            except Exception as e:
                logger.error(f"Error initializing RAG for {use_method}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    try:
        # 假设 rag.aquery 是异步生成器
        resp = await rag.aquery(
            request.query,
            param=QueryParam(
                mode=query_mode,
                stream=query_stream
            ),
        )

        async def event_generator():
            async for chunk in resp:
                if chunk:
                    yield chunk

        return StreamingResponse(event_generator(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/insert", response_model=Response)
async def insert_endpoint(request: InsertRequest):
    try:
        rag = get_rag_instance()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: rag.insert(request.text))
        return Response(status="success", message="Text inserted successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/insert_file", response_model=Response)
async def insert_file(file: UploadFile = File(...)):
    try:
        rag = get_rag_instance()
        file_content = await file.read()
        # Read file content
        try:
            content = file_content.decode("utf-8")
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try other encodings
            content = file_content.decode("gbk")
        # Insert file content
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: rag.insert(content))

        return Response(
            status="success",
            message=f"File content from {file.filename} inserted successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# async def test_completion():
#     # 构造一个简单的测试 Prompt
#     prompt = "What are the key features of FastAPI?"
#     user_message = "Hi!"

#     # 构建 RAG 实例（确保参数正确）
#     rag = LightRAG(
#         working_dir=WORKING_DIR,
#         llm_model_func=llm_model_func,  # 使用您之前定义的 llm_model_func
#         embedding_func=EmbeddingFunc(
#             embedding_dim=768,
#             max_token_size=2048,
#             func=embedding_func,
#         ),
#     )

#     # 调用 `query` 方法
#     try:
#         loop = asyncio.get_event_loop()
#         print("Completion result:")
#         result = rag.query(
#             user_message,
#             param=QueryParam(
#                 mode="hybrid",  # 模式：可选 "hybrid", "retrieval", 或 "generation"
#                 only_need_context=False,  # 是否只返回上下文
#             ),
#         )
#         print(result)
#         #print(asyncio.iscoroutinefunction(rag.query))  # 输出是否为异步函数
#         #return result
#     except Exception as e:
#         print(f"Error during completion: {e}")

# # function test
# async def test_funcs():
#     result = await llm_model_func("How are you?")
#     print("llm_model_func: ", result)

if __name__ == "__main__":
    import uvicorn

    # rag = get_rag_instance()
    # 修改实例变量
    # print(f"Updated RAG_DIR to: {rag.working_dir}")
    #test_funcs()
    # asyncio.run(test_funcs())

    # uvicorn.run(app, host="0.0.0.0", port=int(API_port))
    uvicorn.run(app, host="127.0.0.1", port=int(API_port))

# Usage example
# To run the server, use the following command in your terminal:
# python lightrag_api_openai_compatible_demo.py

# Example requests:
# 1. Query:
# curl -X POST "http://127.0.0.1:8020/query" -H "Content-Type: application/json" -d '{"query": "your query here", "mode": "hybrid"}'

# 2. Insert text:
# curl -X POST "http://127.0.0.1:8020/insert" -H "Content-Type: application/json" -d '{"text": "your text here"}'

# 3. Insert file:
# curl -X POST "http://127.0.0.1:8020/insert_file" -H "Content-Type: application/json" -d '{"file_path": "path/to/your/file.txt"}'

# 4. Health check:
# curl -X GET "http://127.0.0.1:8020/health"
