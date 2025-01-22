import json
import openai
from typing import List, Dict, Callable, Any, Union, Optional, Generator, Tuple

import logging
import os
from src.base_model import BaseLLMModel
# pip install openai==0.28

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 常量定义
TIMEOUT_STREAMING = 120  # 流式对话时的超时时间
TIMEOUT_ALL = 120        # 非流式对话时的超时时间
ENABLE_STREAMING_OPTION = True  # 是否启用流式响应
HIDE_MY_KEY = True               # 是否在UI中隐藏API密钥
CONCURRENT_COUNT = 100           # 允许同时使用的用户数量
SIM_K = 5
INDEX_QUERY_TEMPRATURE = 1.0
TITLE = "OpenAI 🚀"  # 根据需要替换
STANDARD_ERROR_MSG = "☹️发生了错误："  # 根据需要替换
GENERAL_ERROR_MSG = "获取对话时发生错误，请重试"  # 根据需要替换
INITIAL_SYSTEM_PROMPT = "You are a helpful assistant."

def safe_unicode_decode(bytes_obj: bytes) -> str:
    """
    安全地解码 Unicode 字符串，处理可能的编码错误。
    """
    try:
        return bytes_obj.decode("utf-8")
    except UnicodeDecodeError:
        return bytes_obj.decode("utf-8", errors="replace")

def openai_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, str]]] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    stream: bool = False,
    **kwargs,
) -> Union[str, Generator[str, None, None]]:
    """
    同步版本的 OpenAI 模型请求函数，支持缓存。

    Args:
        model (str): 模型名称。
        prompt (str): 用户输入的提示。
        system_prompt (Optional[str]): 系统提示。
        history_messages (Optional[List[Dict[str, str]]]): 对话历史。
        base_url (Optional[str]): OpenAI API 的基础 URL（用于代理等）。
        api_key (Optional[str]): OpenAI API 密钥。
        stream (bool): 是否启用流式响应。
        **kwargs: 其他可选参数。

    Returns:
        Union[str, Generator[str, None, None]]: 完整响应内容或生成器。
    """
    if api_key:
        openai.api_key = api_key
    if base_url:
        openai.api_base = base_url

    history_messages = history_messages or []
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # 添加日志输出
    logger.debug("===== Query Input to OpenAI =====")
    logger.debug(f"Query: {prompt}")
    logger.debug(f"System prompt: {system_prompt}")
    logger.debug("Full context:")
    for msg in messages:
        logger.debug(f"{msg['role']}: {msg['content']}")
    # bm = BaseLLMModel
    # print("ollama_demo")
    try:
        if stream:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                stream=True,
                **kwargs,
            )
        else:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                stream=False,
                **kwargs,
            )
    except Exception as e:
        logger.error(f"OpenAI 请求时出错: {e}")
        return None

    if stream:
        def stream_generator():
            try:
                for chunk in response:
                    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if content:
                        if r"\u" in content:
                            content = safe_unicode_decode(content.encode("utf-8"))
                        yield content
            except Exception as e:
                logger.error(f"流式响应时出错: {e}")
                yield STANDARD_ERROR_MSG + GENERAL_ERROR_MSG

        return stream_generator()
    else:
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        if r"\u" in content:
            content = safe_unicode_decode(content.encode("utf-8"))
        return content

class OllamaAIClient(BaseLLMModel):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        system_prompt: str = INITIAL_SYSTEM_PROMPT,
        temperature: float = 1.0,
        top_p: float = 1.0,
        user_name: str = "",
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,  # OpenAI Python SDK handles timeout internally
    ) -> None:
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
            user=user_name,
        )
        self.api_key = api_key
        self.base_url =  "http://127.0.0.1:12345/v1"
        self.timeout = timeout
        self.need_api_key = False  # 根据实际情况设置

    def get_answer_stream_iter(self) -> Generator[str, None, None]:
        """
        获取流式回答的生成器。
        """
        temp_history = self.history[:-1] # 最后会传入两个, 把 history 末位的先清掉
        if not self.api_key:
            raise ValueError("API key is not set")
        response = openai_complete_if_cache(
            model=self.model_name,
            prompt=self.get_current_prompt(),
            system_prompt=self.system_prompt,
            history_messages=temp_history,
            base_url=self.base_url,
            api_key=self.api_key,
            stream=True,
            temperature=self.temperature,
            top_p=self.top_p,
            # **self.additional_kwargs(),  # 如果有其他参数需要传递
        )
        if response is None:
            logger.error("OpenAI API returned None response")
            yield STANDARD_ERROR_MSG
            return

    
        # response = response.split("。")
        # import time

        try:
            prev_partial_text: str = ""
            for partial_text in response:
                # if partial_text.strip():
                    # print(partial_text)
                if partial_text is None or not isinstance(partial_text, str):
                    logger.warning(f"Unexpected response type: {type(partial_text)}")
                    continue
                prev_partial_text += partial_text
                # time.sleep(0.3)
                yield prev_partial_text
        except Exception as e:
            logger.error(f"Error in get_answer_stream_iter: {str(e)}", exc_info=True)
            yield STANDARD_ERROR_MSG + GENERAL_ERROR_MSG


    def get_answer_at_once(self) -> Tuple[str, int]:
        """
        一次性获取完整回答。
        """
        if not self.api_key:
            raise ValueError("API key is not set")
        temp_history = self.history[:-1]
        response = openai_complete_if_cache(
            model=self.model_name,
            prompt=self.get_current_prompt(),
            system_prompt=self.system_prompt,
            history_messages=temp_history,
            base_url=self.base_url,
            api_key=self.api_key,
            stream=False,
            temperature=self.temperature,
            top_p=self.top_p,
            # **self.additional_kwargs(),  # 如果有其他参数需要传递
        )
        if response:
            try:
                # 假设没有提供 token 计数信息，这里设为 0
                content = response
                total_token_count = 0
                return content, total_token_count
            except Exception as e:
                logger.error(f"解析响应时出错: {e}")
                raise Exception(STANDARD_ERROR_MSG + GENERAL_ERROR_MSG)
        else:
            raise Exception(STANDARD_ERROR_MSG + GENERAL_ERROR_MSG)

    def get_current_prompt(self) -> str:
        """
        获取当前的提示内容，仅包含最新的用户消息。
        """
        latest_user_message = self.get_latest_user_message()
        # print(latest_user_message)
        if latest_user_message:
            return latest_user_message
        return ""

    def get_latest_user_message(self) -> Optional[str]:
        """
        获取历史记录中最新的用户消息。
        """
        user_messages = [msg["content"] for msg in self.history if msg["role"] == "user"]
        if user_messages:
            return user_messages[-1]
        return None

    # def additional_kwargs(self) -> Dict[str, Any]:
    #     """
    #     返回额外的关键字参数，用于传递给 OpenAI API。
    #     根据需要实现或扩展此方法。
    #     """
    #     return {}
    # def auto_load(self):
    #     self.history_file_path = new_auto_history_filename(self.user_name)
    #     return self.load_chat_history()
    
# 示例使用
if __name__ == "__main__":
    client = OpenAIClient(
        model_name="gpt-4",
        api_key="your-openai-api-key",
        base_url=None,  # 如果使用代理，可以设置为代理地址
        temperature=0.7,
        top_p=0.9
    )

    # 添加对话历史（示例）
    client.history = [
        {"role": "system", "content": client.system_prompt},
        {"role": "user", "content": "你好，能帮我写一首诗吗？"}
    ]

    # 获取一次性回答
    try:
        answer, tokens = client.get_answer_at_once()
        print(f"Answer: {answer}\nTokens used: {tokens}")
    except Exception as e:
        print(str(e))

    # 获取流式回答
    try:
        print("Streamed Answer: ", end='')
        for partial in client.get_answer_stream_iter():
            print(partial, end='', flush=True)
        print()  # 换行
    except Exception as e:
        print(str(e))
