import os
import shutil
import traceback
from enum import Enum

import commentjson as json
import gradio as gr
import tiktoken
from loguru import logger

from src import shared
from src.config import (
    retrieve_proxy,
    local_embedding,
    websearch_engine,
    bing_search_api_key,
    google_search_api_key,
    serper_search_api_key,
    searchapi_api_key,
    google_search_cx,
)
from src.index_func import construct_index
from src.presets import (
    MODEL_TOKEN_LIMIT,
    DEFAULT_TOKEN_LIMIT,
    TOKEN_OFFSET,
    REDUCE_TOKEN_FACTOR,
    STANDARD_ERROR_MSG,
    NO_APIKEY_MSG,
    BILLING_NOT_APPLICABLE_MSG,
    NO_INPUT_MSG,
    HISTORY_DIR,
    INITIAL_SYSTEM_PROMPT,
    PROMPT_TEMPLATE,
    WEBSEARCH_PTOMPT_TEMPLATE,
)
from src.search_engine import (
    search_with_google,
    search_with_duckduckgo,
    search_with_bing,
    search_with_searchapi,
    search_with_serper,
)
from src.utils import (
    i18n,
    construct_assistant,
    construct_user,
    save_file,
    hide_middle_chars,
    count_token,
    new_auto_history_filename,
    get_history_names,
    init_history_list,
    get_history_list,
    replace_special_symbols,
    get_first_history_name,
    add_source_numbers,
    add_details,
    replace_today,
    chinese_preprocessing_func,
    get_page_id,
)


class ModelType(Enum):
    Unknown = -1
    OpenAI = 0
    ChatGLM = 1
    OpenAIInstruct = 2
    OpenAIVision = 3
    Claude = 4
    Qwen = 5
    LLaMA = 6
    LM_Studio = 9
    @classmethod
    def get_type(cls, model_name: str):
        model_name_lower = model_name.lower()
        if "gpt" in model_name_lower:
            if "instruct" in model_name_lower:
                model_type = ModelType.OpenAIInstruct
            elif "vision" in model_name_lower:
                model_type = ModelType.OpenAIVision
            else:
                model_type = ModelType.OpenAI
        elif "chatglm" in model_name_lower:
            model_type = ModelType.ChatGLM
        elif "lm_studio" in model_name_lower or "ollama" in model_name_lower:
            model_type = ModelType.LM_Studio
        elif "llama" in model_name_lower or "alpaca" in model_name_lower or "yi" in model_name_lower:
            model_type = ModelType.LLaMA
        else:
            model_type = ModelType.Unknown
        return model_type
    
import threading
thread_local = threading.local()
# 创建线程本地存储
from typing import Dict, Any
from contextlib import contextmanager

class UserContextManager:
    def __init__(self):
        self.contexts = {}
        self.lock = threading.Lock()

    @contextmanager
    def get_context(self, user_name, page_id):
        """上下文管理器，确保在操作期间保持线程安全"""
        key = (user_name, page_id)
        with self.lock:
            if key not in self.contexts:
                self.contexts[key] = {
                    'chatbot': [],
                    'history': [],
                    'system_prompt': []
                }
            # logger.info(f"Context for {user_name} on page {page_id}: {self.contexts[key]}")
            yield self.contexts[key]


    def update_history(self, user_name, page_id, history):
        """更新用户的历史记录"""
        with self.lock:
            key = (user_name, page_id)
            if key in self.contexts:
                self.contexts[key]['history'] = history
                # logger.info(f"Updated history for {user_name} on page {page_id}: {history}")


    def get_chatbot(self, user_name, page_id):
        """获取用户的 chatbot"""
        key = (user_name, page_id)
        with self.lock:
            if key in self.contexts:
                return self.contexts[key]['chatbot']
        return []
    
import asyncio
import asyncio
import threading
from contextlib import contextmanager
import logging
from typing import Iterator, Tuple, Any
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Iterator, Tuple, Optional, Any, Set
import threading
import time
import logging
from contextlib import contextmanager
import uuid


logger = logging.getLogger(__name__)

@dataclass(frozen=True)  # 使用 frozen=True 使类不可变，从而可哈希
class StreamRequest:
    user_name: str
    page_id: str
    inputs: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # 添加唯一标识符
    timestamp: float = field(default_factory=time.time)
    
    # 将可变的 chatbot 和其他参数移到单独的类中
    def __hash__(self):
        return hash((self.user_name, self.page_id, self.request_id))
    
    def __eq__(self, other):
        if not isinstance(other, StreamRequest):
            return NotImplemented
        return (self.user_name, self.page_id, self.request_id) == (other.user_name, other.page_id, other.request_id)

@dataclass
class RequestData:
    chatbot: list
    fake_input: Optional[str] = None
    display_append: str = ""

class RequestQueueManager:
    def __init__(self, max_concurrent: int = 3, timeout: int = 30):
        self.queue = Queue()
        self.active_requests: Set[StreamRequest] = set()
        self.request_data = {}  # 存储请求相关的可变数据
        self.lock = threading.Lock()
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        
    @contextmanager
    def request_handler(self, request: StreamRequest, data: RequestData) -> bool:
        """处理单个请求的上下文管理器"""
        with self.lock:
            if len(self.active_requests) >= self.max_concurrent:
                self.queue.put((request, data))
                yield False
                return
            
            self.active_requests.add(request)
            self.request_data[request.request_id] = data
        
        try:
            yield True
        finally:
            with self.lock:
                self.active_requests.remove(request)
                self.request_data.pop(request.request_id, None)
                self._process_queue()
    
    def _process_queue(self):
        """处理队列中的请求"""
        try:
            while len(self.active_requests) < self.max_concurrent:
                request, data = self.queue.get_nowait()
                if time.time() - request.timestamp > self.timeout:
                    logger.warning(f"Request for {request.user_name} timed out in queue")
                    continue
                    
                self.active_requests.add(request)
                self.request_data[request.request_id] = data
                
        except Empty:
            pass


class BaseLLMModel:
    def __init__(
            self,
            model_name,
            system_prompt=INITIAL_SYSTEM_PROMPT,
            temperature=1.0,
            top_p=1.0,
            n_choices=1,
            stop="",
            max_generation_token=None,
            presence_penalty=0,
            frequency_penalty=0,
            logit_bias=None,
            user="",
            single_turn=False,
    ) -> None:
        self.history = []
        self.all_token_counts = []
        self.model_name = model_name
        self.model_type = ModelType.get_type(model_name)
        self.token_upper_limit = MODEL_TOKEN_LIMIT.get(model_name, DEFAULT_TOKEN_LIMIT)
        self.interrupted = False
        self.system_prompt = system_prompt
        self.api_key = None
        self.need_api_key = False
        self.history_file_path = get_first_history_name(user)
        self.user_name = user
        self.chatbot = []

        self.default_single_turn = single_turn
        self.default_temperature = temperature
        self.default_top_p = top_p
        self.default_n_choices = n_choices
        self.default_stop_sequence = stop
        self.default_max_generation_token = max_generation_token
        self.default_presence_penalty = presence_penalty
        self.default_frequency_penalty = frequency_penalty
        self.default_logit_bias = logit_bias
        self.default_user_identifier = user

        self.single_turn = single_turn
        self.temperature = temperature
        self.top_p = top_p
        self.n_choices = n_choices
        self.stop_sequence = stop
        self.max_generation_token = max_generation_token
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.user_identifier = user

        self.metadata = {}
        self.context_manager = UserContextManager()
        self.history_lock = threading.Lock()  # 用于保护 self.history
        self.partial_text_lock = threading.Lock()  # 用于保护 partial_text
        self.stream_lock = threading.Lock()  # 用于保护 partial_text
        self.queue_manager = RequestQueueManager(1,10)
        self.user_depth = 0

    def get_answer_stream_iter(self):
        """stream predict, need to be implemented
        conversations are stored in self.history, with the most recent question, in OpenAI format
        should return a generator, each time give the next word (str) in the answer
        """
        logger.warning("stream predict not implemented, using at once predict instead")
        response, _ = self.get_answer_at_once()
        yield response

    def get_answer_at_once(self):
        """predict at once, need to be implemented
        conversations are stored in history, with the most recent question, in OpenAI format
        Should return:
        the answer (str)
        total token count (int)
        """
        logger.warning("at once predict not implemented, using stream predict instead")
        response_iter = self.get_answer_stream_iter()
        count = 0
        response = ''
        for response in response_iter:
            count += 1
        return response, sum(self.all_token_counts) + count

    def billing_info(self):
        """get billing infomation, inplement if needed"""
        return BILLING_NOT_APPLICABLE_MSG

    def count_token(self, user_input):
        """get token count from input, implement if needed"""
        return len(user_input)
    


    def stream_next_chatbot(self, inputs, chatbot, fake_input=None, display_append=""):
        self.history_lock = threading.Lock()  # 用于保护 self.history

        def get_return_value():
            return chatbot, status_text

        status_text = i18n("开始实时传输回答……")
        if fake_input:
            chatbot.append((fake_input, ""))
        else:
            chatbot.append((inputs, ""))

        user_token_count = self.count_token(inputs)
        self.all_token_counts.append(user_token_count)
        logger.debug(f"输入token计数: {user_token_count}")
        if display_append:
            display_append = (
                    '\n\n<hr class="append-display no-in-raw" />' + display_append
            )

        partial_text = ""
        token_increment = 1
        
        for partial_text in self.get_answer_stream_iter():
            
            if type(partial_text) == tuple:
                partial_text, token_increment = partial_text
            chatbot[-1] = (chatbot[-1][0], partial_text + display_append)
            self.all_token_counts[-1] += token_increment
            status_text = self.token_message()
            yield get_return_value()
            if self.interrupted:
                self.recover()
                break
        
        self.history.append(construct_assistant(partial_text))



    def next_chatbot_at_once(self, inputs, chatbot, fake_input=None, display_append=""):
        print("走了一次性输出")
        if fake_input:
            chatbot.append((fake_input, ""))
        else:
            chatbot.append((inputs, ""))
        if fake_input is not None:
            user_token_count = self.count_token(fake_input)
        else:
            user_token_count = self.count_token(inputs)
        self.all_token_counts.append(user_token_count)
        ai_reply, total_token_count = self.get_answer_at_once()
        self.history.append(construct_assistant(ai_reply))
        if fake_input is not None:
            self.history[-2] = construct_user(fake_input)
        chatbot[-1] = (chatbot[-1][0], ai_reply + display_append)
        self.all_token_counts[-1] += count_token(construct_assistant(ai_reply))
        status_text = self.token_message()
        return chatbot, status_text
    
    from typing import Optional
    def get_latest_user_message(self) -> Optional[str]:
        # print(self.history)
        """
        获取历史记录中最新的用户消息。
        """
        user_messages = [msg["content"] for msg in self.history if msg["role"] == "user"]
        if user_messages:
            return user_messages[-1]
        return None
    
    def get_current_prompt(self) -> str:
        """
        获取当前的提示内容，仅包含最新的用户消息。
        """
        from typing import Optional

        latest_user_message = self.get_latest_user_message()
        # print(latest_user_message)
        if latest_user_message:
            return latest_user_message
        return ""


    
    def handle_lightRAG(self, prompt, stream_query=True, query_mode='mix', model_name='local',history_message=[]):
        from src.handle_lightRAG_search import stream_query
        # print(self.history)
        history_prompt = self.get_current_prompt()
        input_prompt = prompt
        # print(input_prompt)
        prompt = history_prompt if history_prompt else input_prompt
        if not prompt:
            logger.info("prompt check failed: ", prompt)
            status_text = STANDARD_ERROR_MSG
            yield status_text
        if "studio" in model_name:
            use_method = 'local'
        elif "gpt" in model_name:
            use_method = 'openai'
        else:
            use_method = 'local'

    

        res = stream_query(query=prompt, query_mode=query_mode, query_stream="stream", use_method=use_method)

        # 逐块生成响应
        for chunk in res:
            yield chunk



    def handle_file_upload(self, files, chatbot, language):
        """if the model accepts modal input, implement this function"""
        status = gr.Markdown.update()
        if files:
            construct_index(self.api_key, files=files)
            status = i18n("索引构建完成")
        return gr.Files.update(), chatbot, status

    def prepare_inputs(
            self, real_inputs, use_websearch,
            files, reply_language, chatbot,
            load_from_cache_if_possible=True,
    ):
        display_append = []
        limited_context = False
        if type(real_inputs) == list:
            fake_inputs = real_inputs[0]["text"]
        else:
            fake_inputs = real_inputs
        if files:
            from langchain.vectorstores.base import VectorStoreRetriever
            from langchain.retrievers import BM25Retriever, EnsembleRetriever
            limited_context = True
            msg = "加载索引中……"
            logger.info(msg)
            index, documents = construct_index(
                self.api_key,
                files=files,
                load_from_cache_if_possible=load_from_cache_if_possible,
            )
            assert index is not None, "获取索引失败"
            msg = "索引获取成功，生成回答中……"
            logger.info(msg)
            file_text = " ".join([d.page_content for d in documents])
            file_text_token_limit = self.token_upper_limit / 2  # 文档的token上限为模型token上限的一半
            if self.count_token(file_text) > file_text_token_limit:
                # 文档token数超限使用检索匹配，否则用知识库文件的全部数据做rag
                with retrieve_proxy():
                    if local_embedding:
                        k = 3
                        score_threshold = 0.4
                        vec_retriever = VectorStoreRetriever(
                            vectorstore=index,
                            search_type="similarity_score_threshold",
                            search_kwargs={"k": k, "score_threshold": score_threshold}
                        )
                        bm25_retriever = BM25Retriever.from_documents(
                            documents,
                            preprocess_func=chinese_preprocessing_func
                        )
                        bm25_retriever.k = k
                        retriever = EnsembleRetriever(
                            retrievers=[bm25_retriever, vec_retriever],
                            weights=[0.5, 0.5],
                        )
                    else:
                        k = 5
                        retriever = VectorStoreRetriever(
                            vectorstore=index,
                            search_type="similarity",
                            search_kwargs={"k": k}
                        )
                    try:
                        relevant_documents = retriever.get_relevant_documents(fake_inputs)
                    except:
                        return self.prepare_inputs(
                            fake_inputs,
                            use_websearch,
                            files,
                            reply_language,
                            chatbot,
                            load_from_cache_if_possible=False,
                        )
            else:
                relevant_documents = documents
            reference_results = [
                [d.page_content.strip("�"), os.path.basename(d.metadata["source"])]
                for d in relevant_documents
            ]
            reference_results = add_source_numbers(reference_results)
            display_append = add_details(reference_results)
            display_append = "\n\n" + "".join(display_append)
            if type(real_inputs) == list:
                real_inputs[0]["text"] = (
                    replace_today(PROMPT_TEMPLATE)
                    .replace("{query_str}", fake_inputs)
                    .replace("{context_str}", "\n\n".join(reference_results))
                    .replace("{reply_language}", reply_language)
                )
            else:
                real_inputs = (
                    replace_today(PROMPT_TEMPLATE)
                    .replace("{query_str}", real_inputs)
                    .replace("{context_str}", "\n\n".join(reference_results))
                    .replace("{reply_language}", reply_language)
                )
        elif use_websearch:
            if websearch_engine == "google":
                search_results = search_with_google(fake_inputs, google_search_api_key, google_search_cx)
            elif websearch_engine == "bing":
                search_results = search_with_bing(fake_inputs, bing_search_api_key)
            elif websearch_engine == "searchapi":
                search_results = search_with_searchapi(fake_inputs, searchapi_api_key)
            elif websearch_engine == "serper":
                search_results = search_with_serper(fake_inputs, serper_search_api_key)
            else:
                search_results = search_with_duckduckgo(fake_inputs)
            # reference_results = []
            # for idx, result in enumerate(search_results):
            #     # logger.info("search_results, for idx result in ")

            #     logger.info(f"搜索结果{idx + 1}：{result}")
            #     reference_results.append([result["snippet"], result["url"]]) 
            #     display_append.append(
            #         f"<a href=\"{result['url']}\" target=\"_blank\">{idx + 1}.&nbsp;{result['name']}</a>"
            #     )
            # reference_results = add_source_numbers(reference_results)
            # display_append = (
            #         '<div class = "source-a">' + "".join(display_append) + "</div>"
            # )
            reference_results = []
            for idx, result in enumerate(search_results):
                logger.info(f"搜索结果{idx + 1}：{result}")
                url = result.get("link", "URL not available")  # Using .get() for safety
                reference_results.append([result["snippet"], url])
                display_append.append(
                    f"<a href=\"{url}\" target=\"_blank\">{idx + 1}.&nbsp;{result.get('title', 'Unnamed')}</a>"
                )
            reference_results = add_source_numbers(reference_results)
            display_append = (
                '<div class = "source-a">' + "".join(display_append) + "</div>"
            )

            if type(real_inputs) == list:
                real_inputs[0]["text"] = (
                    replace_today(WEBSEARCH_PTOMPT_TEMPLATE)
                    .replace("{query}", fake_inputs)
                    .replace("{web_results}", "\n\n".join(reference_results))
                    .replace("{reply_language}", reply_language)
                )
            else:
                real_inputs = (
                    replace_today(WEBSEARCH_PTOMPT_TEMPLATE)
                    .replace("{query}", fake_inputs)
                    .replace("{web_results}", "\n\n".join(reference_results))
                    .replace("{reply_language}", reply_language)
                )
        else:
            display_append = ""
        return limited_context, fake_inputs, display_append, real_inputs, chatbot
    
    def get_user_context(self, user_name, page_id):
        """获取用户的上下文"""
        with self.context_manager.get_context(user_name, page_id) as context:
            # logger.info(f"Initial context for {user_name} on page {page_id}: {context}")
            return context
    
    def LightRAG_predict(
            self,
            user_name,
            inputs,
            chatbot,
            stream=False,
            use_websearch=False,
            lightrag_chat_checkbox=False,
            files=None,
            reply_language="中文",
            should_check_token_count=True,

    ):


        page_id = get_page_id(user_name)
        context = self.get_user_context(user_name, page_id)

        if not chatbot:
            context['chatbot'] = []
            context['history'] = []
        elif context['chatbot'] != chatbot:
            context['chatbot'] = list(chatbot)
 

        if lightrag_chat_checkbox:
            if isinstance(inputs, list):
                chatbot.append((inputs[0]["text"], ""))
            else:
                chatbot.append((inputs, ""))
            yield chatbot
        else:
            yield chatbot  # 返回独立的 chatbot



    def predict(
            self,
            user_name,
            inputs,
            chatbot,            
            stream=False,
            use_websearch=False,
            lightrag_chat_checkbox=False,
            files=None,
            reply_language="中文",
            
            should_check_token_count=True,

            
    ):
        
        page_id = get_page_id(user_name)
        context = self.get_user_context(user_name, page_id)
        
        # 更新 history
        if context['history'] != self.history:
            # logger.info(f"Updating history for {user_name}, page {page_id}")
            # context['history'] = self.history
            self.context_manager.update_history(user_name, page_id, self.history)  
        self.history = context['history']  # 使用更新后的 history

        if reply_language == "跟随问题语言（不稳定）":
            reply_language = "the same language as the question, such as English, 中文, 日本語, Español, Français, or Deutsch."

        if lightrag_chat_checkbox:
            # print(self.system_prompt)
            # self.history.append(construct_user(inputs))
            if type(inputs) == list:
                self.history.append(inputs)
            else:
                self.history.append(construct_user(inputs))
            accumulated_text = ""

            # print(self)如果要全量测试 SPM, 你能给出全量测试的具体用例吗, 并在最后提出平常容易忽视的的细节
            from src.models import current_model_name
            # print(current_model_name)
            iter = self.handle_lightRAG(prompt=inputs, stream_query=stream, model_name=current_model_name)   
            logger.info(f"用户{self.user_name}的输入为：{inputs}, 选择使用 lightrag_chat_checkbox, 走调用后台 API逻辑v2, {current_model_name} ")       
            # status_text = "querying, please wait..." # 右上角状态信息
            # status_text = i18n("开始实时传输回答……")
            # print(chatbot, "type", type(chatbot))
            # print(status_text, "type", type(status_text))
            for chunk in iter:
                accumulated_text += chunk
                # 更新 chatbot 的最后一个条目为累积的响
                if chatbot: # and isinstance(chatbot[-1], tuple):
                    chatbot[-1] = (chatbot[-1][0], accumulated_text)
                    # print("----",chatbot)
                else:
                    chatbot.append((input, accumulated_text))

                status_text = i18n("开始实时传输回答……")
                yield chatbot,status_text
            # print(chatbot, "type", type(chatbot))
            self.history.append(construct_assistant(accumulated_text)) 

            # pass # hanhale at LightRAG_predict
        else:
            # if not lightrag_chat_checkbox:
            # self.history.append(construct_user(inputs))

            status_text = i18n("开始实时传输回答……")
            if type(inputs) == list:
                logger.info(f"用户{self.user_name}的输入为：{inputs[0]['text']}")
            else:
                logger.info(f"用户{self.user_name}的输入为：{inputs}")

            if should_check_token_count:
                if type(inputs) == list:
                    yield chatbot + [(inputs[0]["text"], "")], status_text
                else:
                    yield chatbot + [(inputs, "")], status_text


            limited_context, fake_inputs, display_append, inputs, chatbot = self.prepare_inputs(
                real_inputs=inputs,
                use_websearch=use_websearch,
                files=files,
                reply_language=reply_language,
                chatbot=chatbot
            )
            yield chatbot + [(fake_inputs, "")], status_text

            if (
                    self.need_api_key and
                    self.api_key is None
                    and not shared.state.multi_api_key
            ):
                status_text = STANDARD_ERROR_MSG + NO_APIKEY_MSG
                logger.info(status_text)
                chatbot.append((inputs, ""))
                if len(self.history) == 0:
                    self.history.append(construct_user(inputs))
                    self.history.append("")
                    self.all_token_counts.append(0)
                else:
                    self.history[-2] = construct_user(inputs)
                yield chatbot + [(inputs, "")], status_text
                return

            elif len(inputs.strip()) == 0:
                status_text = STANDARD_ERROR_MSG + NO_INPUT_MSG
                logger.info(status_text)
                yield chatbot + [(inputs, "")], status_text
                return
            
            
            if self.single_turn:
                self.history = []
                self.all_token_counts = []
            
            if type(inputs) == list:
                self.history.append(inputs)
            else:
                self.history.append(construct_user(inputs))
            # logger.info(f"After append:  {user_name} self.history={self.history}, context['history']={context['history']}")

            try:
                
                if stream and not lightrag_chat_checkbox:
                    logger.debug("使用流式传输")
                    iter = self.stream_next_chatbot(
                        inputs,
                        chatbot,
                        fake_input=fake_inputs,
                        display_append=display_append,
    
                    )
                    for chatbot, status_text in iter:
                        yield chatbot, status_text

                elif not stream and not lightrag_chat_checkbox:
                    logger.debug("不使用流式传输")
                    chatbot, status_text = self.next_chatbot_at_once(
                        inputs,
                        chatbot,
                        fake_input=fake_inputs,
                        display_append=display_append,
                    )
                    yield chatbot, status_text

            except Exception as e:
                traceback.print_exc()
                status_text = STANDARD_ERROR_MSG + str(e)
                yield chatbot, status_text

            if len(self.history) > 1 and self.history[-1]["content"] != inputs:
                # logger.info(f"{self.user_name},回答为：{self.history[-1]['content']}")
                pass
            # logger.info(f''' if "No valid response accumulated" in {self.history}:  ''')


            if limited_context:
                self.history = []
                self.all_token_counts = []

            max_token = self.token_upper_limit - TOKEN_OFFSET

            if sum(self.all_token_counts) > max_token and len(self.history) > 2 and should_check_token_count:
                count = 0
                while (
                        sum(self.all_token_counts)
                        > self.token_upper_limit * REDUCE_TOKEN_FACTOR
                        and sum(self.all_token_counts) > 0 and len(self.history) > 0
                ):
                    count += 1
                    del self.all_token_counts[:1]
                    del self.history[:2]
                status_text = f"为了防止token超限，模型忘记了早期的 {count} 轮对话"
                logger.info(status_text)
                yield chatbot, status_text


    def retry(
            self,
            user_name,
            inputs,
            chatbot,            
            stream=False,
            use_websearch=False,
            lightrag_chat_checkbox=False,
            files=None,
            reply_language="中文",
            should_check_token_count=True,
    ):  
        self.user_depth += 1
        logger.info("重试中……")
        if len(self.history) > 1:
            inputs = self.history[-2]["content"]
            del self.history[-2:]
            if len(self.all_token_counts) > 0:
                self.all_token_counts.pop()
        elif len(chatbot) > 0:
            inputs = chatbot[-1][0]
            if '<div class="user-message">' in inputs:
                inputs = inputs.split('<div class="user-message">')[1]
                inputs = inputs.split("</div>")[0]
        elif len(self.history) == 1:
            inputs = self.history[-1]["content"]
            del self.history[-1]
        else:
            yield chatbot, f"{STANDARD_ERROR_MSG}上下文是空的"
            return

        iter = self.predict(
            user_name,
            inputs,
            chatbot,            
            stream,
            use_websearch,
            lightrag_chat_checkbox,
            files,
            reply_language,
            should_check_token_count,
        )
        for x in iter:
            yield x
        logger.info("重试完毕")


    def interrupt(self):
        self.interrupted = True

    def recover(self):
        self.interrupted = False

    def set_token_upper_limit(self, new_upper_limit):
        self.token_upper_limit = new_upper_limit
        logger.info(f"token上限设置为{new_upper_limit}")
        self.auto_save()

    def set_temperature(self, new_temperature):
        self.temperature = new_temperature
        self.auto_save()

    def set_top_p(self, new_top_p):
        self.top_p = new_top_p
        self.auto_save()

    def set_n_choices(self, new_n_choices):
        self.n_choices = new_n_choices
        self.auto_save()

    def set_stop_sequence(self, new_stop_sequence: str):
        new_stop_sequence = new_stop_sequence.split(",")
        self.stop_sequence = new_stop_sequence
        self.auto_save()

    def set_max_tokens(self, new_max_tokens):
        self.max_generation_token = new_max_tokens
        self.auto_save()

    def set_presence_penalty(self, new_presence_penalty):
        self.presence_penalty = new_presence_penalty
        self.auto_save()

    def set_frequency_penalty(self, new_frequency_penalty):
        self.frequency_penalty = new_frequency_penalty
        self.auto_save()

    def set_logit_bias(self, logit_bias):
        self.logit_bias = logit_bias
        self.auto_save()

    def encoded_logit_bias(self):
        if self.logit_bias is None:
            return {}
        logit_bias = self.logit_bias.split()
        bias_map = {}
        encoding = tiktoken.get_encoding("cl100k_base")
        for line in logit_bias:
            word, bias_amount = line.split(":")
            if word:
                for token in encoding.encode(word):
                    bias_map[token] = float(bias_amount)
        return bias_map

    def set_user_identifier(self, new_user_identifier):
        self.user_identifier = new_user_identifier
        self.auto_save()

    def set_system_prompt(self, new_system_prompt):
        self.system_prompt = new_system_prompt
        self.auto_save()

    def set_key(self, new_access_key):
        self.api_key = new_access_key.strip()
        msg = i18n("API密钥更改为了") + hide_middle_chars(self.api_key)
        logger.info(msg)
        return self.api_key, msg

    def set_single_turn(self, new_single_turn):
        self.single_turn = new_single_turn
        self.auto_save()

    def reset(self, remain_system_prompt=False):
        self.history = []
        self.all_token_counts = []
        self.interrupted = False
        self.history_file_path = new_auto_history_filename(self.user_name)
        history_name = self.history_file_path[:-5]
        choices = get_history_names(self.user_name)
        # print(self.history_file_path,"---",self.history_file_path,"---",history_name,"---",choices)
        if history_name not in choices:
            choices.insert(0, history_name)
        # print(self.history_file_path,"---",self.history_file_path,"---",history_name,"---",choices)
        system_prompt = self.system_prompt if remain_system_prompt else ""

        self.single_turn = self.default_single_turn
        self.temperature = self.default_temperature
        self.top_p = self.default_top_p
        self.n_choices = self.default_n_choices
        self.stop_sequence = self.default_stop_sequence
        self.max_generation_token = self.default_max_generation_token
        self.presence_penalty = self.default_presence_penalty
        self.frequency_penalty = self.default_frequency_penalty
        self.logit_bias = self.default_logit_bias
        self.user_identifier = self.default_user_identifier

        return (
            [],
            self.token_message([0]),
            gr.Radio.update(choices=choices, value=history_name),
            system_prompt,
            self.single_turn,
            self.temperature,
            self.top_p,
            self.n_choices,
            self.stop_sequence,
            self.token_upper_limit,
            self.max_generation_token,
            self.presence_penalty,
            self.frequency_penalty,
            self.logit_bias,
            self.user_identifier,
        )

    def delete_first_conversation(self):
        if self.history:
            del self.history[:2]
            del self.all_token_counts[:1]
        return self.token_message()

    def delete_last_conversation(self, chatbot):
        if len(chatbot) > 0 and STANDARD_ERROR_MSG in chatbot[-1][1]:
            msg = "由于包含报错信息，只删除chatbot记录"
            chatbot = chatbot[:-1]
            return chatbot, self.history
        if len(self.history) > 0:
            self.history = self.history[:-2]
        if len(chatbot) > 0:
            msg = "删除了一组chatbot对话"
            chatbot = chatbot[:-1]
        if len(self.all_token_counts) > 0:
            msg = "删除了一组对话的token计数记录"
            self.all_token_counts.pop()
        msg = "删除了一组对话"
        self.chatbot = chatbot
        self.auto_save(chatbot)
        return chatbot, msg

    def token_message(self, token_lst=None):
        if token_lst is None:
            token_lst = self.all_token_counts
        token_sum = 0
        for i in range(len(token_lst)):
            token_sum += sum(token_lst[: i + 1])
        return (
                i18n("Token 计数: ")
                + f"{sum(token_lst)}"
                + i18n("，本次对话累计消耗了 ")
                + f"{token_sum} tokens"
        )

    def rename_chat_history(self, filename, chatbot):
        if filename == "":
            return gr.update()
        if not filename.endswith(".json"):
            filename += ".json"
        self.delete_chat_history(self.history_file_path)
        # 命名重复检测
        repeat_file_index = 2
        full_path = os.path.join(HISTORY_DIR, self.user_name, filename)
        while os.path.exists(full_path):
            full_path = os.path.join(
                HISTORY_DIR, self.user_name, f"{repeat_file_index}_{filename}"
            )
            repeat_file_index += 1
        filename = os.path.basename(full_path)

        self.history_file_path = filename
        save_file(filename, self, chatbot)
        return init_history_list(self.user_name)

    def auto_name_chat_history(
            self, name_chat_method, user_question, chatbot, single_turn_checkbox
    ):
        if len(self.history) == 2 and not single_turn_checkbox:
            user_question = self.history[0]["content"]
            if type(user_question) == list:
                user_question = user_question[0]["text"]
            filename = replace_special_symbols(user_question)[:16] + ".json"
            return self.rename_chat_history(filename, chatbot)
        else:
            return gr.update()

    def auto_save(self, chatbot=None):
        if chatbot is not None:
            save_file(self.history_file_path, self, chatbot)

    def export_markdown(self, filename, chatbot):
        if filename == "":
            return
        if not filename.endswith(".md"):
            filename += ".md"
        save_file(filename, self, chatbot)

    def load_chat_history(self, new_history_file_path=None):
        logger.debug(f"{self.user_name} 加载对话历史中……")
        if new_history_file_path is not None:
            if type(new_history_file_path) != str:
                # copy file from new_history_file_path.name to os.path.join(HISTORY_DIR, self.user_name)
                new_history_file_path = new_history_file_path.name
                shutil.copyfile(
                    new_history_file_path,
                    os.path.join(
                        HISTORY_DIR,
                        self.user_name,
                        os.path.basename(new_history_file_path),
                    ),
                )
                self.history_file_path = os.path.basename(new_history_file_path)
            else:
                self.history_file_path = new_history_file_path
        try:
            if self.history_file_path == os.path.basename(self.history_file_path):
                history_file_path = os.path.join(
                    HISTORY_DIR, self.user_name, self.history_file_path
                )
            else:
                history_file_path = self.history_file_path
            if not self.history_file_path.endswith(".json"):
                history_file_path += ".json"
            saved_json = {}
            if os.path.exists(history_file_path):
                with open(history_file_path, "r", encoding="utf-8") as f:
                    saved_json = json.load(f)
            try:
                if type(saved_json["history"][0]) == str:
                    logger.info("历史记录格式为旧版，正在转换……")
                    new_history = []
                    for index, item in enumerate(saved_json["history"]):
                        if index % 2 == 0:
                            new_history.append(construct_user(item))
                        else:
                            new_history.append(construct_assistant(item))
                    saved_json["history"] = new_history
                    logger.info(new_history)
            except:
                pass
            if len(saved_json["chatbot"]) < len(saved_json["history"]) // 2:
                logger.info("Trimming corrupted history...")
                saved_json["history"] = saved_json["history"][-len(saved_json["chatbot"]):]
                logger.info(f"Trimmed history: {saved_json['history']}")
            logger.debug(f"{self.user_name} 加载对话历史完毕")
            self.history = saved_json["history"]
            self.single_turn = saved_json.get("single_turn", self.single_turn)
            self.temperature = saved_json.get("temperature", self.temperature)
            self.top_p = saved_json.get("top_p", self.top_p)
            self.n_choices = saved_json.get("n_choices", self.n_choices)
            self.stop_sequence = list(saved_json.get("stop_sequence", self.stop_sequence))
            self.token_upper_limit = saved_json.get(
                "token_upper_limit", self.token_upper_limit
            )
            self.max_generation_token = saved_json.get(
                "max_generation_token", self.max_generation_token
            )
            self.presence_penalty = saved_json.get(
                "presence_penalty", self.presence_penalty
            )
            self.frequency_penalty = saved_json.get(
                "frequency_penalty", self.frequency_penalty
            )
            self.logit_bias = saved_json.get("logit_bias", self.logit_bias)
            self.user_identifier = saved_json.get("user_identifier", self.user_name)
            self.metadata = saved_json.get("metadata", self.metadata)
            self.chatbot = saved_json["chatbot"]
            return (
                os.path.basename(self.history_file_path)[:-5],
                saved_json["system"],
                saved_json["chatbot"],
                self.single_turn,
                self.temperature,
                self.top_p,
                self.n_choices,
                ",".join(self.stop_sequence),
                self.token_upper_limit,
                self.max_generation_token,
                self.presence_penalty,
                self.frequency_penalty,
                self.logit_bias,
                self.user_identifier,
            )
        except:
            # 没有对话历史或者对话历史解析失败
            logger.info(f"没有找到对话历史记录 {self.history_file_path}")
            self.reset()
            return (
                os.path.basename(self.history_file_path),
                "",
                [],
                self.single_turn,
                self.temperature,
                self.top_p,
                self.n_choices,
                ",".join(self.stop_sequence),
                self.token_upper_limit,
                self.max_generation_token,
                self.presence_penalty,
                self.frequency_penalty,
                self.logit_bias,
                self.user_identifier,
            )

    def delete_chat_history(self, filename):
        if filename == "CANCELED":
            return gr.update(), gr.update(), gr.update()
        if filename == "":
            return i18n("你没有选择任何对话历史"), gr.update(), gr.update()
        if filename and not filename.endswith(".json"):
            filename += ".json"
        if filename == os.path.basename(filename):
            history_file_path = os.path.join(HISTORY_DIR, self.user_name, filename)
        else:
            history_file_path = filename
        md_history_file_path = history_file_path[:-5] + ".md"
        try:
            os.remove(history_file_path)
            os.remove(md_history_file_path)
            return i18n("删除对话历史成功"), get_history_list(self.user_name), []
        except:
            logger.info(f"删除对话历史失败 {history_file_path}")
            return (
                i18n("对话历史") + filename + i18n("已经被删除啦"),
                get_history_list(self.user_name),
                [],
            )

    def auto_load(self):
        self.history_file_path = new_auto_history_filename(self.user_name)
        return self.load_chat_history()

    def like(self):
        """like the last response, implement if needed"""
        return gr.update()

    def dislike(self):
        """dislike the last response, implement if needed"""
        return gr.update()

    def deinitialize(self):
        """deinitialize the model, implement if needed"""
        pass
