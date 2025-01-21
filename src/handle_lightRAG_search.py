import requests
import json
host = '127.0.0.1:8020'
end_point = '/query'
query_mode = 'mix'
query_stream = 'stream'
use_method = 'local'
# use_method = openai or local
# def send_quey():



url = f"http://{host}{end_point}"
def stream_query(query, query_mode="mix", query_stream=True, use_method="local"):
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "mode": query_mode,
        "query_stream": "query_stream",
        "use_method": use_method
    }

    try:
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()  # 检查是否请求成功

            # print("Streaming response:")
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    # 假设服务器发送的是文本数据，可以解码后打印
                    yield chunk.decode('utf-8')  # 解码为字符串

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  # HTTP 错误
        print(response.request.headers,response.request.body,response.text)
        print("----",response.text)
    except Exception as err:
        print(f"Other error occurred: {err}")  # 其他错误
if __name__ == "__main__":
    host = '127.0.0.1:8020'  # 确保端口号与 FastAPI 服务一致
    end_point = '/query'
    query = "请问明天的天气如何？"
    query_mode = 'mix'
    query_stream = True
    use_method = 'local'

    res = stream_query('请问明天的天气如何？', query_mode, query_stream='1', use_method='local')
    print(res)
