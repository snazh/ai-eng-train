import httpx

from data_pipeline.src.llm.base import BaseLLM


class LLamaCppLLM(BaseLLM):


    def __init__(self, base_url: str, timeout: int = 60, max_tokens: int = 300, temperature: float = 0):
        self.base_url = base_url
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature


    def answer(self, prompt: str):
        url = self.base_url + "/v1/chat/completions"
        response = httpx.post(url, json={
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
            }, 
            timeout=self.timeout)
        if response.status_code != 200:
            # will be replaced
            return None
        data = response.json()
        answer = data["choices"][0]["message"].get("content")
        return answer


    def available(self)->bool:
        url = self.base_url + "/health"
        try:
            response = httpx.get(url, timeout=3)
            data = response.json()

            return data.get("status") == "ok"
        except Exception as e:
            return False