
from abc import ABC, abstractmethod


class BaseLLM(ABC):

    @abstractmethod
    def answer(self, prompt:str)->str:
        pass

