from abc import ABC, abstractmethod
from langchain_core.language_models import BaseLanguageModel


class LLMProvider(ABC):
    @abstractmethod
    def get_llm(self) -> BaseLanguageModel:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the model being used"""
        pass


class OpenAILLMProvider(LLMProvider):
    def __init__(self, model_name: str, temperature: float):
        from langchain_openai import ChatOpenAI

        self.model_name = model_name
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

    def get_llm(self) -> BaseLanguageModel:
        return self.llm

    def get_model_name(self) -> str:
        return self.model_name


class YandexLLMProvider(LLMProvider):
    def __init__(self, model_id: str, api_key: str, iam_token: str, temperature: float):
        from yandexchain import YandexLLM

        self.model_id = model_id
        self.llm = YandexLLM(
            model_id=model_id,
            api_key=api_key,
            iam_token=iam_token,
            temperature=temperature,
        )

    def get_llm(self) -> BaseLanguageModel:
        return self.llm

    def get_model_name(self) -> str:
        return self.model_id


class SberLLMProvider(LLMProvider):
    def __init__(self, model_id: str, api_key: str, temperature: float):
        # Заглушка для реализации Sber LLM
        self.model_id = model_id
        self.api_key = api_key
        self.temperature = temperature

    def get_llm(self) -> BaseLanguageModel:
        # Временная заглушка, будет заменена реальной реализацией
        from langchain_core.language_models import FakeListLLM

        return FakeListLLM(responses=["Ответ от Sber LLM (реализация в процессе)"])

    def get_model_name(self) -> str:
        return self.model_id


def create_llm_provider(config: dict) -> LLMProvider:
    provider_type = config.get("LLM_PROVIDER", "openai")
    temperature = config.get("LLM_TEMPERATURE", 0.7)

    if provider_type == "openai":
        return OpenAILLMProvider(config["OPENAI_MODEL_ID"], temperature)
    elif provider_type == "yandex":
        return YandexLLMProvider(
            config["YANDEX_MODEL_ID"],
            config.get("YANDEX_API_KEY", ""),
            config.get("YANDEX_IAM_TOKEN", ""),
            temperature,
        )
    elif provider_type == "sber":
        return SberLLMProvider(
            config["SBER_MODEL_ID"],
            config.get("SBER_API_KEY", ""),
            temperature,
        )
    raise ValueError(f"Неподдерживаемый провайдер LLM: {provider_type}")
