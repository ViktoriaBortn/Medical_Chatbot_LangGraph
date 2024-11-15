from langgraph.graph import StateGraph, START, END
from langchain_openai import OpenAI
import os
from typing import TypedDict

# Установка ключа OpenAI
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
if not OpenAI.api_key:
    raise EnvironmentError("API key for OpenAI is not set. Please define 'OPENAI_API_KEY' in your environment variables.")

# Описание состояния
class State(TypedDict):
    symptoms: str
    recommendation: str

# Инициализация модели
llm = OpenAI(model="text-davinci-003", temperature=0)

# Создание графа состояний с заданной схемой
graph = StateGraph(state_schema=State)

# Логика каждого узла
def greeting(state: State):
    print("Привет! Я медицинский бот. Чем я могу помочь?")
    return {"next_state": "collect_symptoms"}

def collect_symptoms(state: State):
    symptoms = input("Пожалуйста, опишите свои симптомы: ")
    state["symptoms"] = symptoms
    return {"next_state": "provide_recommendation"}

def provide_recommendation(state: State):
    symptoms = state["symptoms"]
    prompt = f"Пациент сообщает следующие симптомы: {symptoms}. Какие возможные причины и рекомендации?"
    response = llm(prompt)
    print("\nВозможные причины и рекомендации:")
    print(response)
    return {"next_state": END}

# Добавление узлов с действиями
graph.add_node("greeting", action=greeting)
graph.add_node("collect_symptoms", action=collect_symptoms)
graph.add_node("provide_recommendation", action=provide_recommendation)

# Добавление переходов
graph.add_edge(START, "greeting")  # Связывание START с первым узлом
graph.add_edge("greeting", "collect_symptoms")
graph.add_edge("collect_symptoms", "provide_recommendation")
graph.add_edge("provide_recommendation", END)  # Завершение графа

# Ручное выполнение графа
if __name__ == "__main__":
    current_state = START  # Установить начальное состояние
    state_data = {}  # Хранилище данных между узлами

    while current_state != END:
        # Проверка существования узла
        if current_state not in graph.nodes:
            raise RuntimeError(f"No node defined for state '{current_state}'")

        # Получение действия узла
        node = graph.nodes[current_state]
        node_action = node.action

        if not node_action:
            raise RuntimeError(f"No action defined for state '{current_state}'")

        # Выполнение действия узла
        result = node_action(state_data)
        current_state = result.get("next_state", END)

    print("Граф завершен.")
