
import json
from typing import Any, Optional

import ollama
from dotenv import load_dotenv

from memory import ChatMessage, BaseMemory
from tools import run_callable, today_is_tool, weather_tool, day_of_week_tool, add_numbers_tool

load_dotenv(interpolate=False)


class BaseChatModel:
    def __init__(
        self,
        client: Any,
        model: str,
        temperature: float = 0.0,
        keep_alive: int = -1,
        format: Optional[str] = None,
        max_stored_memory_messages: int = 50,
    ) -> None:

        self.client = client
        self.model = model
        self.format = format
        self.keep_alive = keep_alive
        self.temperature = temperature
        self.memory = BaseMemory(max_size=max_stored_memory_messages)
        self.system = """You are a helpful AI assistant. When using tools, always provide a natural, complete response using the information gathered.
        Format your response as a short, coherent sentence. You can provide wrong answers, but make sure they are funny and engaging."""

    def message(self, human: str, ai: str) -> ChatMessage:
        return ChatMessage(message={"human": human, "ai": ai})


class OllamaChatModel(BaseChatModel):
    """http://localhost:11434 is the default Ollama port serving API."""

    def __init__(self, tools: list[dict], model: str = "llama3.2") -> None:
        self.model = model
        self.tools = tools
        self.client = ollama.Client(host="http://localhost:11434")
        super().__init__(client=self.client, model=self.model)

    def extract(self, tool_call) -> list:
        """Extract and execute tool call"""
        data = []

        if not isinstance(tool_call, list):
            tool_call = [tool_call]

        for tool in tool_call:
            func_name = tool.function.name
            if isinstance(tool.function.arguments, str):
                func_arguments = json.loads(tool.function.arguments)
            else:
                func_arguments = tool.function.arguments
            result = run_callable(func_name, func_arguments)
            data.append(result)
        return data

    def response(self, user_prompt: str, system_message: str = None) -> ollama.ChatResponse:
        messages = [
            {
                "role": "system",
                "content": system_message if system_message else self.system,
            }
        ]

        for msg in self.memory.get():
            if isinstance(msg, ChatMessage):
                messages.extend(
                    [
                        {"role": "user", "content": msg.human},
                        {"role": "assistant", "content": msg.ai},
                    ]
                )

        messages.append({"role": "user", "content": user_prompt})

        return self.client.chat(
            model=self.model,
            messages=messages,
            format=self.format,
            keep_alive=self.keep_alive,
            tools=self.tools,
            stream=True
        )

    def chat(self, system_message: str = None, save_chat: bool = True) -> None:
        system_message = system_message if system_message else self.system

        while True:
            user_prompt = input("User: ")
            if user_prompt == "bye":
                self.memory.add(self.message(human=user_prompt, ai="Bye"))
                if save_chat:
                    self.memory.save(model_name=str(self.model))
                self.memory.clear()
                print("AI: Bye")
                break

            response = self.response(user_prompt, system_message)

            for part in response:
                print(f"Inner: {part.message}")
                print(f"Inner: Is stream done? {part.done_reason}", end="\n\n")
                # If there are tool calls, process them and get final response
                if hasattr(part.message, "tool_calls") and part.message.tool_calls:
                    collected_data = {}

                    for tool_call in part.message.tool_calls:
                        result = self.extract(tool_call)
                        collected_data[tool_call.function.name] = result

                    final_prompt = (
                        f"Based on the following information:\n"
                        f"{collected_data}"
                        f"Please provide a natural response to the original question: '{user_prompt}'"
                    )

                    final_response = self.client.chat(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "Sumup your response before sending back. Make is short and concise"
                                + " "
                                + system_message,
                            },
                            {"role": "user", "content": final_prompt},
                        ],
                    )
                    response_content = final_response.message.content
                else:
                    # If no tool calls, use the original response
                    response_content = part.message.content

                if response_content:
                    print(f"AI: {response_content}", end="\n\n")
                    self.memory.add(self.message(human=user_prompt, ai=response_content))

                if hasattr(part, "done_reason"):
                    if part.done_reason == "stop":
                        break


def main():
    tools = [today_is_tool, weather_tool, day_of_week_tool, add_numbers_tool]
    model = OllamaChatModel(tools=tools)
    model.chat(save_chat=True)


if __name__ == "__main__":
    main()
