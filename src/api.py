import ast
import json
import os

import openai
import pandas as pd
from tqdm import tqdm

from .configs import Configs


class ChatGPTCall:
    def __init__(self, configs: Configs, outputs_dir: str):
        self.configs = configs
        self.outputs_dir = outputs_dir

    def set_input(self, clinical_note: str) -> list[dict]:
        """
        Setup the ChatGPT input.

        Args:
            clinical_note (str): the original clinical note

        Returns:
            list[dict]: a system prompt and a user input
        """
        system_prompt = self.configs.prompt_configs.system_prompt
        user_prompt = (
            self.configs.prompt_configs.user_prompt + " " + clinical_note
        ).strip()
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def predict(self, dataset: pd.DataFrame) -> list[str]:
        chatgpt_outputs = []
        for clinical_note in tqdm(dataset["text"]):
            chatgpt_input = self.set_input(clinical_note)
            response = openai.ChatCompletion.create(
                messages=chatgpt_input, **self.configs.model_configs.dict()
            )

            chatgpt_output = response["choices"][0]["message"]["content"]
            chatgpt_output = self.parse_output(chatgpt_output)
            chatgpt_outputs += [chatgpt_output]

        with open(os.path.join(self.outputs_dir, "predictions.json"), "w") as json_file:
            json.dump(chatgpt_outputs, json_file)

        return chatgpt_outputs

    def parse_output(self, chatgpt_output: str) -> list[dict]:
        """
        We expect ChatGPT to return a list of JSON objects,
        this function parses the output into ICD code and description

        Args:
            chatgpt_output (str): The JSON string output from ChatGPT

        Returns:
            list[dict]: The parsed outputs from the ChatGPT
        """
        try:
            parsed_outputs = ast.literal_eval(chatgpt_output)
        except ValueError:
            parsed_outputs = [
                {
                    "diagnosis": chatgpt_output,
                    "icd_code": "",
                }
            ]

        return parsed_outputs
