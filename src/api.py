import ast
import json
import os
import time

import openai
import pandas as pd
import tiktoken
from openai.error import RateLimitError
from tqdm import tqdm

from .configs import Configs


class ChatGPTCall:
    def __init__(self, configs: Configs, outputs_dir: str):
        self.configs = configs
        self.outputs_dir = outputs_dir
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")

        self.system_prompt = self.configs.prompt_configs.system_prompt
        self.max_tokens = self.configs.model_configs.max_tokens
        self.system_prompt_length = len(self.encoding.encode(self.system_prompt))

        self.truncation_length = (
            4096 - self.system_prompt_length - self.max_tokens - 15
        )  # 15 possible differences

    def truncate_text_tokens(self, clinical_note) -> tuple[str, int]:
        """
        Truncate a string to have `max_tokens` according to the given encoding.
        """

        # Remove name, unit no., admission date, discharge date, DoB, Sex
        index = clinical_note.find("Service:")
        clinical_note = clinical_note[index:]

        encoded_clinical_note = self.encoding.encode(clinical_note)
        truncated_clinical_note = encoded_clinical_note[: self.truncation_length]
        return self.encoding.decode(truncated_clinical_note), len(encoded_clinical_note)

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
        truncated_user_prompt, original_prompt_length = self.truncate_text_tokens(
            user_prompt
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": truncated_user_prompt},
        ], original_prompt_length

    def predict(self, dataset: pd.DataFrame) -> list[str]:
        chatgpt_outputs = []
        for hadm_id, subject_id, clinical_note in tqdm(
            dataset[["hadm_id", "subject_id", "text"]].values
        ):
            chatgpt_input, original_prompt_length = self.set_input(clinical_note)
            try:
                response = openai.ChatCompletion.create(
                    messages=chatgpt_input, **self.configs.model_configs.dict()
                )
            except RateLimitError:
                # If the rate limit is reached, sleep for ~60 seconds before starting again
                print("Rate limit reached, sleep for 1 minute")
                time.sleep(60)
                print("Resuming inference")
                response = openai.ChatCompletion.create(
                    messages=chatgpt_input, **self.configs.model_configs.dict()
                )

            chatgpt_output = response["choices"][0]["message"]["content"]
            prompt_length = response["usage"]["prompt_tokens"]
            prediction, error = self.parse_output(chatgpt_output)
            chatgpt_output = {
                "hadm_id": hadm_id,
                "subject_id": subject_id,
                "prediction": prediction,
                "original_prompt_length": original_prompt_length,
                "truncated_prompt_length": prompt_length,
                "error": error,
            }
            chatgpt_outputs += [chatgpt_output]

            with open(
                os.path.join(
                    self.outputs_dir, "predictions", f"{subject_id}_{hadm_id}.json"
                ),
                "w",
            ) as json_file:
                json.dump(chatgpt_output, json_file)

        return chatgpt_outputs

    def parse_output(self, chatgpt_output: str) -> tuple[list[dict], bool]:
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
            error = False
        except:
            parsed_outputs = [
                {
                    "diagnosis": chatgpt_output,
                    "icd_code": "",
                }
            ]
            error = True

        return parsed_outputs, error
