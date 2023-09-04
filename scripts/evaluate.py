import argparse
import gc
import json
import logging
import os
import re
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from submodules.CoPHE.scripts import evaluation_setup, multi_level_eval

logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., INFO, DEBUG, ERROR)
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--groundtruth_path", type=str, required=True)
    args = parser.parse_args()

    return args


def load_groundtruth(groundtruth_path: str) -> pd.DataFrame:
    return pd.read_csv(groundtruth_path)


def load_predictions(predictions_dir: str, groundtruth: pd.DataFrame) -> list[dict]:
    predictions = []
    for prediction_file in tqdm(os.listdir(predictions_dir)):
        prediction = json.load(
            open(os.path.join(predictions_dir, prediction_file), "r")
        )
        label_codes = parse_groundtruth(
            groundtruth.loc[
                (groundtruth["hadm_id"] == prediction["hadm_id"])
                & (groundtruth["subject_id"] == prediction["subject_id"])
            ]["labels"].values[0]
        )

        parsed_prediction = parse_prediction(prediction)
        parsed_prediction["label_codes"] = label_codes

        predictions += [parsed_prediction]

    return predictions


def parse_groundtruth(labels: str):
    return labels.split(";")


def parse_prediction(prediction_dict: dict) -> {}:
    predicted_diagnoses = []
    predicted_codes = []
    if not prediction_dict["error"]:
        # If there was no ast.literal parse error,
        # the predictions are correctly structured as a JSON dict
        if type(prediction_dict["prediction"]) == list:
            for prediction in prediction_dict["prediction"]:
                predicted_diagnoses += [prediction["diagnosis"]]
                predicted_codes += [prediction["icd_code"]]
        elif type(prediction_dict["prediction"]) == dict:
            predicted_diagnoses += [prediction_dict["prediction"]["diagnosis"]]
            predicted_codes += [prediction_dict["prediction"]["icd_code"]]
        else:
            raise ValueError(f"Prediction unknown type:\n{prediction}")
    else:
        # If there was ast.literal parse error,
        # the predictions are combined as 1 string in the first diagnosis
        prediction = prediction_dict["prediction"][0]["diagnosis"]

        # Regular expression pattern to match diagnosis and ICD-10 code pairs
        pattern = r"\d+\.\s(.*?)-\s([A-Z]\d{2}\.\d{1,3})"

        matches = re.findall(pattern, prediction, re.DOTALL)

        for match in matches:
            predicted_diagnoses += [match[0].strip()]
            predicted_codes += [match[1]]

    return {
        "hadm_id": prediction_dict["hadm_id"],
        "subject_id": prediction_dict["subject_id"],
        "original_prompt_length": prediction_dict["original_prompt_length"],
        "truncated_prompt_length": prediction_dict["truncated_prompt_length"],
        "error": prediction_dict["error"],
        "predicted_diagnoses": predicted_diagnoses,
        "predicted_codes": predicted_codes,
    }


def retrieve_chapter(code_with_etiology):
    """
    @modr00cka

    Args:
        code_with_etiology (_type_): _description_

    Returns:
        _type_: _description_
    """
    code = code_with_etiology[:3]
    char1 = code[0]
    char2 = code[1]
    if char1 in ["A", "B"]:
        return "I"
    elif char1 == "C":
        return "II"
    elif char1 == "D":
        if str.isnumeric(char2):
            if int(char2) < 5:
                return "II"
            else:
                return "III"
        else:
            return code_with_etiology[:2]
    elif char1 == "E":
        return "IV"
    elif char1 == "F":
        return "V"
    elif char1 == "G":
        return "VI"
    elif char1 == "H":
        if str.isnumeric(char2):
            if int(char2) < 6:
                return "VII"
            else:
                return "VIII"
        else:
            return code_with_etiology[:2]
    elif char1 == "I":
        return "IX"
    elif char1 == "J":
        return "X"
    elif char1 == "K":
        return "XI"
    elif char1 == "L":
        return "XII"
    elif char1 == "M":
        return "XIII"
    elif char1 == "N":
        return "XIV"
    elif char1 == "O":
        return "XV"
    elif char1 == "P":
        return "XVI"
    elif char1 == "Q":
        return "XVII"
    elif char1 == "R":
        return "XVIII"
    elif char1 in ["S", "T"]:
        return "XIX"
    elif char1 in ["V", "W", "X", "Y"]:
        return "XX"
    elif char1 == "Z":
        return "XXI"
    elif char1 == "U":
        return "XXII"
    else:
        logging.info(f"{code} ERROR: {code_with_etiology}")
        return "ERROR"


def prep_unseen_code(unseen_code):
    """
    @modr00cka

    Args:
        unseen_code (_type_): _description_

    Returns:
        _type_: _description_
    """
    result_dict = {}
    ancestry = []

    for i in range(8, 4, -1):
        ancestry.append(unseen_code[:i])

    ancestry.append(unseen_code[:3])
    if str.isalpha(unseen_code[0]):
        ancestry.append(retrieve_chapter(unseen_code))
    else:
        ancestry.append(unseen_code[:2])
    result_dict["concept_id"] = unseen_code
    result_dict["label"] = "hallucinated code"
    result_dict["parents"] = ancestry
    return result_dict


def main():
    args = parse_args()

    groundtruth = load_groundtruth(args.groundtruth_path)
    predictions = load_predictions(args.predictions_dir, groundtruth)

    # Codes to multihot using the matrix
    translation_dict_icd10: dict = evaluation_setup.load_translation_dict_from_icd10(
        "submodules/CoPHE/ICD10/icd10_graph_desc.json"
    )

    logging.info(f"Initial size of icd10 mapping: {len(translation_dict_icd10.keys())}")

    errors = {"parsing_error": [], "no_parsing_error": []}

    for prediction in predictions:
        for code in prediction["predicted_codes"]:
            if code not in translation_dict_icd10:
                if prediction["error"]:
                    errors["parsing_error"] += [code]
                else:
                    errors["no_parsing_error"] += [code]
            if code[:3] not in translation_dict_icd10:
                if prediction["error"]:
                    errors["parsing_error"] += [code[:3]]
                else:
                    errors["no_parsing_error"] += [code[:3]]

    logging.info("Hallucinated codes")
    logging.info(f"With Parsing Error: {len(errors['parsing_error'])}")
    logging.info(f"Without Parsing Error: {len(errors['no_parsing_error'])}")

    fake_codes = dict()
    for error in errors["parsing_error"] + errors["no_parsing_error"]:
        unseen_code_data = prep_unseen_code(error)
        fake_codes[error] = unseen_code_data
    translation_dict_icd10.update(fake_codes)

    # confirming ancestors
    ontology_errors = set()
    for code in translation_dict_icd10:
        for i in range(8, 4, -1):
            if code[:i] not in translation_dict_icd10:
                ontology_errors.add(code[:i])
            if code[:3] not in translation_dict_icd10:
                ontology_errors.add(code[:3])

    # resolving ontology errors
    slack_codes = dict()
    for error in ontology_errors:
        # logging.info(error)
        unseen_code_data = prep_unseen_code(error)
        slack_codes[error] = unseen_code_data
    translation_dict_icd10.update(slack_codes)

    ontology_errors = set()
    for prediction in predictions:
        for code in prediction["predicted_codes"]:
            for i in range(9, 4, -1):
                if code[:i] not in translation_dict_icd10:
                    ontology_errors.add(code[:i])
                if code[:3] not in translation_dict_icd10:
                    ontology_errors.add(code[:3])

    slack_codes = dict()
    for error in ontology_errors:
        unseen_code_data = prep_unseen_code(error)
        slack_codes[error] = unseen_code_data
    translation_dict_icd10.update(slack_codes)

    logging.info(f"Updated size of icd10 mapping: {len(translation_dict_icd10.keys())}")

    predictions_df = pd.DataFrame(predictions)

    del predictions, groundtruth
    gc.collect()

    code2index = {
        code: idx for idx, code in enumerate(list(translation_dict_icd10.keys()))
    }

    predictions_df["y_true"] = predictions_df.apply(
        lambda x: [code2index[code] for code in x["label_codes"]], axis=1
    )
    predictions_df["y_pred"] = predictions_df.apply(
        lambda x: [code2index[code] for code in x["predicted_codes"]], axis=1
    )

    mlb = MultiLabelBinarizer(
        classes=list(code2index.values()),
    )
    y_true_multihot = mlb.fit_transform(predictions_df.y_true)
    y_pred_multihot = mlb.fit_transform(predictions_df.y_pred)
    assert y_true_multihot.shape == y_pred_multihot.shape

    del predictions_df
    gc.collect()

    logging.info("MICRO REPORT ON LEAVES:")
    logging.info(multi_level_eval.report_micro(y_true_multihot, y_pred_multihot))
    logging.info()
    logging.info("MACRO REPORT ON LEAVES:")
    logging.info(multi_level_eval.report_macro(y_true_multihot, y_pred_multihot))
    logging.info()
    logging.info("HIERACHICAL MICRO REPORT ON LEAVES:")
    multi_level_eval.hierarchical_evaluation(
        y_true_multihot,
        y_pred_multihot,
        code2index,
        translation_dict_icd10,
        max_onto_layers=5,
    )


if __name__ == "__main__":
    main()
