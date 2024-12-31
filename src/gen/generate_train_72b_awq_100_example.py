import os, math, numpy as np
import os
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import re, gc
import torch
import random
from pathlib import Path
import argparse
import vllm

HOME_PATH = Path(os.environ["HOME"]) # Change this to your home directory
DATA_PATH = HOME_PATH / Path("data")
OUTPUT_PATH =HOME_PATH / Path("results")

def main(args):
    print("seed:", args.seed)
    TRAIN_GEN_FILENAME = f"train_gen_72b_awq_100_examples_seed_{args.seed}.csv"
    model_path = "Qwen/Qwen2.5-72B-Instruct-AWQ"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train = pd.read_csv(DATA_PATH / "eedi-mining-misconceptions-in-mathematics/train.csv")
    misconception = pd.read_csv(DATA_PATH / "eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")
    correct_answer_texts = []
    for idx, row in tqdm(train.iterrows()):
        correct_answer_text = row[f"Answer{row['CorrectAnswer']}Text"]
        correct_answer_texts.append(correct_answer_text)
    train["CorrectAnswerText"] = correct_answer_texts
    train_pivot = []
    common_cols = ['QuestionId', 'ConstructId', 'ConstructName', 'SubjectId',
                'SubjectName', 'CorrectAnswer', 'QuestionText', 'CorrectAnswerText']
    for i in ["A", "B", "C", "D"]:
        train_ = train.copy()
        train_ = train[common_cols + [f"Answer{i}Text", f"Misconception{i}Id"]]
        train_ = train_.rename({f"Answer{i}Text": "AnswerText",
                                f"Misconception{i}Id": "MisconceptionId"}, axis=1)
        train_["ans"] = i
        train_pivot.append(train_)
    train_pivot = pd.concat(train_pivot).reset_index(drop=True)
    train_pivot = train_pivot[train_pivot["MisconceptionId"].notnull()].reset_index(
        drop=True)
    train_pivot["MisconceptionId"] = train_pivot["MisconceptionId"].astype(int)
    misconceptions_gen = misconception[~misconception.MisconceptionId.isin(train_pivot.MisconceptionId.unique())]
    
    n_examples = 100
    llm_generate_prompt_template = """You are an expert in mathematics. 
    Refer to the examples below to create new problem with given misconception. 

    Misconception: {MisconceptionText}

    The output format shoud be below.

    ```
    ConstructName:  
    SubjectName: 
    Math problem: 
    Answer A text: 
    Answer B text: 
    Answer C text: 
    Answer D text: 
    Answer: 
    Incorrect answer: 
    ```

    The examples are below

    """

    llm_generate_prompt_template_2 = """Example {num}: 

    ConstructName: {ConstructName}
    SubjectName: {SubjectName}
    Math problem: {QuestionText}
    Answer A text: {AnswerAText}
    Answer B text: {AnswerBText}
    Answer C text: {AnswerCText}
    Answer D text: {AnswerDText}
    Answer: {CorrectAnswer}
    Incorrect answer: {ans}
    Misconception: {MisconceptionText} 

    """
                    
    def llm_generate_make_prompt(misconception_gen, rows):
        """
        Generate a prompt based on the given row and template version.
        """
        messages = [
        {"role": "user", "content": llm_generate_prompt_template.format(
                                MisconceptionText=misconception_gen,
                                )}
        ]
        for i in range(n_examples):
            messages[0]["content"] += llm_generate_prompt_template_2.format(
                                num = i,
                                ConstructName=rows[i]["ConstructName"],
                                SubjectName=rows[i]["SubjectName"],
                                QuestionText=rows[i]["QuestionText"],
                                AnswerAText=rows[i]["AnswerAText"],
                                AnswerBText=rows[i]["AnswerBText"],
                                AnswerCText=rows[i]["AnswerCText"],
                                AnswerDText=rows[i]["AnswerDText"],
                                CorrectAnswer=rows[i]["CorrectAnswer"],
                                ans=rows[i]["ans"],
                                MisconceptionText=rows[i]["MisconceptionText"],
                                )

        messages[0]["content"] += llm_generate_prompt_template.format(
                                MisconceptionText=misconception_gen,
                                )
            
        return messages

    base_columns = ["ConstructName", "SubjectName", "QuestionText", "AnswerAText", "AnswerBText", "AnswerCText", "AnswerDText", "CorrectAnswer"]
    def create_example(sample):
        ans = " "
        while ans not in "ABCD":
            _ans = random.choice("ABCD")
            if np.isnan(sample[f"Misconception{_ans}Id"]):
                continue
            ans =_ans
            misconception_text = misconception.iloc[sample[f"Misconception{_ans}Id"].astype(int)]["MisconceptionName"]
            res = {}
            for col in base_columns:
                res[col] = sample[col]
            res["ans"] = ans
            res["MisconceptionText"] = misconception_text
        return res
    
    prompt_text_list = []
    train = train.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    for misconception_gen in tqdm(misconceptions_gen["MisconceptionName"]):
        rows = []
        for i in range(n_examples):
            sample = train.sample(1).iloc[0]
            rows.append(create_example(sample))
        prompt = llm_generate_make_prompt(misconception_gen, rows)
        prompt_text_list.append(prompt)
    misconceptions_gen["prompt"] = prompt_text_list
    text_list = []

    for p in tqdm(misconceptions_gen["prompt"]):
        text = tokenizer.apply_chat_template(
            p,
            tokenize=False,
            add_generation_prompt=True
        )
        text_list.append(text)
    misconceptions_gen["text"] = text_list

    

    llm = vllm.LLM(
        model_path,
        tensor_parallel_size=1,
        quantization="awq",
        gpu_memory_utilization=0.90, 
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=20000,
        disable_log_stats=True
    )
    tokenizer = llm.get_tokenizer()

    responses = llm.generate(
        misconceptions_gen["text"].values,
        vllm.SamplingParams(
            n=1,  # Number of output sequences to return for each prompt.
            top_p=0.8,  # Float that controls the cumulative probability of the top tokens to consider.
            temperature=0,  # randomness of the sampling
            seed=777, # Seed for reprodicibility
            skip_special_tokens=False,  # Whether to skip special tokens in the output.
            max_tokens=4096,  # Maximum number of tokens to generate per output sequence.
        ),
        use_tqdm=True
    )

    responses = [x.outputs[0].text for x in responses]
    misconceptions_gen["fullLLMText"] = responses

    def extract_response(text):
        return ",".join(re.findall(r"<response>(.*?)</response>", text)).strip()

    misconceptions_gen["llmNewProblem"] = responses

    columns = [
    "ConstructName", 
    "SubjectName",
    "Math problem",
    "Answer A text",
    "Answer B text",
    "Answer C text",
    "Answer D text",
    "Answer",
    "Incorrect answer",
    ]
    gen_data_dict = {
        "ConstructName": [],
        "SubjectName": [],
        "CorrectAnswer": [],
        "QuestionText": [],
        "AnswerAText": [],
        "AnswerBText": [],
        "AnswerCText": [],
        "AnswerDText": [],
        "MisconceptionAId": [],
        "MisconceptionBId": [],
        "MisconceptionCId": [],
        "MisconceptionDId": [],
    }

    train_gen = pd.DataFrame()
    for idx, row in tqdm(misconceptions_gen.iterrows()):
        try:
            d = {}
            for i in range(len(columns)-1):
                d[columns[i]] = row["llmNewProblem"].split(columns[i]+": ")[1].split("\n"+columns[i+1]+": ")[0]
            d[columns[-1]] = row["llmNewProblem"].split(columns[-1]+": ")[1][0]
            d["MisconceptionId"] = row["MisconceptionId"]
            gen_data_dict["ConstructName"].append(d["ConstructName"])
            gen_data_dict["SubjectName"].append(d["SubjectName"])
            gen_data_dict["CorrectAnswer"].append(d["Answer"])
            gen_data_dict["QuestionText"].append(d["Math problem"])
            gen_data_dict["AnswerAText"].append(d["Answer A text"])
            gen_data_dict["AnswerBText"].append(d["Answer B text"])
            gen_data_dict["AnswerCText"].append(d["Answer C text"])
            gen_data_dict["AnswerDText"].append(d["Answer D text"])
            for s in "ABCD":
                if s == d["Incorrect answer"]:
                    gen_data_dict[f"Misconception{s}Id"].append(d["MisconceptionId"])
                else:
                    gen_data_dict[f"Misconception{s}Id"].append(np.nan)
        except:
            print("MisconceptionId:", row["MisconceptionId"])
            print(row["llmNewProblem"])
    for key in gen_data_dict:
        train_gen[key] = gen_data_dict[key]
    
    train_gen.to_csv(OUTPUT_PATH / "train_gen" / TRAIN_GEN_FILENAME, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(args)