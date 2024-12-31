# =========================
# libraries
# =========================
from tqdm import tqdm
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
import torch
from transformers import (
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import roc_auc_score
from pathlib import Path
import time
import logging
from contextlib import contextmanager
import sys
import random
from torch.utils.data import Dataset
from trl import DataCollatorForCompletionOnlyLM
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# =========================
# constants
# =========================
HOME_PATH = Path(os.environ["HOME"]) # Change this to your home directory
DATA_DIR = HOME_PATH / Path("data/eedi-mining-misconceptions-in-mathematics")
OUTPUT_DIR = HOME_PATH / Path("results")
TRAIN_PATH = DATA_DIR / "train.csv"
TRAIN_GEN_PATH = OUTPUT_DIR / Path("train_gen/train_gen_8k.csv")
MISCONCEPTION_MAPPING_PATH = DATA_DIR / "misconception_mapping.csv"
FOLD_PATH = HOME_PATH / "eedi_fold.csv"
# =========================
# settings
# =========================
exp = "015"
exp_dir = OUTPUT_DIR / "exp" / f"ex{exp}"
model_dir = exp_dir / "model"

exp_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
logger_path = exp_dir / f"ex{exp}.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

candidate_path = OUTPUT_DIR / Path("exp239_val_pred_239_240_241.parquet")
    
seed = 0


@dataclass
class Config:
    output_dir: str = model_dir
    checkpoint: str = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
    max_length: int = 1024
    n_splits: int = 5
    optim_type: str = "adamw_torch_fused"
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    per_device_eval_batch_size: int = 8
    n_epochs: int = 2
    lr: float = 1e-4
    warmup_steps: int = 20
    lora_r: int = 16
    lora_alpha: float = lora_r * 2
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    fold = [0]
    candidate = 35
    nega_num = 20


config = Config()
candidate_gen_path = OUTPUT_DIR / Path(f"exp/ex010_infer_gen_fold_{config.fold[0]}/exp010_infer_gen_fold_{config.fold[0]}_val_pred.parquet")


LOGGER = logging.getLogger()
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


def setup_logger(out_file=None, stderr=True,
                 stderr_level=logging.INFO, file_level=logging.DEBUG):
    LOGGER.handlers = []
    LOGGER.setLevel(min(stderr_level, file_level))

    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stderr_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")
    return LOGGER


@ contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s')


setup_logger(out_file=logger_path)


# ======================================
# Model and Training Setup
# ======================================


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_model_and_tokenizer(config: Config):
    """
    Set up the model and tokenizer for training.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    tokenizer.padding_side = "left"

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        inference_mode=False,
        task_type=TaskType.SEQ_CLS,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = AutoModelForCausalLM.from_pretrained(
        Config.checkpoint,
        # device_map="cpu",
        pad_token_id=tokenizer.pad_token_id)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    return model, tokenizer


def prepare_datasets(
    train: pd.DataFrame, test: pd.DataFrame, tokenizer, config: Config, fold_idx: int
):
    """
    Prepare datasets for training and evaluation.
    """
    train_ds = Dataset.from_pandas(train[train.fold != fold_idx])
    val_ds = Dataset.from_pandas(train[(train.fold == fold_idx) & ()])
    test_ds = Dataset.from_pandas(test)

    encode = CustomTokenizer(tokenizer, max_length=config.max_length)
    encode_test = CustomTokenizer(
        tokenizer, max_length=config.max_length, is_train=False
    )

    train_ds = train_ds.map(encode, batched=True)
    val_ds = val_ds.map(encode, batched=True)
    test_ds = test_ds.map(encode_test, batched=True)

    return train_ds, val_ds, test_ds


class CustomTokenizer:
    """
    Custom tokenizer wrapper for batch processing.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, max_length: int, is_train: bool = True
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

    def __call__(self, batch: dict) -> dict:
        tokenized = self.tokenizer(
            batch["prompt"], max_length=self.max_length, truncation=True
        )
        if self.is_train:
            labels = batch["y"]
            return {**tokenized, "labels": labels}
        else:
            return {**tokenized}


def setup_trainer(model, tokenizer, train_ds, val_ds, config: Config, fold_idx: int):
    """
    Set up the Trainer for model training.
    """
    output_dir_fold = os.path.join(config.output_dir, f"fold{fold_idx}")

    training_args = TrainingArguments(
        output_dir=output_dir_fold,
        overwrite_output_dir=False,
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_checkpointing=True,
        save_total_limit=None,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='epoch',
        optim=config.optim_type,
        fp16=True,
        learning_rate=config.lr,
        warmup_steps=config.warmup_steps,
        metric_for_best_model="auc",
        greater_is_better=True,
        report_to="none",
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    return Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )


def compute_metrics(p):
    """
    Compute AUC score for model evaluation.
    """
    preds, labels = p
    preds = torch.sigmoid(torch.tensor(preds)).numpy()
    score = roc_auc_score(labels, preds)
    return {"auc": score}


prompt_template = """
You are a Mathematics teacher. Your task is to determine if the incorrect AnswerText aligns with the provided Misconception for the given Question about {ConstructName} ({SubjectName}).

Return "Yes" if the incorrect AnswerText reflects the Misconception, otherwise return "No". Do not provide explanations.

Answer "Yes" or "No" only.
- Question: {QuestionText}
- Correct Answer: {CorrectAnswerText}
- Incorrect Answer: {AnswerText}
- Misconception: {MisconceptionName}
"""


def make_prompt(row, tokenizer):
    """
    Generate a prompt based on the given row and template version.
    """
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt_template.format(
            ConstructName=row["ConstructName"],
            AnswerText=row["AnswerText"],
            CorrectAnswerText=row["CorrectAnswerText"],
            SubjectName=row["SubjectName"],
            QuestionText=row["QuestionText"],
            MisconceptionName=row["MisconceptionName"]
        )}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,

    )


class EediDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
    ):
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index) -> dict:
        row = self.df.iloc[index]

        inputs = {
            "input_ids": row["input_ids"],
        }

        return inputs


def preprocess_row(row: pd.Series, tokenizer: PreTrainedTokenizerBase) -> dict:
    item = tokenizer(row["prompt"], add_special_tokens=False, truncation=False)
    return item


def preprocess_df(df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase) -> pd.DataFrame:
    items = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        items.append(preprocess_row(row, tokenizer))

    df = pd.concat([
        df,
        pd.DataFrame(items)
    ], axis=1)
    return df


# ============================
# main
# ============================
train = pd.read_csv(TRAIN_PATH)
train_gen = pd.read_csv(TRAIN_GEN_PATH).sample(2500)
train_gen['QuestionId'] = np.arange(len(train_gen)) + 100000
train_gen['ConstructId'] = np.arange(len(train_gen)) + 100000
train_gen['SubjectId'] = np.arange(len(train_gen)) + 100000
train = pd.concat([train, train_gen]).reset_index(drop=True)
misconception = pd.read_csv(MISCONCEPTION_MAPPING_PATH)
candidate = pd.read_parquet(candidate_path)
candidate_gen = pd.read_parquet(candidate_gen_path)
candidate = pd.concat([candidate, candidate_gen]).reset_index(drop=True)
tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
df_fold = pd.read_csv(FOLD_PATH)

# ============================
# train preprocess
# ============================
print("train preprocess")
train_pivot = []
common_cols = ['QuestionId', 'ConstructId', 'ConstructName', 'SubjectId',
               'SubjectName', 'CorrectAnswer', 'QuestionText']
for i in ["A", "B", "C", "D"]:
    train_ = train.copy()
    train_ = train[common_cols + [f"Answer{i}Text", f"Misconception{i}Id"]]
    train_ = train_.rename({f"Answer{i}Text": "AnswerText",
                            f"Misconception{i}Id": "MisconceptionId"}, axis=1)
    train_["ans"] = i
    train_pivot.append(train_)
train_pivot = pd.concat(train_pivot).reset_index(drop=True)
train_pivot_correct_ans = train_pivot[
    train_pivot["CorrectAnswer"] == train_pivot["ans"]].reset_index(drop=True)
train_pivot_correct_ans = train_pivot_correct_ans[[
    "QuestionId", "AnswerText"]].reset_index(drop=True)
train_pivot_correct_ans.columns = ["QuestionId", "CorrectAnswerText"]
train_pivot = train_pivot[train_pivot["MisconceptionId"].notnull()].reset_index(
    drop=True)
train_pivot["MisconceptionId"] = train_pivot["MisconceptionId"].astype(int)


# ============================
# canidate preprocess
# ============================
print("candidate preprocess")
candidate_pivot = []
for mi, p, q, a in tqdm(zip(candidate["MisconceptionId"],
                       candidate["pred"],
                       candidate["QuestionId"],
                       candidate["ans"])):
    p = p.split(" ")
    p = [int(i) for i in p]
    p = p[:config.candidate]
    candidate_ = pd.DataFrame()
    candidate_["MisconceptionId"] = p
    candidate_["gt"] = mi
    candidate_["QuestionId"] = q
    candidate_["ans"] = a
    candidate_["rank"] = np.arange(len(candidate_))
    candidate_ = pl.DataFrame(candidate_)
    candidate_pivot.append(candidate_)

candidate_pivot = pl.concat(candidate_pivot).to_pandas().reset_index(drop=True)
candidate_pivot["candidate"] = 1

# ============================
# train preprocess2
# ============================
print("train preprocess2")
train_pivot_gt = train_pivot[["QuestionId",
                              "MisconceptionId", "ans"]].reset_index(drop=True)
train_pivot_gt["gt"] = train_pivot_gt["MisconceptionId"]
train_pivot_gt["candidate"] = 0
train_pivot_gt["rank"] = 0
train_pivot_gt = train_pivot_gt[candidate_pivot.columns]

# ============================
# add gt
# ============================
print("add gt")
candidate_pivot = pd.concat(
    [candidate_pivot, train_pivot_gt]).reset_index(drop=True)
candidate_pivot = candidate_pivot.drop_duplicates(
    subset=["QuestionId", "ans", "MisconceptionId"]).reset_index(drop=True)
candidate_pivot["y"] = candidate_pivot["gt"] == candidate_pivot["MisconceptionId"]
candidate_pivot["y"] = candidate_pivot["y"].astype(float)

# ============================
# merge correct answer
# ============================
print("merge correct answer")
# candidate_pivot = candidate_pivot.merge(
#     train_pivot_correct_ans, how="left", on="QuestionId")
candidate_pivot = pl.DataFrame(candidate_pivot)
train_pivot_correct_ans = pl.DataFrame(train_pivot_correct_ans)
candidate_pivot = candidate_pivot.join(train_pivot_correct_ans, on="QuestionId").to_pandas()

# prompt
print("prompt")
merge_cols = ["QuestionId", "ConstructName",
              "SubjectName", "QuestionText", "AnswerText", "ans"]
candidate_pivot = candidate_pivot.merge(
    train_pivot[merge_cols], how="left", on=["QuestionId", "ans"])

candidate_pivot = candidate_pivot.merge(
    misconception, how="left", on="MisconceptionId")
candidate_pivot["prompt"] = candidate_pivot.apply(
    lambda row: make_prompt(row, tokenizer), axis=1)

# fold
print("fold")
df_fold = df_fold.drop_duplicates(subset=["QuestionId"]).reset_index(drop=True)
candidate_pivot = candidate_pivot.merge(
    df_fold[["QuestionId", "fold"]], how="left", on="QuestionId")

# id
print("id")
candidate_pivot["id"] = candidate_pivot["QuestionId"].astype(str) + \
    "_" + candidate_pivot["ans"].astype(str)

candidate_pivot["y"] = candidate_pivot["y"].astype(int)
y_map = {1: "Yes", 0: "No"}
candidate_pivot["y_label"] = candidate_pivot["y"].map(y_map)

candidate_pivot["prompt"] = candidate_pivot["prompt"] + \
    "Answer:" + candidate_pivot["y_label"]
candidate_pivot = preprocess_df(candidate_pivot, tokenizer)
with timer("train"):
    set_seed(seed)
    for fold in Config.fold:
        tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
        tokenizer.padding_side = "left"
        data_collator = DataCollatorForCompletionOnlyLM(
            "Answer:", tokenizer=tokenizer)
        peft_config = LoraConfig(
            r=Config.lora_r,
            lora_alpha=Config.lora_alpha,
            lora_dropout=Config.lora_dropout,
            bias=Config.lora_bias,
            inference_mode=False,
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

        model = AutoModelForCausalLM.from_pretrained(
            Config.checkpoint,
            # device_map="cpu",
            pad_token_id=tokenizer.pad_token_id)
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        x_train = candidate_pivot[(candidate_pivot["fold"]
                                   != fold)].reset_index(drop=True)
        x_train_posi = x_train[x_train["y"] == 1].reset_index(drop=True)
        x_train_neg = x_train[(x_train["y"] == 0) & (x_train["MisconceptionId"].isin(
            x_train_posi["MisconceptionId"].values))].reset_index(drop=True)
        x_train_neg = x_train_neg.sample(frac=1, random_state=seed).reset_index(
            drop=True)
        x_train_neg["num"] = np.arange(len(x_train_neg))
        x_train_neg["nega_rank"] = x_train_neg.groupby(by="id")["num"].rank()
        x_train_neg = x_train_neg[x_train_neg["nega_rank"] < config.nega_num
                                  ].reset_index(
            drop=True)
        x_train = pd.concat([x_train_posi, x_train_neg]).reset_index(drop=True)

        output_dir_fold = os.path.join(Config.output_dir, f"fold{fold}")

        training_args = TrainingArguments(
            output_dir=output_dir_fold,
            overwrite_output_dir=False,
            num_train_epochs=config.n_epochs,
            per_device_train_batch_size=Config.per_device_train_batch_size,
            gradient_accumulation_steps=Config.gradient_accumulation_steps,
            per_device_eval_batch_size=Config.per_device_eval_batch_size,
            gradient_checkpointing=True,
            save_total_limit=None,
            # evaluation_strategy='epoch',
            save_strategy='epoch',
            logging_strategy='epoch',
            optim=config.optim_type,
            fp16=True,
            learning_rate=Config.lr,
            warmup_steps=Config.warmup_steps,
            report_to="none",
            gradient_checkpointing_kwargs={"use_reentrant": False},
            seed=seed,
        )

        trainer = Trainer(
            args=training_args,
            model=model,
            tokenizer=tokenizer,
            train_dataset=EediDataset(x_train),
            data_collator=data_collator,
        )
        trainer_output = trainer.train()