#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import sys
from itertools import islice

import datasets
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
import argparse


import transformers
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    ElectraConfig,
    ElectraForSequenceClassification,
    ElectraTokenizer,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.activations import get_activation

# Import RobertaClassificationHead explicitly
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#ROOT_PATH = "data/representations-large"


logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout),  logging.FileHandler("logfile.log")],
)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


class CustomRobertaClassificationHead(RobertaClassificationHead):
    """Custom classification head with additional non-linear layers."""

    def __init__(self, config):
        super().__init__(config)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size // 2)  # Use hidden_size
        self.activation = nn.ReLU()
        self.out_proj = nn.Linear(config.hidden_size // 2, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.activation(x)
        x = self.out_proj(x)
        return x

class CustomRobertaForSequenceClassification(RobertaForSequenceClassification):
    """Custom Roberta model with a modified classification head."""

    def __init__(self, config):
        super().__init__(config)
        self.classifier = CustomRobertaClassificationHead(config)


class CustomBertClassificationHead(nn.Module):
    """Custom classification head with non-linear layers for MultiBERTs."""

    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.activation1 = nn.ReLU()
        self.dropout = nn.Dropout(config.classifier_dropout or config.hidden_dropout_prob)
        self.dense2 = nn.Linear(config.hidden_size // 2, config.hidden_size // 4)
        self.activation2 = nn.GELU()
        self.out_proj = nn.Linear(config.hidden_size // 4, config.num_labels)

    def forward(self, pooled_output):
        x = self.dense1(pooled_output)
        x = self.activation1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.activation2(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class CustomBertForSequenceClassification(BertForSequenceClassification):
    """Custom Bert model with a modified classification head for MultiBERTs."""

    def __init__(self, config):
        super().__init__(config)
        self.classifier = CustomBertClassificationHead(config)

class CustomElectraClassificationHead(nn.Module):
    """Custom head for sentence-level classification tasks with non-linear layers."""

    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.activation1 = get_activation("gelu")
        self.dropout1 = nn.Dropout(config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)

        self.dense2 = nn.Linear(config.hidden_size // 2, config.hidden_size // 4)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)

        self.out_proj = nn.Linear(config.hidden_size // 4, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.activation1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.activation2(x)
        x = self.out_proj(x)
        return x

class CustomElectraForSequenceClassification(ElectraForSequenceClassification):
    """Custom Electra model with a modified classification head for MultiBERTs."""

    def __init__(self, config):
        super().__init__(config)
        self.classifier = CustomElectraClassificationHead(config)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the task to train on: "
            + ", ".join(task_to_keys.keys())
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(
                    "Unknown task, you should pick one in "
                    + ",".join(task_to_keys.keys())
                )
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError(
                "Need either a GLUE task, a training/validation file or a dataset name."
            )
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    seed_x: bool = field(
        default=False,
        metadata={
            "help": "seeds"
            "with private models)."
        },
    )


def extract_hidden_states(model, dataloader, step_num, batch_size, max_seq_length):
    """
    Extract the hidden states of the model for a given dataloader.

    Parameters:
    model (torch.nn.Module): A PyTorch model to extract the hidden states from.
    train_dataloader (torch.utils.data.DataLoader): The data loader for the training set to extract the hidden states from.
    step_num (int): The number of steps to extract the hidden states for.
    batch_size (int): The size of the batches to extract the hidden states from.

    Returns:
    hidden_states (list): A list of hidden states for each sample in the dataloader. The shape of each hidden state is (layer_num, step_num*batch_size, hidden_size).
    """

    model.to(DEVICE)
    hidden_states = []
    all_labels = []
    logger.info(f"bs: {batch_size}, max_seq_length: {max_seq_length}")
    # for _ in tqdm(range(step_num)):
    for inputs in tqdm(islice(dataloader, 10)):  # just go through all steps
        model.train()
        # k = attention_mask, v = inputs["attention_mask"]
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(DEVICE)
        labels = inputs["labels"]
        inputs["labels"] = torch.zeros_like(inputs["labels"])
        all_labels.append(labels.cpu())  # Collect labels
        inputs["labels"] = torch.zeros_like(labels)  # Dummy labels for forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            logger.info(len(outputs["hidden_states"]))
            logger.info(outputs["hidden_states"][0].shape)
            last_size_hidden_states = outputs["hidden_states"][0].shape[2]
            hidden_states_tmp = (
                torch.cat(outputs["hidden_states"], dim=0)
                .reshape(
                    13, batch_size, max_seq_length, last_size_hidden_states
                )  # it should be batch_size, but raises error due to size mismatch..
                .detach()
                .mean(axis=2)
            )
            hidden_states.append(hidden_states_tmp)
    logger.info(f"Hidden states shape: {hidden_states[0].shape}")
    all_labels = torch.cat(all_labels, dim=0)
    hidden_states = torch.cat(
        hidden_states, dim=1
    )  # concatenate sample representations along batch dimension
    return hidden_states, all_labels

def save_tensor(tensor, root_path, model_name_path, task_name, split, seed=None):
    # Dynamically adjust file name based on whether a seed is used
    if seed is not None:
        file_path = os.path.join(root_path + model_name_path, f"task-{task_name}-{split}-seed-{seed}")
    else:
        file_path = os.path.join(root_path + model_name_path, f"task-{task_name}-{split}")
    torch.save(tensor, file_path)
    logger.info(f"{'seed: ' + str(seed) if seed is not None else ''} task: {task_name} {split} saved")


def process_and_save_hidden_states(model, dataloader, step_num, batch_size, max_seq_length, root_path, model_name_path, task_name, split, seed=None):
    hidden_states, labels = extract_hidden_states(
        model,
        dataloader,
        step_num=step_num,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )
    save_tensor(hidden_states, root_path, model_name_path, task_name, split, seed)
    save_tensor(labels, root_path, model_name_path, task_name, f"labels_{split}", seed)



def main():
    parser = argparse.ArgumentParser(description="Extract embeddings and labels from a model.")
    parser.add_argument("--root_path", type=str, default="data/nonlin/", help="Path to the directory where the extracted embeddings are stored.")
    parser.add_argument("--task_name", type=str, default=["cola", "mnli", "mrpc", "qnli", "qqp", "sst2","rte"], nargs="+", help='Name(s) of the task(s) (e.g., "cola", "mnli", "mrpc", "qnli", "qqp", "sst2")')
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels for the classification task.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached preprocessed datasets or not.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory where the embeddings and labels will be saved.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for the dataloaders.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computation (e.g., cpu or cuda).")
    parser.add_argument("--model_name_path", default="roberta-base", type=str, help="Path to the pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--model_type", type=str, default=["roberta", "electra","multiberts"],  help="Type of model to use (bert, roberta, electra).")
    parser.add_argument("--cache_dir", type=str, default="data/cache", help="Directory to store the pretrained models downloaded from huggingface.co.")
    parser.add_argument("--num_seeds", type=int, default=25, help="Number of seeds to use for generating embeddings.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device for evaluation.")

    args = parser.parse_args()

    ROOT_PATH = args.root_path
    DEVICE = args.device


    for model_type in args.model_type:
        # set model_parameters
        if model_type == "multiberts":
            seed_range = range(args.num_seeds)  # Loop over seeds for multiberts
            model_name_template = "google/multiberts-seed_{}"
            config_class = BertConfig
            tokenizer_class = BertTokenizer
            model_class = CustomBertForSequenceClassification
        elif model_type == "roberta":
            seed_range = [None]  # No seed-specific versions for roberta
            model_name_template = "roberta-base"
            config_class = RobertaConfig
            tokenizer_class = RobertaTokenizer
            model_class = CustomRobertaForSequenceClassification
        elif model_type == "electra":
            seed_range = [None]  # No seed-specific versions for electra
            model_name_template = "google/electra-small-discriminator"
            config_class = ElectraConfig
            tokenizer_class = ElectraTokenizer
            model_class = CustomElectraForSequenceClassification
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        

        for TASK_NAME in args.task_name:

            for SEED in seed_range:
                model_name_path = model_name_template.format(SEED) if "{}" in model_name_template else model_name_template


                # Centralized configuration logic
                config = config_class.from_pretrained(
                    model_name_path,
                    num_labels=args.num_labels,
                    finetuning_task=TASK_NAME,
                    cache_dir=args.cache_dir,
                )
                tokenizer = tokenizer_class.from_pretrained(
                    model_name_path,
                    cache_dir=args.cache_dir,
                    use_fast=True,
                )
                model = model_class.from_pretrained(
                    model_name_path,
                    config=config,
                    cache_dir=args.cache_dir,
                )

                logger.info(f"task name: {TASK_NAME}")
                model_name_path = model_type
                data_args = DataTrainingArguments(task_name=TASK_NAME, max_seq_length=128)
                training_args = TrainingArguments(output_dir="output_dir", per_device_train_batch_size=8)
                set_seed(training_args.seed)

                log_level = logging.INFO
                logger.setLevel(log_level)
                datasets.utils.logging.set_verbosity(log_level)
                transformers.utils.logging.set_verbosity(log_level)
                transformers.utils.logging.enable_default_handler()
                transformers.utils.logging.enable_explicit_format()

                logger.info("-------------------------------------")
                logger.info(f"Task: {data_args.task_name}, Seed: {SEED}")
                logger.info(f"Running on {DEVICE}")

                # Downloading and loading a dataset from the hub.
                raw_datasets = load_dataset("nyu-mll/glue", data_args.task_name)  # , cache_dir=model_args.cache_dir)
                sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

                # Labels
                is_regression = data_args.task_name == "stsb"
                if not is_regression:
                    label_list = raw_datasets["train"].features["label"].names
                    num_labels = len(label_list)
                else:
                    num_labels = 1
                
                
                model_folder = os.path.join(ROOT_PATH, model_name_path)
                os.makedirs(model_folder, exist_ok=True)
                logger.info(f"Created dir:{model_folder}")

                
                padding = "max_length"
                # Some models have set the order of the labels to use, so let's make sure we do use it.
                label_to_id = None
                if (
                    config.label2id != PretrainedConfig(num_labels=num_labels).label2id
                    and data_args.task_name is not None
                    and not is_regression
                ):
                    # Some have all caps in their config, some don't.
                    label_name_to_id = {k.lower(): v for k, v in config.label2id.items()}
                    if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
                        label_to_id = {
                            i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)
                        }
                    else:
                        logger.warning(
                            "Your model seems to have been trained with labels, but they don't match the dataset: ",
                            f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                            "\nIgnoring the model labels as a result.",
                        )
                elif data_args.task_name is None and not is_regression:
                    label_to_id = {v: i for i, v in enumerate(label_list)}

                if label_to_id is not None:
                    config.label2id = label_to_id
                    config.id2label = {id: label for label, id in config.label2id.items()}
                elif data_args.task_name is not None and not is_regression:
                    config.label2id = {l: i for i, l in enumerate(label_list)}
                    config.id2label = {id: label for label, id in config.label2id.items()}

                if data_args.max_seq_length > tokenizer.model_max_length:
                    logger.warning(
                        f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                    )
                max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

                def preprocess_function(examples):
                    # Tokenize the texts
                    args = (
                        (examples[sentence1_key],)
                        if sentence2_key is None
                        else (examples[sentence1_key], examples[sentence2_key])
                    )
                    result = tokenizer(
                        *args, padding=padding, max_length=max_seq_length, truncation=True
                    )

                    # Map labels to IDs (not necessary for GLUE tasks)
                    if label_to_id is not None and "label" in examples:
                        result["label"] = [
                            (label_to_id[l] if l != -1 else -1) for l in examples["label"]
                        ]
                    return result

                with training_args.main_process_first(desc="dataset map pre-processing"):
                    raw_datasets = raw_datasets.map(
                        preprocess_function,
                        batched=True,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on dataset",
                    )

                data_collator = default_data_collator
                train_dataset = raw_datasets["train"]
                if data_args.max_train_samples is not None:
                    train_dataset = train_dataset.select(range(data_args.max_train_samples))

                if (
                    "validation" not in raw_datasets
                    and "validation_matched" not in raw_datasets
                ):
                    raise ValueError("--do_eval requires a validation dataset")
                eval_dataset = raw_datasets[
                    "validation_matched" if data_args.task_name == "mnli" else "validation"
                ]
                if data_args.max_eval_samples is not None:
                    eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

                if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                    raise ValueError("--do_predict requires a test dataset")
                predict_dataset = raw_datasets[
                    "test_matched" if data_args.task_name == "mnli" else "test"
                ]

                train_dataset = train_dataset.remove_columns("idx")
                eval_dataset = eval_dataset.remove_columns("idx")
                predict_dataset = predict_dataset.remove_columns("idx")
                save_tensor(train_dataset, f"{ROOT_PATH}dataset/","", data_args.task_name,f"train", None )
                save_tensor(eval_dataset, f"{ROOT_PATH}dataset/","", data_args.task_name,f"eval", None )    
                save_tensor(predict_dataset, f"{ROOT_PATH}dataset/","", data_args.task_name, f"test", None )

                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=training_args.per_device_train_batch_size,
                    collate_fn=data_collator,
                )
                # train_dataloader_iter = enumerate(train_dataloader)
                val_dataloader = DataLoader(
                    eval_dataset,
                    batch_size=training_args.per_device_train_batch_size,
                    collate_fn=data_collator,
                )
                # val_dataloader_iter = enumerate(val_dataloader)
                test_dataloader = DataLoader(
                    predict_dataset,
                    batch_size=training_args.per_device_eval_batch_size,
                    collate_fn=data_collator,
                )
                # test_dataloader_iter = enumerate(test_dataloader)
                logger.info(f"seed: {SEED} task: {TASK_NAME}")
                model_args = ModelArguments(model_name_or_path=model_name_path, seed_x=SEED)

                logger.info(
                    "-------------------------------------------------------------------------"
                )
                logger.info(
                    f"train {len(train_dataloader)} val: {val_dataloader} test: {test_dataloader}"
                )
                # Process and save train hidden states and labels
                process_and_save_hidden_states(
                    model, train_dataloader, len(train_dataloader), train_dataloader.batch_size,
                    max_seq_length, ROOT_PATH, model_name_path, TASK_NAME, "train", SEED
                )

                # Process and save eval hidden states and labels
                process_and_save_hidden_states(
                    model, val_dataloader, len(val_dataloader), val_dataloader.batch_size,
                    max_seq_length, ROOT_PATH, model_name_path, TASK_NAME, "eval", SEED
                )

                # Process and save test hidden states and labels
                process_and_save_hidden_states(
                    model, test_dataloader, len(test_dataloader), test_dataloader.batch_size,
                    max_seq_length, ROOT_PATH, model_name_path, TASK_NAME, "test", SEED
                )



if __name__ == "__main__":
    main()
