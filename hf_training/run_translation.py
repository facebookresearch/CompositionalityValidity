#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
# Adapted from Orhan 2021, to replicate T5 on COGS

import logging
import os
import pdb
import sys
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    M2M100Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0.dev0")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_pretrained_weights: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use the pretrained model weights or random weights"}
    )
    model_parallel: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use model parallelism (experimental)"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    benchmark: str = field(default=None, metadata={"help": "Benchmark (COGS or SCAN)."})
    source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={"help": "Target language id for translation."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    gen_conditions_file: Optional[str] = field(default=None, metadata={"help": "Generalization conditions file (a jsonlines)."})
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacreblue) on "
            "a jsonlines file."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (sacreblue) on " "a jsonlines file."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        elif self.source_lang is None or self.target_lang is None:
            raise ValueError("Need to specify the source language and the target language.")

        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension == "json", "`train_file` should be a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension == "json", "`validation_file` should be a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is expected, e.g. with "
            "`--source_prefix 'translate English to German: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training arguments %s", training_args)
    logger.info("Model arguments %s", model_args)
    logger.info("Data arguments %s", data_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.max_length = data_args.max_target_length
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_args.use_pretrained_weights:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None
        )
    else:
        print('Using a model with random weights')
        model = AutoModelForSeq2SeqLM.from_config(config)

    if model_args.model_name_or_path == 'Rostlab/prot_t5_base_mt_uniref50':
        from torch.nn import Embedding, Linear
        print('This is the protein T5 model. Replacing the shared embedding layer with a randomized T5 size embedding.')
        model.shared = Embedding(32128, 768)
        model.encoder.embed_tokens = model.shared
        model.decoder.embed_tokens = model.shared
        model.lm_head = Linear(in_features=768, out_features=32128, bias=False)
    elif model_args.model_name_or_path == 'Rostlab/prot_t5_xl_bfd':
        from torch.nn import Embedding, Linear
        print('This is the protein T5 model. Replacing the shared embedding layer with a randomized T5 size embedding.')
        model.shared = Embedding(32128, 1024)
        model.encoder.embed_tokens = model.shared
        model.decoder.embed_tokens = model.shared
        model.lm_head = Linear(in_features=1024, out_features=32128, bias=False)
    else:
        print('Using the original embedding layer.')

    # print out model for inspection
    print(model)

    # optinally add model parallelism here
    if model_args.model_parallel:
        import torch
        print('Using model parallel on {:d} GPUs'.format(torch.cuda.device_count()))
        assert model_args.model_name_or_path in ['t5-11b', 't5-3b', 't5-large', 'google/mt5-xl', 'Rostlab/prot_t5_xl_bfd'], "Use model parallel only for sufficiently large models."
        assert torch.cuda.device_count() > 1, "Model parallelism requires more than 1 GPU."
        if torch.cuda.device_count() == 4:
            device_map = {
                0: [0, 1, 2], 
                1: [3, 4, 5, 6, 7, 8, 9], 
                2: [10, 11, 12, 13, 14, 15, 16], 
                3: [17, 18, 19, 20, 21, 22, 23]}
        elif torch.cuda.device_count() == 3:
            device_map = {
                0: [0, 1, 2, 3], 
                1: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
                2: [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
        elif torch.cuda.device_count() == 2:
            device_map = {
                0: [0, 1, 2, 3, 4, 5, 6, 7], 
                1: [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}

        model.parallelize(device_map)
    
    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer,
                                                                              MBartTokenizerFast,
                                                                              MBart50Tokenizer,
                                                                              MBart50TokenizerFast,
                                                                              M2M100Tokenizer)):
        assert (
            data_args.target_lang is not None and data_args.source_lang is not None
        ), "mBart requires --target_lang and --source_lang"
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer)):
        if data_args.source_lang is not None:
            tokenizer.src_lang = data_args.source_lang
        if data_args.target_lang is not None:
            tokenizer.tgt_lang = data_args.target_lang

    # Get the language codes for input/target.
    source_lang = data_args.source_lang.split("_")[0]

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warn(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        inputs = [ex["en"] for ex in examples["translation"]]
        targets = [ex["mentalese"] for ex in examples["translation"]]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        train_dataset = datasets["train"]
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metric
    metric = load_metric("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def sequence_accuracy(predictions: np.ndarray, labels: np.ndarray,
                          pad_token_id: int) -> np.ndarray:
        """Calculates the sequence accuracy for each sequence in the batch, between 0 and 1."""
        batch_size_preds, max_length_preds = predictions.shape
        batch_size_labels, max_length_labels = labels.shape
        assert batch_size_labels == batch_size_preds, "mismatch in batch size in predictions and labels in"\
                                                      " sequence_accuracy()"
        assert len(predictions.shape) == 2, "sequence accuracy only implemented for 2d predictions [bsz, seq_len]."
        max_length = max(max_length_preds, max_length_labels)
        predictions = np.pad(predictions, ((0, 0), (0, max_length - max_length_preds)))
        input_mask = (labels != np.zeros_like(labels) + pad_token_id).astype(np.int32)
        labels = np.pad(labels, ((0, 0), (0, max_length - max_length_labels)))
        correct_predictions = ((predictions == labels) * input_mask).sum(axis=1)
        length_per_example = input_mask.sum(axis=1)
        accuracy_per_sequence = correct_predictions / length_per_example
        return accuracy_per_sequence

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = preds[:, 1:]  # Get rid of <SOS> token.
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        accuracy_per_sequence = sequence_accuracy(preds, labels, pad_token_id=tokenizer.pad_token_id)
        exact_matches = (accuracy_per_sequence == 1.)
        exact_match_percentage = exact_matches.sum() / len(accuracy_per_sequence)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result["mean_sequence_accuracy"] = np.mean(accuracy_per_sequence)
        result["exact_match_percentage"] = exact_match_percentage
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
        )
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        metrics = test_results.metrics
        max_test_samples = data_args.max_test_samples if data_args.max_test_samples is not None else len(test_dataset)
        metrics["test_samples"] = min(max_test_samples, len(test_dataset))

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        # compute exact match accuracies by condition
        test_labels = test_results.label_ids
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            test_labels = np.where(test_labels != -100, test_labels, tokenizer.pad_token_id)

        test_predictions = test_results.predictions[:, 1:]
        if isinstance(test_predictions, tuple):
            test_predictions = test_predictions[0]

        print('Predictions:', test_predictions)
        print('Labels:', test_labels)

        accuracy_per_sequence = sequence_accuracy(test_predictions, test_labels, pad_token_id=tokenizer.pad_token_id)
        exact_matches = (accuracy_per_sequence == 1.)       

        if data_args.benchmark == 'COGS':
            if 'gen' in data_args.test_file:
                eval_split = 'gen'
                with open(data_args.gen_conditions_file, 'r') as f:
                    condition_list = json.load(f)

                exact_match_acc_by_condition = {}
                unique_conditions = list(set(condition_list))
                for cond in unique_conditions:
                    idx = [i for i, x in enumerate(condition_list) if x == cond]
                    exact_match_acc_by_condition[cond] = exact_matches[idx].sum() / len(idx)


                # overall accuracy
                exact_match_acc_by_condition["overall"] = exact_matches.sum() / len(exact_matches)

                logger.info("Exact match accuries by condition: %s", exact_match_acc_by_condition)
            else:
                eval_split = 'test'
                exact_match_acc = exact_matches.sum() / len(exact_matches)
                logger.info(f"Exact match accuracy for test: {exact_match_acc}")
                exact_match_acc_by_condition = exact_match_acc

            # save results
            save_filename = '{}accuracies_{}.json'.format(training_args.output_dir, eval_split)

            with open(save_filename, 'w') as f:
                json.dump(exact_match_acc_by_condition, f)

        elif data_args.benchmark == 'SCAN':
            # overall accuracy
            exact_match_acc = exact_matches.sum() / len(exact_matches)

            logger.info("Exact match accuracy: %f", exact_match_acc)

            # save results
            save_filename = '{}accuracy_{}.json'.format(training_args.output_dir, 'test')

            with open(save_filename, 'w') as f:
                json.dump(exact_match_acc, f)

        # generate predictions 
        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                test_preds = tokenizer.batch_decode(test_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                test_preds = [pred.strip() for pred in test_preds]
                # Strip the dataset name and split
                test_list = data_args.test_file.split('/')
                dataset_name = test_list[test_list.index('data') + 1]
                split = test_list[test_list.index('data') + 2][:len(test_list[test_list.index('data') + 2]) - 6]
                if 'gen' in data_args.test_file:
                    eval_split = 'gen'
                else:
                    eval_split = 'test'
                # TODO: Remove this line if made public
                os.environ["BASE_DIR"] = "/private/home/kaisersun/CompGenComparision"
                base_dir = os.environ["BASE_DIR"]

                output_test_preds_file = os.path.join(base_dir, 'preds/', dataset_name + '/', split + '/', 't5-base_s' + str(training_args.seed) + '_' + eval_split + '.txt')
                with open(output_test_preds_file, "w") as writer:
                    for line in test_preds:
                        writer.write(f"{line}\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()