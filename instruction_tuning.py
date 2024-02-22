import json
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

from torch.utils.data import DataLoader

import os
import random
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer,TrainerCallback
import torch
import wandb
from transformers import set_seed,AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,LlamaTokenizer,AutoModelForSeq2SeqLM,Seq2SeqTrainer,Seq2SeqTrainingArguments
import evaluate
import numpy as np
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="decapoda-research/llama-7b-hf")


@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    test_data_path: str = field(default=None, metadata={"help": "Path to the testing data."})
    quantify: bool = field(default=False, metadata={"help": "quantify or not"})
    set_seed: int = field(default=42, metadata={"help": "set seed"})
    project_name: str = field(default=None, metadata={"help": "wandb"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default="./")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    predict_with_generate: bool = field(default=True)
        
        
@dataclass
class Seq2SeqTrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    predict_with_generate: bool = field(default=True)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess_encoder_decoder(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    sources_tokenized, targets_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (sources, targets)]

    return dict(input_ids=sources_tokenized['input_ids'], labels=targets_tokenized['input_ids'])

def preprocess_decoder_only(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)

class SaveCallback(TrainerCallback):

    def __init__(self, model, epoch, output_dir,trainer):
        self.model = model
        self.epoch = epoch
        self.output_dir = output_dir
        self.trainer = trainer

            
    def on_epoch_end(self, args, state, control, **kwargs):
        self.model.save_pretrained(os.path.join(self.output_dir,f"epoch{self.epoch}_model"))
        self.trainer.evaluate()
        self.epoch += 1

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, model_args):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        
        list_data_dict=[]
        with open(data_path, 'r', encoding = 'utf-8') as f:
            for line in f.readlines():
                dic = json.loads(line)    
                list_data_dict.append(dic)
        #random dataset
        random.shuffle(list_data_dict)
        #debug use
        # list_data_dict=list_data_dict[:200]
        
        sources = []
        targets = []
        
        logging.warning("Formatting inputs...")

        for example in list_data_dict:
            source = example["s2s_input"]
            target = f"{example['s2s_output']}{tokenizer.eos_token}"
            if len(tokenizer.encode(source + target)) <= tokenizer.model_max_length:
                sources.append(source)
                targets.append(target)

        logging.warning("Tokenizing inputs... This may take some time...")
        
        if "t5" in model_args.model_name_or_path:
            data_dict = preprocess_encoder_decoder(sources, targets, tokenizer)
        elif "llama" in model_args.model_name_or_path:
            data_dict = preprocess_decoder_only(sources, targets, tokenizer)
        else:
            print('no format for this model')

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args,model_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.train_data_path,model_args=model_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def train():
    secret_value_0='your wandb'
    wandb.login(key=secret_value_0)
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(data_args.set_seed)
    setup_seed(data_args.set_seed)

    training_args.output_dir = os.path.join(training_args.output_dir,data_args.project_name,training_args.run_name)
    if training_args.local_rank == 0:
        wandb.init(
                    project=data_args.project_name,
                    name=training_args.run_name,
                    notes="reverse question and compare to find error answer",
                    )
# load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

# load model
    if data_args.quantify:
        pass
        
    else:
        if "t5" in model_args.model_name_or_path:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path,cache_dir=training_args.cache_dir,torch_dtype=torch.bfloat16)
        elif "llama" in model_args.model_name_or_path:
            model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,cache_dir=training_args.cache_dir,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2")
        else:
            print('no format for this model')
    
    
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )        
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args,model_args=model_args)
    
    #Tell Trainer not to attempt DataParallel
    # model.is_parallelizable = False
    # model.model_parallel = False

    trainer = Seq2SeqTrainer(model=model, tokenizer=tokenizer,args=training_args, **data_module)
    model.config.use_cache = False

    trainer.train()
    wandb.finish()
train()