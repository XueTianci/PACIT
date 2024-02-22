import os
import json
import copy
import glob
import tqdm
import pandas as pd
from transformers import HfArgumentParser,GPT2TokenizerFast,AutoTokenizer
from datasets import load_dataset
from transformers import set_seed
import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
from dataclasses import dataclass, field
# from nltk import sent_tokenize
from run_s2s import DataTrainingArguments
from ni_collator import DataCollatorForNI
@dataclass
class CustomizedArguments:
    output_dir: str = field(
        default="../data/text2text/", metadata={"help": "The directory for saving splits."}
    )
    set_seed: int = field(
        default= 42, metadata={"help": "The seed for saving splits."}
    )
    model_name: str = field(
        default= "", metadata={"help": "For the tokenizer"}
    )

parser = HfArgumentParser((DataTrainingArguments, CustomizedArguments))
args, customized_args = parser.parse_args_into_dataclasses()
set_seed(customized_args.set_seed)
raw_datasets = load_dataset(
    "../src/ni_dataset.py",
    data_dir=args.data_dir, 
    task_dir=args.task_dir, 
    max_num_instances_per_task=args.max_num_instances_per_task,
    max_num_instances_per_eval_task=args.max_num_instances_per_eval_task,
    cache_dir='./'
)

print('loading tokenizer')
tokenizer = AutoTokenizer.from_pretrained(customized_args.model_name,cache_dir = './')
print('finish loading')
data_collator = DataCollatorForNI(
    tokenizer,
    model=None,
    padding="max_length" if args.pad_to_max_length else "longest",
    max_source_length=args.max_source_length,
    max_target_length=args.max_target_length,
    add_task_definition=args.add_task_definition,
    num_pos_examples=args.num_pos_examples,
    num_neg_examples=args.num_neg_examples,
    add_explanation=args.add_explanation,
    text_only=True
)


for split in ["train",'test']:
    os.makedirs(os.path.join(customized_args.output_dir, "held_out",f"{split}"), exist_ok=True)
    with open(os.path.join(customized_args.output_dir, "held_out",f"{split}", "SuperNI-few-shot.jsonl"), "w") as fout2, \
        open(os.path.join(customized_args.output_dir, "held_out",f"{split}", "PACIT-few-shot-GT.jsonl"), "w") as fout4, \
        open(os.path.join(customized_args.output_dir, "held_out",f"{split}", "PACIT-few-shot-Flip.jsonl"), "w") as fout6, \
        open(os.path.join(customized_args.output_dir, "held_out",f"{split}", "PACIT-few-shot-Random.jsonl"), "w") as fout8:
        for example in tqdm.tqdm(raw_datasets[split]):
            sources,change_sources,change_flip_sources,random_sources = data_collator([example])
            temp={}
            temp["s2s_input"] = sources[0]["input"]
            temp["s2s_output"] = sources[0]["output"]
            fout2.write(json.dumps(temp) + "\n")

            temp={}
            temp["s2s_input"] = change_sources[0]["input"]
            temp["s2s_output"] = change_sources[0]["output"]
            fout4.write(json.dumps(temp) + "\n")

            temp={}
            temp["s2s_input"] = change_flip_sources[0]["input"]
            temp["s2s_output"] = change_flip_sources[0]["output"]
            fout6.write(json.dumps(temp) + "\n")

            temp={}
            temp["s2s_input"] = random_sources[0]["input"]
            temp["s2s_output"] = random_sources[0]["output"]
            fout8.write(json.dumps(temp) + "\n")
        
