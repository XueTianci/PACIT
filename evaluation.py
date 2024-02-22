import transformers
from transformers import set_seed,T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM,AutoTokenizer
import os
import numpy as np
import torch
import wandb
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
import json
from typing import Optional, Dict, Sequence
from torch.utils.data import Dataset
import logging
import random
import numpy
from vllm import LLM, SamplingParams

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def vllm_model_request_ori(model,input_prompt,tokenizer,args):
    sampling_params = SamplingParams(temperature=args.temperature,max_tokens=args.max_new_tokens,skip_special_tokens=True)

    outputs = model.generate([input_prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text
    return generated_text

def load_test_data(args, tokenizer):
    logging.warning("Loading data...")
    list_data_dict=[]
    with open(args.data_path, 'r', encoding = 'utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line)    
            list_data_dict.append(dic)
    return list_data_dict
    
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer,padding_side: str) -> Dict:
    """Tokenize a list of strings."""
    tokenizer.padding_side = padding_side
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            # truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    attention_mask = labels = [tokenized.attention_mask[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
        attention_mask = attention_mask,
    )


def preprocess_encoder_decoder(sources,decoder_input,targets,tokenizer,args):
    """Preprocess the data by tokenizing."""
    if args.decoder_input:
        sources_tokenized = _tokenize_fn(sources, tokenizer,'right')
        decoder_input_tokenized = _tokenize_fn(decoder_input, tokenizer,'left')
        targets_tokenized = _tokenize_fn(targets, tokenizer,'right')
        return dict(input_ids=sources_tokenized['input_ids'], attention_mask = sources_tokenized['attention_mask'], decoder_input_ids=decoder_input_tokenized['input_ids'], labels=targets_tokenized['input_ids'])
    else:

        sources_tokenized, targets_tokenized = [_tokenize_fn(strings, tokenizer,'right') for strings in (sources, targets)]
        return dict(input_ids=sources_tokenized['input_ids'], attention_mask = sources_tokenized['attention_mask'],labels=targets_tokenized['input_ids'])


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, args, tokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        self.args = args
        list_data_dict=[]
        with open(args.data_path, 'r', encoding = 'utf-8') as f:
            for line in f.readlines():
                dic = json.loads(line)    
                list_data_dict.append(dic)
        #random dataset
        # random.shuffle(list_data_dict)
        #debug use
        # list_data_dict=list_data_dict[:100]
        
        logging.warning("Formatting inputs...")
        sources = [
            example["s2s_input"]
            for example in list_data_dict
        ]
        targets = [f"{example['s2s_output']}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        
        if self.args.decoder_input:
            decoder_input = [example["s2s_decoder_input"] for example in list_data_dict]

            data_dict = preprocess_encoder_decoder(sources, decoder_input, targets, tokenizer,self.args)
            self.input_ids = data_dict["input_ids"]
            self.decoder_input_ids = data_dict['decoder_input_ids']
            self.labels = data_dict["labels"]
            self.attention_mask = data_dict["attention_mask"]
        else:
            data_dict = preprocess_encoder_decoder(sources, None, targets, tokenizer,self.args)
            self.input_ids = data_dict["input_ids"]
            self.labels = data_dict["labels"]
            self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.args.decoder_input:
            return dict(input_ids=self.input_ids[i], decoder_input_ids=self.decoder_input_ids[i], labels=self.labels[i], attention_mask = self.attention_mask[i])
        else:
            return dict(input_ids=self.input_ids[i], labels=self.labels[i], attention_mask = self.attention_mask[i])

class MyCollator(object):
    def __init__(self, tokenizer,args):
        self.tokenizer = tokenizer
        self.args = args
    def __call__(self, batch):
        collated = {k: [] for k in batch[0].keys()}
        for x in batch:
            for k, v in x.items():
                collated[k].append(v.view(-1))
        if self.args.decoder_input:
            temp_decoder = {}
            temp_decoder['input_ids'] = collated['decoder_input_ids']
            self.tokenizer.padding_side='left'
            temp_decoder = self.tokenizer.pad(temp_decoder,padding=True)
            collated['decoder_input_ids'] = temp_decoder['input_ids']

        temp ={}
        temp['input_ids'] = collated['input_ids']
        self.tokenizer.padding_side='right'
        temp = self.tokenizer.pad(temp,padding=True)


        collated['input_ids'] = temp['input_ids']
        collated['attention_mask'] = torch.nn.utils.rnn.pad_sequence(collated['attention_mask'], batch_first=True, padding_value=0)    
        collated['labels'] = torch.nn.utils.rnn.pad_sequence(collated['labels'], batch_first=True, padding_value=IGNORE_INDEX)    
        return collated

def t5_eval_generation(model,tokenizer,test_dataset,args,logger):
    print('start eval for generation')
    collate = MyCollator(tokenizer,args)
    dataloader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,  # TODO
            collate_fn=collate,
        )

    model.eval()
    final_generate_text = []
    final_target_text = []
    rouge = evaluate.load("rouge")
    for inputs in tqdm(dataloader, desc='Evaluating on Test Set'):
        with torch.no_grad():
            labels = inputs["labels"]
            inputs['input_ids'] = inputs['input_ids'].to(model.device)
            if args.decoder_input:
                inputs['decoder_input_ids'] = inputs['decoder_input_ids'].to(model.device)
                inputs['decoder_input_ids'] = model._shift_right(inputs['decoder_input_ids'])
                generated_ids = model.generate(
                                                input_ids=inputs['input_ids'],
                                                decoder_input_ids=inputs['decoder_input_ids'],
                                                # do_sample=True,
                                                # min_length=args.min_length,
                                                max_new_tokens=args.max_new_tokens,
                                                pad_token_id=tokenizer.pad_token_id,
                                                temperature=args.temperature,
                                                    )
            else:
                generated_ids = model.generate(
                                                input_ids=inputs['input_ids'],
                                                # do_sample=True,
                                                # min_length=args.min_length,
                                                max_new_tokens=args.max_new_tokens,
                                                pad_token_id=tokenizer.pad_token_id,
                                                temperature=args.temperature,
                                                    )
            prefix_text = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)

            decoded_preds_ori = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            decoded_preds_split =[d.split('The output: ')[-1] for d in decoded_preds_ori]

            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels_ori = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_labels_split =[d.split('The output: ')[-1] for d in decoded_labels_ori]

            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            with open(os.path.join(args.output_dir,'result.jsonl'), "a+",encoding="utf-8") as f1:
                for pr,po,ps,lo,ls in zip(prefix_text,decoded_preds_ori,decoded_preds_split,decoded_labels_ori,decoded_labels_split):
                    final_generate_text.append(ps)
                    final_target_text.append(ls)
                    temp={}
                    temp['prefix_text'] = pr
                    temp["decoded_preds_ori"]=po
                    temp["decoded_preds_split"]=ps
                    temp["decoded_labels_ori"]=lo
                    temp["decoded_labels_split"]=ls
                    f1.write(json.dumps(temp) + "\n")

    result = rouge.compute(predictions=final_generate_text, references=final_target_text, use_stemmer=True)

    prediction_lens = [len(gen.split(' ')) for gen in final_generate_text]
    result["gen_len"] = np.mean(prediction_lens)
    
    print({k: round(v * 100, 4) for k, v in result.items()})
    with open(os.path.join(args.output_dir,'result.jsonl'), "a+",encoding="utf-8") as f1:
        f1.write(json.dumps({k: round(v * 100, 4) for k, v in result.items()}) + '\n')

def eval_generation(model,tokenizer,test_dataset,args):

    final_generate_text = []
    final_target_text = []
    rouge = evaluate.load("rouge")
    for inputs in tqdm(test_dataset, desc='Evaluating on Test Set'):
            
        if args.decoder_input:
            preds_ori = vllm_model_request_ori(model,inputs["s2s_input"] + inputs["s2s_decoder_input"],tokenizer,args)
            prefix_text = [inputs["s2s_input"] + inputs["s2s_decoder_input"]]
        else:
            preds_ori = vllm_model_request_ori(model,inputs["s2s_input"],tokenizer,args)
            prefix_text = [inputs["s2s_input"]]

        preds_split =[d.split('The output: ')[-1] for d in [preds_ori]]

        labels_ori = inputs["s2s_output"]
        labels_split =[d.split('The output: ')[-1] for d in [labels_ori]]

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir,'result.jsonl'), "a+",encoding="utf-8") as f1:
            for pr,po,ps,lo,ls in zip(prefix_text,[preds_ori],preds_split,[labels_ori],labels_split):
                final_generate_text.append(ps)
                final_target_text.append(ls)
                temp={}
                temp['prefix_text'] = pr
                temp["decoded_preds_ori"]=po
                temp["decoded_preds_split"]=ps
                temp["decoded_labels_ori"]=lo
                temp["decoded_labels_split"]=ls
                f1.write(json.dumps(temp) + "\n")

    result = rouge.compute(predictions=final_generate_text, references=final_target_text, use_stemmer=True)
    prediction_lens = [len(gen.split(' ')) for gen in final_generate_text]
    result["gen_len"] = np.mean(prediction_lens)
    with open(os.path.join(args.output_dir,'result.jsonl'), "a+",encoding="utf-8") as f1:
        f1.write(json.dumps({k: round(v * 100, 4) for k, v in result.items()}) + '\n')

def arg_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument("--output_dir", type=str, default="./", help="")
    parser.add_argument("--project_name", type=str, default="./", help="")
    parser.add_argument("--run_name", type=str, default="./", help="")
    parser.add_argument("--model_name_or_path", type=str, default="./", help="")
    parser.add_argument("--data_path", type=str, default="./", help="")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="")
    parser.add_argument("--temperature", type=float, default=0, help="")
    parser.add_argument("--max_length", type=int, default=1024, help="")
    parser.add_argument("--decoder_input", type=bool, default=False, help="")
    parser.add_argument("--tokenizer", type=str, default="", help="")
    parser.add_argument("--eval_method", type=str, default="", help="")
    args = parser.parse_args()
    return args

def main():

    args = arg_parser()
    set_seed(args.random_seed)
    setup_seed(args.random_seed)
    args.output_dir = os.path.join(args.output_dir,args.project_name,args.run_name)

    if 't5' in args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
                    args.tokenizer,
                    cache_dir='./',
                    model_max_length=args.max_length,
                    padding_side="right",
                    use_fast=False,
                )
        model = AutoModelForSeq2SeqLM.from_pretrained(args.tokenizer,cache_dir='./',device_map='auto',torch_dtype=torch.bfloat16)
        model.load_state_dict(torch.load(args.model_name_or_path))

        test_dataset = SupervisedDataset(args,tokenizer)
        if args.eval_method == "generation":
            t5_eval_generation(model,tokenizer,test_dataset,args,wandb)
        elif args.eval_method == "attention":
            t5_eval_attention(model,tokenizer,test_dataset,args,wandb)

    else:
        tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer,
                cache_dir='./',
                model_max_length=args.max_length,
                padding_side="right",
                use_fast=False,
            )
        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
        tokenizer.add_special_tokens(special_tokens_dict)
        
        temp_path = os.path.dirname(args.model_name_or_path)
        
        model = LLM(model = temp_path, tokenizer = temp_path, tensor_parallel_size=1)
        test_dataset = load_test_data(args, tokenizer)
        eval_generation(model,tokenizer,test_dataset,args)

main()