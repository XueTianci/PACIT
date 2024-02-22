import json
import re
import os
import random

# re pattern
def_pattern = r'Definition: .*'
test_sample_pattern = r'Now complete the following example -\n[\s\S]*\nOutput: '
example_pattern = r'Example.*The output: '
thinking_pattern = r' I would.*The output: '
example_num = r'[0-9]'
wrong_correct_pattern = r'wrong|correct'

def read_data(path):
    datas = []
    with open(path, 'r', encoding = 'utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line)    
            datas.append(dic)
    return datas

def random_replace(match):
    choices = ['Positive Example', 'Negative Example']
    return random.choice(choices)

def swap_pos_neg(match):
    word = match.group(0)
    if word == "Positive Example":
        return "Negative Example"
    elif word == "Negative Example":
        return "Positive Example"

def SuperNI_zero_shot(datas,write_path):
    with open(write_path, 'w', encoding = 'utf-8') as f:
        for d in datas:
            def_matches = re.findall(def_pattern, d['s2s_input'])
            definition = def_matches[0]

            test_sample_matches = re.findall(test_sample_pattern, d['s2s_input'])
            test_sample = test_sample_matches[0]

            example = re.findall(example_pattern, d['s2s_output'])

            temp={}
            temp['s2s_input'] = definition + '\n\n' + test_sample
            if len(example) != 0:
                temp['s2s_output'] = d['s2s_output'].split(example[0])[-1]
            else:
                temp['s2s_output'] = d['s2s_output']
            f.write(json.dumps(temp) + "\n")

def SuperNI_few_shot_inference_Flip(datas,write_path):
    with open(write_path, 'w', encoding = 'utf-8') as f:
        for d in datas:
            temp = {}
            temp['s2s_input'] = re.sub(r'Positive Example|Negative Example', swap_pos_neg, d['s2s_input'])
            temp['s2s_output'] = d['s2s_output']
            f.write(json.dumps(temp) + "\n")

def SuperNI_few_shot_inference_Random(datas,write_path):
    with open(write_path, 'w', encoding = 'utf-8') as f:
        for d in datas:
            temp = {}
            temp['s2s_input'] = re.sub(r'Positive Example|Negative Example', random_replace, d['s2s_input'])
            temp['s2s_output'] = d['s2s_output']
            f.write(json.dumps(temp) + "\n")


def PACIT_zero_shot_inference_random(datas,write_path):
    with open(write_path, 'w', encoding = 'utf-8') as f:
        for d in datas:
            def_matches = re.findall(def_pattern, d['s2s_input'])
            definition = def_matches[0]

            test_sample_matches = re.findall(test_sample_pattern, d['s2s_input'])
            test_sample = test_sample_matches[0]

            example = re.findall(example_pattern, d['s2s_output'])


            change_list = [
                            'Example 1 is correct, I would learn from these correct examples. The output: ',
                            'Example 1 is wrong, I would avoid these wrong examples. The output: ',
                            'Example 1 is correct, and example 2 is wrong. I would avoid these wrong examples, and learn from these correct examples. The output: ',
                            'Example 1 is wrong, and example 2 is correct. I would avoid these wrong examples, and learn from these correct examples. The output: ',
                            ]

            temp={}
            temp['s2s_input'] = definition + '\n\n' + test_sample
            temp['s2s_decoder_input'] = random.choice(change_list)
            if len(example) != 0:
                temp['s2s_output'] = d['s2s_output'].split(example[0])[-1]
            else:
                temp['s2s_output'] = d['s2s_output']
            f.write(json.dumps(temp) + "\n")


def PACIT_few_shot_inference(datas,write_path):
    with open(write_path, 'w', encoding = 'utf-8') as f:
        for d in datas:
            
            example = re.findall(example_pattern, d['s2s_output'])

            temp={}
            if len(example) != 0:
                temp['s2s_input'] = d['s2s_input']
                temp['s2s_decoder_input'] = example[0]
                temp['s2s_output'] = d['s2s_output'].split(example[0])[-1]

            else:
                temp['s2s_input'] = d['s2s_input']
                temp['s2s_decoder_input']=""
                temp['s2s_output'] = d['s2s_output']
            f.write(json.dumps(temp) + "\n")


###################################### held_out #############################################
base_dir = "./data/seed-42/defintion_pos_1_neg_1_train_60_test_100/"
split = "test"
held_in_or_out = 'held_out'
os.makedirs(os.path.join(base_dir,held_in_or_out,split), exist_ok=True)

##################################### SuperNI #####################################
# ######## To SuperNI_zero_shot #########
datas = read_data(os.path.join(base_dir,held_in_or_out,split,'SuperNI-few-shot.jsonl'))
SuperNI_zero_shot(datas,os.path.join(base_dir,held_in_or_out,split,'SuperNI-zero-shot.jsonl'))

##################################### PACIT #####################################
# To the format that decodes with random classification label for zero-shot
datas = read_data(os.path.join(base_dir,held_in_or_out,split,'SuperNI-few-shot.jsonl'))
PACIT_zero_shot_inference_random(datas,os.path.join(base_dir,held_in_or_out,split,'PACIT-zero-shot-inference-Random.jsonl'))

# To the format that decodes with ground truth classification label
datas = read_data(os.path.join(base_dir,held_in_or_out,split,'PACIT-few-shot-GT.jsonl'))
PACIT_few_shot_inference(datas,os.path.join(base_dir,held_in_or_out,split,'PACIT-few-shot-inference-GT.jsonl'))

# To the format that decodes with Random classification label
datas = read_data(os.path.join(base_dir,held_in_or_out,split,'PACIT-few-shot-Random.jsonl'))
PACIT_few_shot_inference(datas,os.path.join(base_dir,held_in_or_out,split,'PACIT-few-shot-inference-Random.jsonl'))



split = "train"
os.makedirs(os.path.join(base_dir,held_in_or_out,split), exist_ok=True)
datas = read_data(os.path.join(base_dir,held_in_or_out,split,'SuperNI-few-shot.jsonl'))
SuperNI_zero_shot(datas,os.path.join(base_dir,held_in_or_out,split,'SuperNI-zero-shot.jsonl'))
