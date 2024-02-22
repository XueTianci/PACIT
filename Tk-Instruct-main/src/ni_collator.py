#collator
import logging
import random
import string
from transformers.data.data_collator import *

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForNI:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_task_definition: bool = True
    num_pos_examples: int = 0
    num_neg_examples: int = 0
    add_explanation: bool = False
    tk_instruct: bool = False
    text_only: bool=False
    

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []
        change_sources=[]
        change_flip_sources=[]
        random_sources=[]
        for instance in batch:
            if self.tk_instruct:
                all_valid_encodings = [
                    # instruction only
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 0, "num_neg_examples": 0, "add_explanation": False}, 
                    # example only
                    {"add_task_name": False, "add_task_definition": False, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False}, 
                    # instruction + pos examples
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False}, 
                    # instruction + pos examples + neg examples 
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 2, "add_explanation": False},
                    # instruction + pos (w. explanation) 
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": True}, 
                ]
                encoding_schema = random.choice(all_valid_encodings)
                add_task_name = encoding_schema["add_task_name"]
                add_task_definition = encoding_schema["add_task_definition"]
                num_pos_examples = encoding_schema["num_pos_examples"]
                num_neg_examples = encoding_schema["num_neg_examples"]
                add_explanation = encoding_schema["add_explanation"]
            else:
                add_task_name = self.add_task_name
                add_task_definition = self.add_task_definition
                num_pos_examples = self.num_pos_examples
                num_neg_examples = self.num_neg_examples
                add_explanation = self.add_explanation 
                
            task_input = ""
            # add the input first.
            task_input += "Now complete the following example -\n"
            task_input += f"Input: {instance['Instance']['input'].strip()}"
            if not task_input[-1] in string.punctuation:
                task_input += "."
            task_input += "\n"
            task_input += "Output: "

            task_name = ""
            if add_task_name:
                task_name += instance["Task"] + ". "

            definition = ""
            if add_task_definition:
                if isinstance(instance["Definition"], list):
                    definition = "Definition: " + instance["Definition"][0].strip() # TODO: should we use <Definition>?
                else:
                    definition = "Definition: " + instance["Definition"].strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                definition += "\n\n"

                
            examples=[]
            # try to add positive examples.
            pos_examples = []
            for idx, pos_example in enumerate(random.sample(instance["Positive Examples"][:],k=min(num_pos_examples,len(instance["Positive Examples"])))):
                pos_example_str = f" Positive Example {idx+1} -\n"
                pos_example_str += f" Input: {pos_example['input'].strip()}"
                if pos_example_str[-1] not in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                pos_example_str += f" Output: {pos_example['output'].strip()}"
                if pos_example_str[-1] not in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n" 
                if add_explanation and "explanation" in pos_example:
                    pos_example_str += f" Explanation: {pos_example['explanation'].strip()}"
                    if pos_example_str[-1] not in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n"
                pos_example_str += "\n"
                if len(self.tokenizer(definition + " ".join(pos_examples) + pos_example_str + task_input)["input_ids"]) <= self.max_source_length:
                    examples.append((pos_example,"pos"))
                    pos_examples.append(pos_example_str)
                else:
                    break

            # try to add negative examples.
            neg_examples = []
            for idx, neg_example in enumerate(random.sample(instance["Negative Examples"][:],k=min(num_neg_examples,len(instance["Negative Examples"])))):
                neg_example_str = f" Negative Example {idx+1} -\n"
                neg_example_str += f" Input: {neg_example['input'].strip()}"
                if neg_example_str[-1] not in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                neg_example_str += f" Output: {neg_example['output'].strip()}"
                if neg_example_str[-1] not in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                if add_explanation and "explanation" in neg_example:
                    neg_example_str += f" Explanation: {neg_example['explanation'].strip()}"
                    if neg_example_str[-1] not in string.punctuation:
                        neg_example_str += "."
                    neg_example_str += "\n"
                neg_example_str += "\n"
                if len(self.tokenizer(definition + " ".join(pos_examples) + " ".join(neg_examples) + neg_example_str + task_input)["input_ids"]) <= self.max_source_length:
                    examples.append((neg_example,"neg"))
                    neg_examples.append(neg_example_str)
                else:
                    break 
              
            source = task_name + definition + "".join(pos_examples) + "".join(neg_examples) + task_input
            if instance['Instance']["output"]:
            # Randomly select one reference if multiple are provided.
                label=random.choice(instance['Instance']["output"])
                try:
                    if label[-1] not in string.punctuation:
                        label +="."
                except:
                    print(label)
            temp={"input":source,"output":label}
            sources.append(temp)

            #TADIS TADIS-Ground_Truth format
            random.shuffle(examples)
            flag=[]
            pos_index=[]
            neg_index=[]
            example_str=""
            for idx, example in enumerate(examples):
                flag.append(example[1])
                if example[1] == "pos":
                    pos_index.append(f"example {idx+1}")
                else:
                    neg_index.append(f"example {idx+1}")
                example_str += f" Example {idx+1} -\n"
                example_str += f" Input: {example[0]['input'].strip()}"
                if example_str[-1] not in string.punctuation:
                    example_str += "."
                example_str += "\n"
                example_str += f" Output: {example[0]['output'].strip()}"
                if example_str[-1] not in string.punctuation:
                    example_str += "."
                example_str += "\n"
                if add_explanation and "explanation" in example:
                    example_str += f" Explanation: {example[0]['explanation'].strip()}"
                    if example_str[-1] not in string.punctuation:
                        example_str += "."
                    example_str += "\n"
                example_str += "\n"
            change_source = task_name + definition + example_str + task_input
            
            #TADIS-Ground_Truth output format
            output_instruction=""
            dic={"neg":"wrong","pos":"correct"}
            for idx, example in enumerate(examples):
                if idx == 0:
                    output_instruction += f" Example {idx+1} is {dic[example[1]]}, "
                elif idx == len(examples)-1:
                    output_instruction += f"and example {idx+1} is {dic[example[1]]}. "
                else:
                    output_instruction += f"example {idx+1} is {dic[example[1]]}, "
            if flag==[]:
                output_instruction +=""
            elif "pos" not in flag:
                output_instruction += f"I would avoid these wrong examples. The output: "
            elif "neg" not in flag:
                output_instruction += f"I would learn from these correct examples. The output: "
            else:
                output_instruction += f"I would avoid these wrong examples, and learn from these correct examples. The output: "

            change_sources.append({"input":change_source,"output":output_instruction + label})

            #TADIS-flip output format
            output_instruction=""
            dic={"neg":"correct","pos":"wrong"}
            for idx, example in enumerate(examples):
                if idx == 0:
                    output_instruction += f" Example {idx+1} is {dic[example[1]]}, "
                elif idx == len(examples)-1:
                    output_instruction += f"and example {idx+1} is {dic[example[1]]}. "
                else:
                    output_instruction += f"example {idx+1} is {dic[example[1]]}, "
            if flag==[]:
                output_instruction +=""
            elif "pos" not in flag:
                output_instruction += f"I would learn from these correct examples. The output: "
            elif "neg" not in flag:
                output_instruction += f"I would avoid these wrong examples. The output: "
            else:
                output_instruction += f"I would avoid these wrong examples, and learn from these correct examples. The output: "
            change_flip_sources.append({"input":change_source,"output":output_instruction + label})
            
            #TADIS-flip output format
            random_flag=[]
            output_instruction=""
            dic={"neg":"correct","pos":"wrong"}
            for idx, example in enumerate(examples):
                if idx == 0:
                    temp = random.choice(['pos','neg'])
                    random_flag.append(temp)
                    output_instruction += f" Example {idx+1} is {dic[temp]}, "
                elif idx == len(examples)-1:
                    temp = random.choice(['pos','neg'])
                    random_flag.append(temp)
                    output_instruction += f"and example {idx+1} is {dic[temp]}. "
                else:
                    temp = random.choice(['pos','neg'])
                    random_flag.append(temp)
                    output_instruction += f"example {idx+1} is {dic[temp]}, "
            if random_flag==[]:
                output_instruction +=""
            elif "pos" not in random_flag:
                output_instruction += f"I would learn from these correct examples. The output: "
            elif "neg" not in random_flag:
                output_instruction += f"I would avoid these wrong examples. The output: "
            else:
                output_instruction += f"I would avoid these wrong examples, and learn from these correct examples. The output: "
            random_sources.append({"input":change_source,"output":output_instruction + label})

        return sources,change_sources,change_flip_sources,random_sources
                                              
