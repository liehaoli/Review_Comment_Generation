import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "qlora-llama3.2-five-epochs-review", # "lora_model_test", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# alpaca_prompt = You MUST copy from above!

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

import json
# read the first 500 lines of the test data
test_data_file_path = "Review_Comment_Generation/msg-test-500-tuned.jsonl"
test_data = []
with open(test_data_file_path, 'r') as f:
    for i in range(500): # 500
        line = f.readline()
        data = json.loads(line)
        test_data.append(data)


# write all the output part of the test data to a file as the gold for comparison
gold_file_path = "msg-500-gold-5-epochs.txt"
with open(gold_file_path, 'w') as f:
    for data in test_data:
        # remove "New Code: " from the output
        value = data['output'].replace("New Code: ", "")
        f.write(value + "\n")

from datasets import load_dataset

test_file = "Review_Comment_Generation/msg-test-500-tuned.jsonl"
test_dataset = load_dataset("json", data_files= test_file)

for i in range(500):
    # inference example
# alpaca_prompt = Copied from above
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            # "Continue the fibonnaci sequence.", # instruction
            # "1, 1, 2, 3, 5, 8", # input
            # "", # output - leave this blank for generation!
            test_dataset['train']['instruction'][i],
            test_dataset['train']['input'][i],
            "",
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)
    tokenizer.batch_decode(outputs)
    # write the output to a file
    with open("msg-500-output-5-epochs.txt", 'a') as f:
        f.write(tokenizer.batch_decode(outputs)[0] + "\n")
        
        
# write a code that checks the output file and retrieves all lines following ### Response:\n
# write the output to a new file
output_file_path = "msg-500-pred-5-epochs.txt"
with open("msg-500-output-5-epochs.txt", 'r') as f:
    lines = f.readlines()
    with open(output_file_path, 'w') as f:
        for i in range(len(lines)):
            # if "New Code:" in lines[i]:
            #     # write the next line trimming <|end_of_text|> from the end
            #     f.write(lines[i][9:])
            if "### Response" in lines[i]:
                f.write(lines[i+1][:-16])
                f.write("\n")
                
print("Done Training and Saving")