import json

# 输入和输出文件路径
input_file_path = 'E:\自然语言理解\Exp\msg-test-500-tuned.jsonl'  # 原始文件路径
output_file_path = 'E:\自然语言理解\Exp\msg-test-500-tuned-chatgpt.jsonl'  # 输出文件路径

# 新的 instruction 内容
new_instruction = "Please provide formal code review for software developers in one sentence for following test case, implementing the \"output\" part. Just give the answer and make it simple, short and clear."

# 打开输入文件并读取
with open(input_file_path, 'r', encoding='utf-8') as infile:
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # 解析每行的JSON数据
            data = json.loads(line)
            
            # 将 'instruction' 键的值替换为新的内容
            data['instruction'] = new_instruction
            
            # 将 'output' 键的值替换为空字符串
            if 'output' in data:
                data['output'] = ''
            
            # 将修改后的数据写入输出文件
            outfile.write(json.dumps(data) + '\n')

print(f"处理完成，结果已保存至 {output_file_path}")
