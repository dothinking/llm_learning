import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

print("Loading pre-trained model and tokenizer...")

# load model
model_path = "./final_model"
model = GPT2LMHeadModel.from_pretrained(model_path)
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# load tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

def poem_chat():
    print("\n--- 预训练基座版模型 ---")
    print("提示：当前模型仅完成预训练，它会尝试『续写』诗词。输入 'quit' 退出。")
    
    while True:
        user_input = input("\n🧐 输入：")
        if user_input.lower() == 'quit':
            break

        # inference
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,      # 限制长度
                do_sample=True,          # 开启采样
                top_p=0.9,               # 核采样，过滤低概率词
                temperature=0.7,         # 控制随机性，0.7 比较稳健
                repetition_penalty=1.2,  # 重点！增加惩罚，减少“重复”现象
                pad_token_id=tokenizer.eos_token_id
            )

        # decode
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n输出：\n{result}")

if __name__ == "__main__":
    poem_chat()
