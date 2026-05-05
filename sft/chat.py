import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

print("加载模型...")
sft_model_path = "final_model"
tokenizer = GPT2TokenizerFast.from_pretrained(sft_model_path, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(sft_model_path, local_files_only=True)
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

def poem_chat():
    print("\n--- SFT模型 ---")
    print("输入诗词指令，输入 'quit' 退出。")

    while True:
        try:
            user_input = input("\n输入：")
            if user_input.lower() == 'quit': break
            prompt = f"{user_input}\n输出："
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    repetition_penalty=1.2,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                )

            full_result = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "输出：" in full_result:
                answer = full_result.split("输出：")[-1].strip()
            else:
                answer = full_result

            print(f"\n输出：{answer}")
        except Exception as e:
            print(f"错误: {str(e)}")

if __name__ == "__main__":
    poem_chat()
