import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from peft import PeftModel

base_model_path = "../pre_training/final_model/"
lora_model_path = "./final_model"

print(f"加载基座模型: {base_model_path}")
tokenizer = GPT2TokenizerFast.from_pretrained(base_model_path, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(base_model_path, local_files_only=True)

print(f"加载 LoRA 权重: {lora_model_path}")
model = PeftModel.from_pretrained(model, lora_model_path)
model = model.merge_and_unload()

model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

print(f"模型加载完成，参数量: {model.num_parameters() / 1e6:.2f} M")

def poem_chat():
    print("\n--- LoRA SFT模型 ---")
    print("输入诗词指令，输入 'quit' 退出。")

    while True:
        try:
            user_input = input("\n输入：")
            if user_input.lower() in ("quit", "exit", "q"):
                break
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
