'''Pretrain model.'''
import sys
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from datasets import load_dataset


class TeeLogger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

sys.stdout = TeeLogger("log.txt")

# load tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('tokenizer')
tokenizer.add_special_tokens({
    "eos_token": "<EOS>",
    "additional_special_tokens": ["<EOS>"]
})
tokenizer.pad_token = tokenizer.eos_token

# testing callback
class VisualProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0:
            print(f"\n\n--- 第 {state.global_step} 步试运行 ---")
            prompt = "落霞与孤鹜齐飞"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            model.eval()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,      # 生成 100 个字以内
                    do_sample=True,          # 采样模式，增加多样性
                    top_k=50,
                    top_p=0.95,
                    temperature=0.8,         # 越低越保守，越高越有创造力
                    pad_token_id=tokenizer.eos_token_id
                )
            model.train() # 切换回训练模式

            decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"生成的文本内容：\n{decoded_text}")

# ==========================================
# 1. Initialize model structure (Baby-LLM)
# ==========================================
config = GPT2Config(
    vocab_size=len(tokenizer),
    hidden_size=768,
    num_hidden_layers=8,
    num_attention_heads=12,
    max_position_embeddings=256
)
model = GPT2LMHeadModel(config)
model.to("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 模型结构已建立，词表大小: {len(tokenizer)}，总参数量: {model.num_parameters() / 1e6:.2f} M")

# ==========================================
# 2. data loader
# ==========================================
def load_tokenized_dataset():
    def tokenize_function(examples):
        texts = [f"{text}<EOS>" for text in examples["content"]]
        return tokenizer(texts, truncation=True, max_length=256)

    print("正在处理数据...")
    raw_dataset = load_dataset("csv", data_files='../dataset/poetry.csv', split="train")

    tokenized = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=raw_dataset.column_names
    )

    split_dataset = tokenized.train_test_split(test_size=0.05, seed=42)
    print(f"训练集: {len(split_dataset['train'])} 条, 验证集: {len(split_dataset['test'])} 条")
    return split_dataset["train"], split_dataset["test"]


# ==========================================
# 3. training
# ==========================================
if __name__ == "__main__":
    training_args = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=5,
        per_device_train_batch_size=128,
        save_steps=500,
        save_total_limit=4,
        logging_steps=500,
        learning_rate=3e-4,
        warmup_steps=1000,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        fp16=False,
        push_to_hub=False,
        report_to="none",
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        load_best_model_at_end=False
    )

    train_dataset, eval_dataset = load_tokenized_dataset()

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[VisualProgressCallback()]
    )

    print("🚀 开始预训练...")
    trainer.train()

    # warm start from checkpoint
    # trainer.train(resume_from_checkpoint="checkpoints/checkpoint-xxx")

    # 保存最终版本
    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")
    print("⭐ 模型训练完成。")
