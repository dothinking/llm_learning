'''LoRA fine-tune GPT2 for poetry SFT.'''
import sys
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


class TeeLogger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, "w", encoding="utf-8")
        import io
        self.terminal = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', errors='replace')

    def write(self, message):
        try:
            self.terminal.write(message)
            self.terminal.flush()
        except:
            pass
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        try:
            self.terminal.flush()
        except:
            pass
        self.log_file.flush()

    def close(self):
        self.log_file.close()


sys.stdout = TeeLogger("log.txt")


class VisualProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % 500 == 0:
            print(f"\n\n--- 第 {state.global_step} 步试运行 ---")
            prompt = "创作一首思乡诗\n输出："
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            model.eval()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=120,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id
                )
            model.train()
            decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"生成的文本内容：\n{decoded_text}")


model_dir = "../pre_training/final_model/"
data_path = "../dataset/sft.csv"

print(f"模型路径: {model_dir}")
print(f"数据路径: {data_path}")

# ==========================================
# 1. Load tokenizer
# ==========================================
tokenizer = GPT2TokenizerFast.from_pretrained(model_dir, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"词表大小: {len(tokenizer)}")

# ==========================================
# 2. Load model
# ==========================================
model = GPT2LMHeadModel.from_pretrained(model_dir)
model.config.use_cache = False
print(f"模型已加载，总参数量: {model.num_parameters() / 1e6:.2f} M")

# ==========================================
# 3. Apply LoRA
# ==========================================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj", "c_fc"],
    use_rslora=True
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==========================================
# 4. Data loader
# ==========================================
def load_tokenized_dataset():
    def format_sft(examples):
        texts = []
        for ins, answer in zip(
            examples["instruction"], examples["answer"]
        ):
            text = f"{ins}\n输出：{answer}"
            texts.append(text)
        return tokenizer(texts, truncation=True, max_length=256)

    print("正在处理数据...")
    raw_dataset = load_dataset("csv", data_files=data_path, split="train")

    tokenized = raw_dataset.map(
        format_sft,
        batched=True,
        num_proc=4,
        remove_columns=raw_dataset.column_names
    )

    split_dataset = tokenized.train_test_split(test_size=0.05, seed=42)
    print(f"训练集: {len(split_dataset['train'])} 条, 验证集: {len(split_dataset['test'])} 条")
    return split_dataset["train"], split_dataset["test"]


class DataCollatorForCompletionOnlyLM:
    def __init__(self, tokenizer, response_template="输出："):
        self.tokenizer = tokenizer
        self.response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)

    def __call__(self, features):
        batch = self.tokenizer.pad(features, return_tensors="pt")
        labels = batch["input_ids"].clone()

        for i in range(len(labels)):
            input_ids = labels[i].tolist()
            template_len = len(self.response_template_ids)
            for j in range(len(input_ids) - template_len + 1):
                if input_ids[j:j+template_len] == self.response_template_ids:
                    labels[i, :j+template_len] = -100
                    break

        batch["labels"] = labels
        return batch


# ==========================================
# 5. Training
# ==========================================
if __name__ == "__main__":
    training_args = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        save_steps=200,
        save_total_limit=3,
        logging_steps=100,
        learning_rate=5e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=False,
        push_to_hub=False,
        report_to="none",
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    train_dataset, eval_dataset = load_tokenized_dataset()

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            response_template="输出："
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[VisualProgressCallback()]
    )

    print("开始 LoRA 微调...")
    trainer.train()

    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")
    print("LoRA 微调完成，模型已保存至 ./final_model")
