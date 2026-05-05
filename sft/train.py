from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    TrainingArguments,
    Trainer
)

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

base_model_path = "../pre_training/final_model/"
sft_data_path = "../dataset/sft.csv"
output_sft_path = "./final_model"

print("正在加载预训练基座模型...")
tokenizer = GPT2TokenizerFast.from_pretrained(base_model_path, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(base_model_path)

def load_tokenized_dataset():
    print("正在加载 SFT 数据集...")

    def tokenize_function(examples):
        texts = []
        for ins, answer in zip(examples["instruction"], examples["answer"]):
            user_prompt = f"{ins}\n输出：{answer}"
            texts.append(user_prompt)

        return tokenizer(texts, truncation=True, max_length=256, padding=False)

    raw_dataset = load_dataset("csv", data_files=sft_data_path, split="train")
    return raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_dataset.column_names
    )

training_args = TrainingArguments(
    output_dir="checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    save_steps=200,
    logging_steps=100,
    learning_rate=3e-5,
    weight_decay=0.01,
    fp16=False,
    push_to_hub=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=load_tokenized_dataset(),
    data_collator=DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template="输出：")
)

print("启动SFT...")
trainer.train()

trainer.save_model(output_sft_path)
tokenizer.save_pretrained(output_sft_path)
print(f"✅ SFT完成！模型保存在: {output_sft_path}")
