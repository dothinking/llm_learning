'''Train tokenizer.'''
import csv
from tokenizers import ByteLevelBPETokenizer

# 1. prepare data: read poetry.csv and extract content
filename = "tokenizer/temp_corpus.txt"

# Try multiple encodings to handle potential encoding issues
encodings_to_try = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']
rows_processed = 0

for encoding in encodings_to_try:
    try:
        with open('../dataset/poetry.csv', 'r', encoding=encoding) as f_csv, \
             open(filename, "w", encoding="utf-8") as f_out:
            reader = csv.DictReader(f_csv)
            for row in reader:
                content = row.get("content", "").strip()
                if content:
                    f_out.write(f"{content}<EOS>\n")
                    rows_processed += 1
        print(f"✅ 成功使用 {encoding} 编码读取，处理了 {rows_processed} 行数据")
        break
    except UnicodeDecodeError:
        print(f"❌ {encoding} 编码失败，尝试下一个...")
        continue
    except Exception as e:
        print(f"❌ {encoding} 编码出现其他错误: {e}")
        continue
else:
    print("❌ 所有编码都无法解析文件，请检查文件编码")
    exit(1)

# 2. initialize BPE tokenizer
tokenizer = ByteLevelBPETokenizer()

# 3. train tokenizer
tokenizer.train(files=[filename],
                vocab_size=20000,     # 词表大小，小模型建议 10k-32k
                min_frequency=10,      # 至少出现两次的词才被收录
                show_progress=True,
                special_tokens=[
                    "<s>", "<pad>", "</s>", "<unk>", "<mask>", "<EOS>"  # 添加诗歌专用标记
                ])

# 4. save tokenizer
tokenizer.save_model("tokenizer")
print("✅ Done!")
