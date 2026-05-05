'''test tokenizer.'''
from transformers import GPT2TokenizerFast

# load tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer")

tokenizer.add_special_tokens({
    "eos_token": "<EOS>",
    "additional_special_tokens": ["<EOS>"]
})

# test
test_text = "汉陵今日无抔土，惟独先生有钓台。<EOS>"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
tokens = [tokenizer.decode([i]) for i in encoded]

print(f"原始文本: {test_text}")
print(f"Token IDs: {encoded}")
print(f"decoded: {decoded}")
print(f"切分词块: {tokens}")
