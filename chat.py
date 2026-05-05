import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from peft import PeftModel
from typing import Optional

app = FastAPI()

BASE_DIR = "./"

pre_training_model = None
sft_model = None
lora_model = None
tokenizer = None


def load_models():
    global pre_training_model, sft_model, lora_model, tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading pre_training model...")
    pre_training_path = BASE_DIR + "pre_training/final_model"
    pre_training_model = GPT2LMHeadModel.from_pretrained(pre_training_path)
    pre_training_model.to(device)
    pre_training_model.eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(pre_training_path)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading sft model...")
    sft_path = BASE_DIR + "sft/final_model"
    sft_model = GPT2LMHeadModel.from_pretrained(sft_path, local_files_only=True)
    sft_model.to(device)
    sft_model.eval()

    print("Loading lora model...")
    base_model_path = BASE_DIR + "pre_training/final_model"
    lora_path = BASE_DIR + "lora/final_model"
    base_for_lora = GPT2LMHeadModel.from_pretrained(base_model_path, local_files_only=True)
    lora_model = PeftModel.from_pretrained(base_for_lora, lora_path)
    lora_model = lora_model.merge_and_unload()
    lora_model.to(device)
    lora_model.eval()

    print("All models loaded!")


class GenerateRequest(BaseModel):
    input_text: str
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    do_sample: bool = True


def generate_pre_training(text: str, params: GenerateRequest):
    inputs = tokenizer(text, return_tensors="pt").to(pre_training_model.device)
    with torch.no_grad():
        outputs = pre_training_model.generate(
            **inputs,
            max_new_tokens=params.max_new_tokens,
            do_sample=params.do_sample,
            top_p=params.top_p,
            temperature=params.temperature,
            repetition_penalty=params.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


def generate_sft(text: str, params: GenerateRequest):
    prompt = f"{text}\n输出："
    inputs = tokenizer(prompt, return_tensors="pt").to(sft_model.device)
    with torch.no_grad():
        outputs = sft_model.generate(
            **inputs,
            max_new_tokens=params.max_new_tokens,
            do_sample=params.do_sample,
            top_p=params.top_p,
            temperature=params.temperature,
            repetition_penalty=params.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "输出：" in full_result:
        answer = full_result.split("输出：")[-1].strip()
    else:
        answer = full_result
    return answer


def generate_lora(text: str, params: GenerateRequest):
    prompt = f"{text}\n输出："
    inputs = tokenizer(prompt, return_tensors="pt").to(lora_model.device)
    with torch.no_grad():
        outputs = lora_model.generate(
            **inputs,
            max_new_tokens=params.max_new_tokens,
            do_sample=params.do_sample,
            top_p=params.top_p,
            temperature=params.temperature,
            repetition_penalty=params.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "输出：" in full_result:
        answer = full_result.split("输出：")[-1].strip()
    else:
        answer = full_result
    return answer


@app.on_event("startup")
async def startup():
    load_models()


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    results = {}
    errors = {}

    try:
        results["pre_training"] = generate_pre_training(req.input_text, req)
    except Exception as e:
        errors["pre_training"] = str(e)

    try:
        results["sft"] = generate_sft(req.input_text, req)
    except Exception as e:
        errors["sft"] = str(e)

    try:
        results["lora"] = generate_lora(req.input_text, req)
    except Exception as e:
        errors["lora"] = str(e)

    return {"results": results, "errors": errors}


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_CONTENT


HTML_CONTENT = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GPT2 古诗生成模型测试</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f0f1a;
    color: #e0e0e0;
    min-height: 100vh;
}
.header {
    text-align: center;
    padding: 24px 16px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-bottom: 1px solid #2a2a4a;
}
.header h1 {
    font-size: 28px;
    background: linear-gradient(90deg, #e94560, #0f3460, #533483);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header p { color: #888; margin-top: 6px; font-size: 14px; }
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }
.input-section {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    border: 1px solid #2a2a4a;
}
.input-section label { display: block; margin-bottom: 8px; font-weight: 600; color: #aaa; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
textarea {
    width: 100%;
    height: 80px;
    background: #0f0f1a;
    border: 1px solid #3a3a5a;
    border-radius: 8px;
    color: #e0e0e0;
    padding: 12px;
    font-size: 16px;
    resize: vertical;
    font-family: inherit;
}
textarea:focus { outline: none; border-color: #e94560; }
.params-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-top: 16px;
}
.param-item label { margin-bottom: 4px; }
.param-item input[type="range"] {
    width: 100%;
    accent-color: #e94560;
}
.param-item .value-display {
    text-align: right;
    color: #e94560;
    font-size: 13px;
    font-weight: bold;
}
.param-item input[type="number"] {
    width: 100%;
    background: #0f0f1a;
    border: 1px solid #3a3a5a;
    border-radius: 6px;
    color: #e0e0e0;
    padding: 8px;
    font-size: 14px;
}
.param-item input[type="number"]:focus { outline: none; border-color: #e94560; }
.toggle-item {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 20px;
}
.toggle-item input[type="checkbox"] {
    width: 18px;
    height: 18px;
    accent-color: #e94560;
}
.toggle-item label { margin: 0; font-size: 14px; color: #ccc; }
.btn-generate {
    display: block;
    width: 100%;
    padding: 14px;
    margin-top: 16px;
    background: linear-gradient(135deg, #e94560, #533483);
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.2s;
}
.btn-generate:hover { opacity: 0.9; }
.btn-generate:disabled { opacity: 0.5; cursor: not-allowed; }
.results-section {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 16px;
}
@media (max-width: 900px) {
    .results-section { grid-template-columns: 1fr; }
}
.result-card {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #2a2a4a;
    transition: border-color 0.3s;
    display: flex;
    flex-direction: column;
}
.result-card:hover { border-color: #3a3a5a; }
.result-card .card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
    padding-bottom: 12px;
    border-bottom: 1px solid #2a2a4a;
}
.result-card .icon { font-size: 24px; }
.result-card .title { font-size: 16px; font-weight: 700; }
.result-card .subtitle { font-size: 11px; color: #666; }
.card-pre .title { color: #e94560; }
.card-sft .title { color: #0f3460; }
.card-sft .title { color: #4fc3f7; }
.card-lora .title { color: #533483; }
.card-lora .title { color: #ce93d8; }
.result-content {
    flex: 1;
    background: #0f0f1a;
    border-radius: 8px;
    padding: 14px;
    font-size: 15px;
    line-height: 1.8;
    white-space: pre-wrap;
    word-break: break-all;
    min-height: 100px;
    color: #ccc;
}
.result-content.error { color: #e94560; }
.result-content.loading { color: #666; }
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}
.status-dot.idle { background: #555; }
.status-dot.loading { background: #f0ad4e; animation: pulse 1s infinite; }
.status-dot.done { background: #5cb85c; }
.status-dot.error { background: #e94560; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
</style>
</head>
<body>
<div class="header">
    <h1>GPT2 古诗生成模型测试</h1>
    <p>预训练基座 vs SFT微调 vs LoRA微调</p>
</div>
<div class="container">
    <div class="input-section">
        <label>📝 输入提示词</label>
        <textarea id="inputText" placeholder="请输入提示词"></textarea>

        <div class="params-grid">
            <div class="param-item">
                <label>🌡️ Temperature</label>
                <input type="range" id="temperature" min="0.1" max="2.0" step="0.1" value="0.7">
                <div class="value-display" id="temperatureVal">0.7</div>
            </div>
            <div class="param-item">
                <label>🎯 Top P</label>
                <input type="range" id="topP" min="0.1" max="1.0" step="0.05" value="0.9">
                <div class="value-display" id="topPVal">0.9</div>
            </div>
            <div class="param-item">
                <label>🔄 Repetition Penalty</label>
                <input type="range" id="repPenalty" min="1.0" max="2.0" step="0.1" value="1.2">
                <div class="value-display" id="repPenaltyVal">1.2</div>
            </div>
            <div class="param-item">
                <label>📏 Max New Tokens</label>
                <input type="number" id="maxNewTokens" min="10" max="512" value="100">
            </div>
        </div>

        <div class="toggle-item">
            <input type="checkbox" id="doSample" checked>
            <label for="doSample">启用采样 (do_sample)</label>
        </div>

        <button class="btn-generate" id="btnGenerate" onclick="generate()">🚀 生成对比</button>
    </div>

    <div class="results-section">
        <div class="result-card card-pre">
            <div class="card-header">
                <span class="icon">🧱</span>
                <div>
                    <div class="title">预训练基座</div>
                    <div class="subtitle">Pre-training · 直接续写</div>
                </div>
                <span class="status-dot idle" id="dot-pre"></span>
            </div>
            <div class="result-content" id="result-pre">等待输入...</div>
        </div>
        <div class="result-card card-sft">
            <div class="card-header">
                <span class="icon">🎓</span>
                <div>
                    <div class="title">SFT 微调</div>
                    <div class="subtitle">Supervised Fine-Tuning · 指令跟随</div>
                </div>
                <span class="status-dot idle" id="dot-sft"></span>
            </div>
            <div class="result-content" id="result-sft">等待输入...</div>
        </div>
        <div class="result-card card-lora">
            <div class="card-header">
                <span class="icon">⚡</span>
                <div>
                    <div class="title">LoRA 微调</div>
                    <div class="subtitle">Low-Rank Adaptation · 参数高效 | 指令跟随</div>
                </div>
                <span class="status-dot idle" id="dot-lora"></span>
            </div>
            <div class="result-content" id="result-lora">等待输入...</div>
        </div>
    </div>
</div>

<script>
document.getElementById('temperature').addEventListener('input', function() {
    document.getElementById('temperatureVal').textContent = this.value;
});
document.getElementById('topP').addEventListener('input', function() {
    document.getElementById('topPVal').textContent = this.value;
});
document.getElementById('repPenalty').addEventListener('input', function() {
    document.getElementById('repPenaltyVal').textContent = this.value;
});

document.getElementById('inputText').addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'Enter') {
        generate();
    }
});

function formatPoetry(text) {
    var count = 0;
    return text.replace(/[，。！？；：、,!?;:\\.]/g, function(match) {
        count++;
        if (count % 2 === 0) {
            return match + '\\n';
        }
        return match;
    });
}

async function generate() {
    const inputText = document.getElementById('inputText').value.trim();
    if (!inputText) { alert('请输入文本'); return; }

    const btn = document.getElementById('btnGenerate');
    btn.disabled = true;
    btn.textContent = '⏳ 生成中...';

    const models = ['pre', 'sft', 'lora'];
    const modelKeys = ['pre_training', 'sft', 'lora'];
    models.forEach(m => {
        document.getElementById('result-' + m).textContent = '生成中...';
        document.getElementById('result-' + m).className = 'result-content loading';
        document.getElementById('dot-' + m).className = 'status-dot loading';
    });

    const params = {
        input_text: inputText,
        max_new_tokens: parseInt(document.getElementById('maxNewTokens').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        top_p: parseFloat(document.getElementById('topP').value),
        repetition_penalty: parseFloat(document.getElementById('repPenalty').value),
        do_sample: document.getElementById('doSample').checked
    };

    try {
        const resp = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        const data = await resp.json();

        models.forEach((m, i) => {
            const key = modelKeys[i];
            const el = document.getElementById('result-' + m);
            const dot = document.getElementById('dot-' + m);
            if (data.results && data.results[key]) {
                el.textContent = formatPoetry(data.results[key]);
                el.className = 'result-content';
                dot.className = 'status-dot done';
            } else if (data.errors && data.errors[key]) {
                el.textContent = '❌ 错误: ' + data.errors[key];
                el.className = 'result-content error';
                dot.className = 'status-dot error';
            } else {
                el.textContent = '无结果';
                el.className = 'result-content';
                dot.className = 'status-dot idle';
            }
        });
    } catch (e) {
        models.forEach(m => {
            document.getElementById('result-' + m).textContent = '❌ 请求失败: ' + e.message;
            document.getElementById('result-' + m).className = 'result-content error';
            document.getElementById('dot-' + m).className = 'status-dot error';
        });
    }

    btn.disabled = false;
    btn.textContent = '🚀 生成对比';
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
