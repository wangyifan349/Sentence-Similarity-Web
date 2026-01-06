from flask import Flask, request, render_template_string
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# 支持的句向量模型
MODEL_OPTIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": "英文-MiniLM",
    "sentence-transformers/distiluse-base-multilingual-cased-v2": "多语言-DistilUSE",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "多语言-PP-MiniLM-L12",
    "sentence-transformers/paraphrase-MiniLM-L6-v2": "英文-PP-MiniLM-L6",
}
MODEL_CACHE = {}
def get_model_and_tokenizer(name):
    if name not in MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        MODEL_CACHE[name] = (tokenizer, model, device)
    return MODEL_CACHE[name]
def encode(text, tokenizer, model, device):
    if isinstance(text, str):
        text = [text]
    batch = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=128)
    for k in batch:
        batch[k] = batch[k].to(device)
    with torch.no_grad():
        out = model(**batch)
        hiddens = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        emb = (hiddens * mask).sum(1) / mask.sum(1)
        emb = F.normalize(emb, p=2, dim=-1)
    return emb.cpu()
def sim(a, b):
    return torch.matmul(a, b.T).item()
app = Flask(__name__)
HTML = """
<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>句子相似度检测（预训练SBERT）</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <!-- Bootstrap 5 CDN -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
      background: linear-gradient(135deg, #ffeae6 0%, #fff9e5 100%);
      min-height: 100vh;
      color: #6a220e;
    }
    .main-box {
      background: rgba(255,243,222, 0.98);
      border-radius: 16px;
      box-shadow: 0 0 30px #f8d0b0a0;
      padding: 36px 38px 16px 38px;
      margin: 50px auto;
      max-width: 550px;
      border: 2px solid #ffc388;
    }
    .form-label, label, select, option { color: #883b1b; }
    .btn-main {
      background: linear-gradient(90deg, #ffcfcf 60%, #ffe082 100%);
      border: none;
      color: #5c1b00;
      font-weight: bold;
    }
    .btn-main:hover {
      background: linear-gradient(90deg, #ffe3e3 60%, #ffe7aa 100%);
    }
    .sim-score-res {
      background: #fff6e6;
      border: 1.5px solid #ffdab0;
      border-radius: 8px;
      padding: 12px;
      margin-top: 15px;
      color: #af6d09;
      font-size: 1.2rem;
      text-align: center;
    }
  </style>
</head>
<body>
<div class="main-box">
  <h3 class="mb-4" style="color:#bc614a;">句子相似度比对（多模型护眼）</h3>
  <form method="post">
    <div class="mb-3">
      <label class="form-label"><b>模型选择：</b></label>
      <select name="model" class="form-select">
        {% for model_id, label in models.items() %}
            <option value="{{model_id}}" {% if model_id==model %}selected{% endif %}>{{label}}</option>
        {% endfor %}
      </select>
    </div>
    <div class="mb-3">
      <label class="form-label"><b>句子1：</b></label>
      <input type="text" name="sent1" class="form-control" value="{{sent1 or ''}}" required autocomplete="off">
    </div>
    <div class="mb-3">
      <label class="form-label"><b>句子2：</b></label>
      <input type="text" name="sent2" class="form-control" value="{{sent2 or ''}}" required autocomplete="off">
    </div>
    <button type="submit" class="btn btn-main w-100 mb-2">比较相似度</button>
  </form>
  {% if sim is not none %}
    <div class="sim-score-res">
      <div><b>模型:</b> {{models[model]}}</div>
      <div><b>相似度得分：</b> <span style="font-size:1.25em">{{ "%.4f"|format(sim) }}</span></div>
    </div>
  {% endif %}
</div>
</body>
</html>
"""
@app.route("/", methods=["GET", "POST"])
def index():
    sim_score = None
    model = list(MODEL_OPTIONS)[0]
    sent1 = sent2 = ""
    if request.method == "POST":
        sent1 = request.form.get("sent1", "")
        sent2 = request.form.get("sent2", "")
        model = request.form.get("model", model)
        if sent1 and sent2 and model:
            tokenizer, mdl, device = get_model_and_tokenizer(model)
            v1 = encode(sent1, tokenizer, mdl, device)
            v2 = encode(sent2, tokenizer, mdl, device)
            sim_score = sim(v1, v2)
    return render_template_string(HTML, sim=sim_score, sent1=sent1, sent2=sent2, models=MODEL_OPTIONS, model=model)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
