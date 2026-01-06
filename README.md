# Sentence-Similarity-Web

## 项目简介

Sentence-Similarity-Web 实现了句子余弦相似度的在线比对，其核心是使用**句子编码器**将文本转换为向量，再基于**余弦相似度（cosine similarity）**度量文本间的语义相关程度。

---

## 原理说明

### 1. 句子编码器（Sentence Encoder）

本项目采用 Huggingface 生态下的**预训练句子编码器**（如 SBERT、MiniLM 等）。  
其作用是将输入的自然语言句子 $S$ 映射为稠密的高维向量 $v$：

$$
S \rightarrow v = f(S)
$$

- $f(\cdot)$ 由深度神经网络建模，捕获语义信息。
- 常用如 Sentence-BERT（SBERT），MiniLM，DistilUSE 等编码器。

### 2. 余弦相似度（Cosine Similarity）

对于任意两个句子的向量表达 $v_1, v_2$，余弦相似度定义为：

$$
\text{CosineSim}(v_1, v_2) = \frac{v_1 \cdot v_2}{\|v_1\| \cdot \|v_2\|}
$$

- 取值范围 $[-1, 1]$，数值越大表示两个句子意义越接近。

核心工作流程如下：

1. 对输入的两个句子 $S_1, S_2$，用同一编码器得到向量 $v_1, v_2$
2. 计算 $\text{CosineSim}(v_1, v_2)$，作为句子相似度得分

---

## 安装与使用

1. 安装依赖：

   ```bash
   pip install flask torch transformers
   ```

2. 运行服务：

   ```bash
   python app.py
   ```

3. 打开浏览器访问 [http://127.0.0.1:5000](http://127.0.0.1:5000)

4. 输入两个句子，选择模型后，页面将输出二者的余弦相似度。

---

## 许可证

本项目基于 **GNU GPL v3（GNU通用公共许可证 第3版）** 开源：



**原作者：** [wangyifan349](https://github.com/wangyifan349)
