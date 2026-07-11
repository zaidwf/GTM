# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

GTM-Transformer (Google Trends Multimodal Transformer) 是一个多模态时尚产品销量预测模型。该项目是论文 [*Well Googled is Half Done: Multimodal Forecasting of New Fashion Product Sales with Image-based Google Trends*](https://arxiv.org/abs/2109.09824) 的官方 PyTorch 实现。

核心思路：对新上市的时尚产品进行零样本（zero-shot）销量预测，融合三种模态的信息：
1. **图像** — 产品图片（通过预训练 ResNet50 提取视觉特征）
2. **文本** — 产品类别、颜色、面料等属性（通过预训练 BERT 提取语义特征）
3. **时间序列** — Google Trends 搜索趋势数据（通过 Transformer Encoder 编码）

## 项目结构

```
GTM-Transformer/
├── train.py                          # 训练入口脚本
├── forecast.py                       # 推理/评估入口脚本
├── requirements.txt                  # Python 依赖（版本已固定）
├── models/
│   ├── GTM.py                        # GTM 模型（基于 Transformer Decoder）+ 所有子模块
│   └── FCN.py                        # FCN 基线模型（全连接网络解码器）+ 与 GTM.py 共享的子模块副本
├── utils/
│   └── data_multitrends.py           # 数据集类 ZeroShotDataset：数据加载与预处理
├── dataset/                          # 需要自行下载 VISUELLE 数据集放入此目录
├── ckpt/                             # 模型检查点保存目录
├── results/                          # 推理结果保存目录
└── log/                              # 训练日志目录
```

## 各文件作用详解

### `train.py` — 训练脚本
- 解析命令行参数，加载训练/测试 CSV 数据、类别/颜色/面料标签编码、Google Trends 数据
- 通过 `ZeroShotDataset` 构建 DataLoader（训练时 batch_size=128，验证时 batch_size=1）
- 根据 `--model_type` 选择 GTM 或 FCN 模型实例化
- 使用 PyTorch Lightning 的 Trainer 进行训练，每个 epoch 验证一次
- 默认使用 Wandb 记录日志，支持 TensorBoard 作为替代
- 自动保存验证 MAE 最低的模型检查点

### `forecast.py` — 推理/评估脚本
- 加载训练好的检查点，在测试集上运行推理
- 计算 MAE 和 WAPE 两种评估指标（原始尺度 + 反归一化后尺度）
- 结果（预测值、真实值、产品代码）保存到 `results/` 目录

### `models/GTM.py` — GTM 模型架构（核心文件）

包含以下模块类：

| 类名 | 作用 |
|------|------|
| `PositionalEncoding` | Transformer 位置编码（正弦/余弦） |
| `TimeDistributed` | 时间维度复用层，将任意模块沿时间轴应用到每个时间步（仿 Keras TimeDistributed） |
| `FusionNetwork` | 静态特征融合网络：将图像特征、文本特征、时间虚拟变量拼接后通过 MLP 融合 |
| `GTrendEmbedder` | Google Trends 编码器：用 TransformerEncoder 对多变量时间序列编码，支持分组掩码 |
| `TextEmbedder` | 文本编码器：将产品属性拼接成文本，通过 BERT 提取词向量并映射到嵌入空间 |
| `ImageEmbedder` | 图像编码器：冻结的 ResNet50（去掉最后两层），输出 2048 维空间特征图 |
| `DummyEmbedder` | 时间虚拟变量编码器：对日/周/月/年四个时间特征分别嵌入后融合 |
| `TransformerDecoderLayer` | 自定义 Transformer 解码层：交叉注意力（无自注意力），返回注意力权重 |
| `GTM` | 主模型（`pl.LightningModule`）：组合上述模块，包含完整训练/验证逻辑 |

**GTM 前向传播数据流：**
1. 图像 → ImageEmbedder (ResNet50) → 空间特征图
2. 文本属性 → TextEmbedder (BERT) → 文本嵌入
3. 时间虚拟变量 → DummyEmbedder → 时间嵌入
4. 以上三者 → FusionNetwork → 融合后的静态特征向量（作为 Decoder 的 query/tgt）
5. Google Trends → GTrendEmbedder (TransformerEncoder) → 趋势编码（作为 Decoder 的 memory）
6. Decoder（TransformerDecoder）交叉注意力 → 输出层 → 12 个月销量预测

支持两种解码模式：
- **非自回归（默认）**：一次性输出 12 个月的预测
- **自回归（`--autoregressive 1`）**：逐月生成，使用三角掩码

### `models/FCN.py` — FCN 基线模型
- 与 GTM 共享相同的编码器模块（`DummyEmbedder`、`ImageEmbedder`、`TextEmbedder`、`GTrendEmbedder`、`FusionNetwork`）
- 区别是解码器使用简单的全连接网络替代 Transformer Decoder
- 将融合的静态特征与展平的 Google Trends 编码拼接后通过 MLP 直接输出预测
- 用于消融实验，验证 Transformer Decoder 的有效性

### `utils/data_multitrends.py` — 数据预处理
- `ZeroShotDataset` 类封装所有数据预处理逻辑
- 对每个产品：提取发布日前 52 周的 Google Trends 数据（类别/颜色/面料三个维度），分别做 MinMax 归一化
- 读取产品图片并应用 Resize(256,256) + ImageNet 标准化
- 输出为一个 `TensorDataset`，包含：销量(12个月)、类别、颜色、面料、时间特征、Google Trends、图像

### `requirements.txt`
- 固定版本的依赖：PyTorch 1.8.2、PyTorch Lightning 1.5.0、Transformers 4.11.3、fairseq 0.10.2 等

## 常用命令

### 安装环境
```bash
python3 -m venv gtm_venv
source gtm_venv/bin/activate   # Windows: gtm_venv\Scripts\activate.bat
pip install -r requirements.txt
```

### 训练
```bash
# 使用 GTM 模型训练（默认）
python train.py --data_folder dataset

# 使用 FCN 基线模型训练
python train.py --model_type FCN --data_folder dataset

# 训练时关闭某些模态（消融实验）
python train.py --use_img 0 --use_text 0   # 仅用 Google Trends
python train.py --use_trends 0              # 仅用图像+文本
```

### 推理/评估
```bash
python forecast.py --data_folder dataset --ckpt_path ckpt/model.pth
```

### 关键参数说明
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_type` | GTM | 模型选择：GTM 或 FCN |
| `--output_dim` | 12 | 预测月数 |
| `--trend_len` | 52 | Google Trends 回溯周数 |
| `--num_trends` | 3 | 趋势维度（类别/颜色/面料） |
| `--embedding_dim` | 32 | 嵌入维度 |
| `--hidden_dim` | 64 | 隐藏层维度 |
| `--autoregressive` | 0 | 是否自回归解码（仅 GTM） |
| `--use_encoder_mask` | 1 | 是否使用分组编码掩码 |
| `--batch_size` | 128 | 批次大小 |
| `--epochs` | 200 | 训练轮数 |
| `--gpu_num` | 0 | GPU 编号 |

## 技术要点

- **零样本预测**：模型在全新产品上做预测，训练集和测试集的产品不重叠。因此预测完全依赖图像、文本属性和 Google Trends 等外部信号，而非历史销量。
- **归一化因子**：验证时 MAE 使用硬编码的 1065 作为反归一化缩放因子（训练集销量最大值，见 `GTM.py:321` 和 `FCN.py:268`）。
- **Google Trends 编码掩码**：`GTrendEmbedder._generate_encoder_mask` 生成分组掩码，按 `gcd(sequence_length, forecast_horizon)` 为间隔分组，使时间序列中相邻时间步可互相关注。
- **优化器**：使用 fairseq 的 Adafactor（自适应学习率），而非常见 Adam。
- **模型权重**：`FCN.py` 和 `GTM.py` 中存在重复的模块类定义（`PositionalEncoding`、`TimeDistributed`、`FusionNetwork`、`GTrendEmbedder`、`TextEmbedder`、`ImageEmbedder`、`DummyEmbedder`），修改时需注意保持一致。
- **数据集**：VISUELLE 数据集需从[此链接](https://forms.gle/cVGQAmxhHf7eRJ937)申请下载，解压到 `dataset/` 目录。该目录不在版本控制中（`.gitignore`）。
