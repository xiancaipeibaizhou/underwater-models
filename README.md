# Underwater Acoustic Target Recognition

基于深度学习的水下声学目标识别（Underwater Acoustic Target Recognition）模型库。本项目主要针对 **ShipsEar** 等水下音频数据集，提供从数据预处理、特征提取（如Log-Mel谱图）、模型训练到结果分析的完整代码流水线。项目核心框架基于 **PyTorch** 和 **PyTorch Lightning** 构建。

## 📂 项目结构

```text
underwater-model/
├── Datasets/                           # 📦 数据集与加载模块
│   ├── ShipsEar_Data_Preprocessing.py  # 原始音频切割与风浪去噪预处理
│   ├── ShipsEar_dataloader.py          # 标准帧级打榜划分 (Frame-level Random Split)
│   └── split_audit_report.json         # 数据集分布审计报告 (自动生成)
├── src/
│   └── models/                         
│       └── custom_model.py             # 🧠 核心模型骨架：Dynamic-HTAN 全套架构
├── Utils/                              # 🛠️ 核心组件库
│   ├── Feature_Extraction_Layer.py     # 前端 Log-Mel 时频特征提取
│   ├── LitModel.py                     # PyTorch Lightning 监督训练核心封装
│   └── LogMelFilterBank.py             # Mel 滤波器组数学实现
├── Demo_Parameters.py                  # ⚙️ 基础超参数配置中心
├── demo_light.py                       # 🚀 HTAN 模型主训练入口 (支持动态传参)
├── baseline_resnet.py                  # 📊 竞品对比实验基线 (纯视觉黑盒模型)
├── feature_similarity_analysis.py      # 📈 t-SNE 高维特征聚类可视化工具
├── plot_curves.py                      # 📉 混淆矩阵与训练曲线生成器
└── requirements.txt                    # 📦 Python 环境依赖包列表
```

## 🧠 核心模型架构与图构建流程 (HTAN)

本项目在 `src/models/custom_model.py` 中定义了核心创新模型 **HTAN** (包含物理启发频率图网络 `HarmonicFrequencyGCN`)。其图构建与特征流转的严谨过程如下：

1. **时频特征提取与图重构 (逐时间步的频率图)**：前端多尺度 CNN 提取时频特征。将每个时间步单独提取出来，对频率维度进行建图，张量变形为 `[B * T_out, F_out, C]`，节点代表单个时间片上的“频率 bin”。
2. **物理先验拓扑的初始化 (静态结构约束)**：通过声学物理规律构造频率先验拓扑矩阵 `A_prior`，包含：节点自连接、相邻频带连续性连接、以及单向判断后对称赋值的 2/3/4 倍谐波连接，最后进行行归一化。
3. **动态注意力打分与掩码融合 (先验决定拓扑，动态决定权重)**：前向传播时，计算动态注意力得分。`A_prior` 仅作为**结构掩码**，屏蔽不符合物理先验（非连续、非谐波）的边（将其注意力得分置为 `-1e9`），再对合法边做 Softmax 得到最终的动态概率转移矩阵。
4. **图卷积传递与频率池化 (特征更新)**：使用动态概率矩阵进行消息传递，经过线性层、ReLU、残差和 LayerNorm 更新节点特征。随后恢复维度，对频率维度执行 `mean` 和 `max` 双池化并拼接，生成时间序列特征，送入后续的双向 GRU 和时间注意力层（TemporalAttention）。

> **⚠️ 重要说明：当前默认训练链路与 AST 模型**
> 尽管 `custom_model.py` 定义了严密的 HTAN 频率图网络，但目前仓库默认的训练入口 (`demo_light.py` -> `Utils/Network_functions.py`) 实际上主要实例化并运行的是 **AST (Audio Spectrogram Transformer) 系列模型**（如 `ASTBase`, `ASTAdapter`, `ASTLoRA` 等）。如果您希望训练完整的 HTAN 模型，请在 `Utils/Network_functions.py` 中自行将 `initialize_model` 的逻辑分支指向 `HTAN`。

## 🛠️ 环境依赖

请确保您的计算机上已安装 Python 3.8 或更高版本。建议使用虚拟环境（如 Conda 或 venv）。

使用以下命令安装所需的 Python 依赖包：

```bash
git clone [https://github.com/your-username/underwater-model.git](https://github.com/your-username/underwater-model.git)
cd underwater-model
pip install -r requirements.txt
```

*主要依赖包括：`torch`, `pytorch-lightning`, `librosa`, `numpy`, `pandas`, `matplotlib`, `scikit-learn` 等。*

## 🚀 快速开始

### 1. 数据准备
1. 下载 **ShipsEar** 数据集，并将原始音频文件（`.wav`）放入 `shipsEar_AUDIOS/` 目录下。
2. 运行标签生成脚本，为音频文件生成对应的类别标签：
   ```bash
   cd shipsEar_AUDIOS
   python auto_label.py
   cd ..
   ```
3. 运行数据预处理脚本进行离线特征提取或数据清理（视具体需求而定）：
   ```bash
   python Datasets/ShipsEar_Data_Preprocessing.py
   ```

### 2. 参数配置
在开始训练之前，您可以通过修改 `Demo_Parameters.py` 文件来调整全局超参数：
* **数据参数**：采样率 (Sample Rate)、帧长 (Frame Length)、跳步 (Hop Length)
* **训练参数**：批次大小 (`batch_size`)、学习率 (`learning_rate`)、最大训练轮数 (`max_epochs`)
* **模型参数**：网络结构的具体维度和深度

### 3. 模型训练与评估
项目使用 PyTorch Lightning 封装了标准的训练、验证和测试流程。直接运行 `demo_light.py` 即可启动训练：

```bash
python demo_light.py
```
*说明：训练过程中会自动在终端输出进度条，并在每轮结束后验证准确率。最优模型权重会自动保存，训练结束后将在测试集上进行最终评估。*

### 4. 批量执行实验 (消融实验)
如果您需要测试不同的参数组合或运行消融实验（Ablation Study），可以使用提供的 Shell 脚本自动化运行：

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```
*实验的输出结果与评估指标将自动追加保存至 `htan_ablations_results.csv` 文件中，方便后续对比分析。*

## 📊 结果分析与可视化

训练或消融实验完成后，您可以使用内置的分析脚本对模型性能进行深度剖析：

* **绘制训练曲线**（Loss 和 Accuracy）：
  ```bash
  python plot_curves.py
  ```
* **特征相似度与特征空间分析**（如 t-SNE 降维可视化）：
  ```bash
  python feature_similarity_analysis.py
  ```

## 📌 数据集划分说明

为了保证实验的可重复性，数据集的划分被固定并记录在以下文件中：
* `shipsear_data_split.json`: 记录具体的划分配置和路径映射。
* `split_indices.txt`: 具体的样本索引。
* `split_audit_report.json`: 数据集划分的分布审计，确保训练/验证/测试集中各类别的均衡性。
