import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =====================================================================
# 1. 构建先验知识图谱邻接矩阵 (Adjacency Matrix)
# =====================================================================
def build_prior_adjacency():
    """
    根据文献规则构建 11x11 的初始先验邻接矩阵。
    节点顺序假设为：
    0-2: 音调基元, 3-5: 音色基元, 6-7: 响度基元, 8-9: 周期性基元, 10: 深度语义基元
    """
    num_nodes = 11
    # 初始化所有边权重为 0.1 (不关联)
    A = torch.ones((num_nodes, num_nodes)) * 0.1
    
    # 强关联边权重设为 0.7
    edges_0_7 = [
        # 音调内部全连接 (三角结构)
        (0, 1), (0, 2), (1, 2),
        # 音色内部全连接 (三角结构)
        (3, 4), (3, 5), (4, 5),
        # 响度内部连接
        (6, 7),
        # 周期性内部连接
        (8, 9)
    ]
    
    # 设置对称权重
    for i, j in edges_0_7:
        A[i, j] = 0.7
        A[j, i] = 0.7
        
    # 深度基元(10)作为桥梁与所有其他节点连接
    for i in range(10):
        A[10, i] = 0.7
        A[i, 10] = 0.7
        
    # 对角线自身连接 (Self-loops) 设为 1.0 (文献中通常 A_tilde = A + I)
    A.fill_diagonal_(1.0)
    return A

# =====================================================================
# 2. 标准图卷积网络层 (GCN Layer)
# =====================================================================
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # x: [Batch, Nodes, Features] (B, 11, C)
        # adj: [Batch, Nodes, Nodes] (B, 11, 11) 或者 [11, 11]
        
        # 归一化邻接矩阵 D^{-1/2} A D^{-1/2}
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(x.size(0), -1, -1)
            
        rowsum = adj.sum(dim=-1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        
        # 构造对角矩阵 (Batch wise)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj_normalized = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

        # H^(l+1) = A_norm * H^(l) * W
        support = torch.matmul(x, self.weight)
        output = torch.bmm(adj_normalized, support) + self.bias
        return F.relu(output)

# =====================================================================
# 3. 基元关联性判别网络 (用于动态更新边权重)
# =====================================================================
class EdgeUpdateNetwork(nn.Module):
    def __init__(self, feature_dim=128):
        super(EdgeUpdateNetwork, self).__init__()
        # 文献中提到通过 4 层网络将 128 维降到 1 维 (128->64->32->16->1)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # 输出 0~1 的新权重
        )

    def forward(self, x_nodes):
        """
        x_nodes: 图的节点特征矩阵 [Batch, 11, 128]
        根据文献，输入为两节点差的绝对值: |Xi - Xj|
        """
        B, N, C = x_nodes.shape
        # 扩展维度计算所有节点对的差异
        x_i = x_nodes.unsqueeze(2).expand(B, N, N, C)
        x_j = x_nodes.unsqueeze(1).expand(B, N, N, C)
        
        # 绝对差值特征: [Batch, 11, 11, 128]
        diff_features = torch.abs(x_i - x_j)
        
        # 计算新的边权重: [Batch, 11, 11, 1] -> [Batch, 11, 11]
        new_adj = self.net(diff_features).squeeze(-1)
        
        # 保证对角线（自连接）依然为 1.0
        mask = torch.eye(N, device=x_nodes.device).unsqueeze(0).bool()
        new_adj = new_adj.masked_fill(mask, 1.0)
        
        return new_adj

# =====================================================================
# 4. GraphLSTM 单元 (用 GCN 替代 LSTM 的全连接)
# =====================================================================
class GraphLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes=11):
        super(GraphLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 为输入门、遗忘门、输出门和细胞状态构建 GCN (取代传统的 Linear)
        self.gcn_i = GraphConvLayer(input_dim + hidden_dim, hidden_dim)
        self.gcn_f = GraphConvLayer(input_dim + hidden_dim, hidden_dim)
        self.gcn_o = GraphConvLayer(input_dim + hidden_dim, hidden_dim)
        self.gcn_c = GraphConvLayer(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, adj, hx, cx):
        # x: [Batch, Nodes, Input_Dim]
        # hx, cx: [Batch, Nodes, Hidden_Dim]
        
        # 在特征维度上拼接 x 和 hx (对应公式中的拼接操作)
        combined = torch.cat([x, hx], dim=-1) # [Batch, Nodes, Input_Dim + Hidden_Dim]
        
        # 分别经过 GCN
        i_t = torch.sigmoid(self.gcn_i(combined, adj))
        f_t = torch.sigmoid(self.gcn_f(combined, adj))
        o_t = torch.sigmoid(self.gcn_o(combined, adj))
        c_tilde = torch.tanh(self.gcn_c(combined, adj))
        
        # 细胞状态与隐藏状态更新 (对应 LSTM 门控机制)
        c_next = f_t * cx + i_t * c_tilde
        h_next = o_t * torch.tanh(c_next)
        
        return h_next, c_next

# =====================================================================
# 5. 核心：知识更新立体语义网络 (完整模型组合)
# =====================================================================
class KnowledgeUpdateStereoSemanticNet(nn.Module):
    def __init__(self, num_classes=5, feature_dim=128, hidden_dim=64):
        super(KnowledgeUpdateStereoSemanticNet, self).__init__()
        self.num_nodes = 11
        self.hidden_dim = hidden_dim
        
        # 注册静态先验知识图谱矩阵作为不参与梯度更新的 buffer
        self.register_buffer('prior_adj', build_prior_adjacency())
        
        # 1. 图卷积提取层 (对应文献中的初始特征融合)
        self.gcn1 = GraphConvLayer(feature_dim, feature_dim)
        
        # 2. 基元关联性判别网络 (用于优化连接权重)
        self.edge_updater = EdgeUpdateNetwork(feature_dim)
        
        # 3. GraphLSTM 用于嵌入时序信息
        self.graph_lstm = GraphLSTMCell(feature_dim, hidden_dim, self.num_nodes)
        
        # 4. 全连接分类器
        # 将 11个节点的 hidden_dim 展平后进行分类
        self.classifier = nn.Sequential(
            nn.Linear(self.num_nodes * hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_seq):
        """
        x_seq: 经过特征提取并降维后的时序图特征
               维度: [Batch, Seq_Len, 11 (Nodes), 128 (Feature_Dim)]
        """
        B, SeqLen, N, C = x_seq.shape
        device = x_seq.device
        
        # 初始化 LSTM 隐藏状态
        h_t = torch.zeros(B, N, self.hidden_dim).to(device)
        c_t = torch.zeros(B, N, self.hidden_dim).to(device)
        
        # 针对序列中的每一个时间步进行处理
        for t in range(SeqLen):
            x_t = x_seq[:, t, :, :] # 当前时间步的节点特征 [B, 11, 128]
            
            # --- 步骤 A: 知识嵌入 (初步 GCN) ---
            # 使用静态先验矩阵进行一次基础特征融合
            x_gcn = self.gcn1(x_t, self.prior_adj)
            
            # --- 步骤 B: 知识更新 (边更新网络) ---
            # 利用融合后的特征，动态推断节点间的深层联系
            dynamic_adj = self.edge_updater(x_gcn)
            
            # --- 步骤 C: 时序融合 (GraphLSTM) ---
            # 将动态调整后的邻接矩阵和当前特征送入 GraphLSTM
            h_t, c_t = self.graph_lstm(x_t, dynamic_adj, h_t, c_t)
            
        # 取最后一个时间步的隐藏状态作为整个序列的表征
        # 展平所有节点的特征: [Batch, 11 * Hidden_Dim]
        graph_representation = h_t.view(B, -1)
        
        # 分类预测
        logits = self.classifier(graph_representation)
        return logits

# =====================================================================
# 测试与运行示例
# =====================================================================
if __name__ == "__main__":
    # 假设参数
    BATCH_SIZE = 8
    SEQ_LEN = 5        # 文献中将数据切为多段等长信号，体现时序性
    NUM_NODES = 11     # 11 个基元特征
    FEATURE_DIM = 128  # PCA降维后统一为 128 维
    NUM_CLASSES = 5    # 分类数目
    
    # 初始化网络
    model = KnowledgeUpdateStereoSemanticNet(num_classes=NUM_CLASSES)