# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2024/10/11 1:42
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2024/10/11 1:42

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMv1(nn.Module):
    def __init__(self, seq=6, n_fea=1, lstm_hidden_dim=16, lstm_layers=4):
        """
         LSTM model for time series prediction.

        Args:
            seq (int): Input sequence length.
            n_fea (int): Number of features (2 for occupancy and price).
            node (int): Number of nodes (e.g., spatial units or charging stations).
            lstm_hidden_dim (int): Number of hidden units in the LSTM layer.
            lstm_layers (int): Number of LSTM layers.
        """
        super(LSTMv1, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers

        # LSTM layer with input_size set to n_fea
        self.lstm = nn.LSTM(input_size=n_fea, hidden_size=self.lstm_hidden_dim, num_layers=self.lstm_layers,
                            batch_first=True)

        # Linear layer to map LSTM output to the desired output size
        self.fc = nn.Linear(seq*self.lstm_hidden_dim, 1)
        self.relu=nn.ReLU()

    def forward(self, input_data):
        # Combine occ and prc features into a single input tensor
        # x = torch.stack([occ, prc], dim=-1)  # shape: [batch_size, seq_len, node, n_fea]
        x = input_data[:, 0:6]
        # Reshape to (batch_size * node, seq_len, n_fea)
        x = torch.reshape(x, [x.shape[0], x.shape[1], 1])

        # Pass through LSTM layer
        lstm_out, _ = self.lstm(x)  # lstm_out shape: [batch_size * node, seq_len, lstm_hidden_dim]

        b, s, f = lstm_out.shape
        lstm_out = lstm_out.reshape(b, s * f)
        # Apply the fully connected layer
        x = self.fc(lstm_out)  # shape: [batch_size * node, 1]
        x = self.relu(x)
        return x


class PAG(nn.Module):
    def __init__(self, a_sparse, seq=12, kcnn=1, k=6, m=2, pred_type='all'):
        super(PAG, self).__init__()
        self.feature = seq
        self.seq = seq - kcnn + 1
        self.alpha = 0.5
        self.m = m
        self.a_sparse = a_sparse
        self.nodes = a_sparse.shape[0]

        # GAT
        self.conv2d = nn.Conv2d(1, 1, (kcnn, 1))  # input.shape = [batch, channel, width, height]
        self.gat_lyr = models.MultiHeadsGATLayer(a_sparse, self.seq, self.seq, 4, 0, 0.2)
        self.gcn = nn.Linear(in_features=self.seq, out_features=self.seq)

        # TPA
        self.lstm = nn.LSTM(m, m, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.seq - 1, out_features=k)
        self.fc2 = nn.Linear(in_features=k, out_features=m)
        self.fc3 = nn.Linear(in_features=k + m, out_features=1)
        self.decoder = nn.Linear(self.seq, 1)

        # Activation
        self.dropout = nn.Dropout(p=0.5)
        self.LeakyReLU = nn.LeakyReLU()

        #
        adj1 = copy.deepcopy(self.a_sparse.to_dense())
        adj2 = copy.deepcopy(self.a_sparse.to_dense())
        for i in range(self.nodes):
            adj1[i, i] = 0.000000001
            adj2[i, i] = 0
        degree = 1.0 / (torch.sum(adj1, dim=0))
        degree_matrix = torch.zeros((self.nodes, self.feature))
        for i in range(seq):
            degree_matrix[:, i] = degree
        self.degree_matrix = degree_matrix
        self.adj2 = adj2

    def forward(self, occ, prc):  # occ.shape = [batch,node, seq]
        b, n, s = occ.shape
        # data = torch.stack([occ, prc], dim=3).reshape(b*n, s, -1).unsqueeze(1)
        data = occ.unsqueeze(-1).reshape(b * n, s, -1).unsqueeze(1)
        data = self.conv2d(data)
        data = data.squeeze().reshape(b, n, -1)

        # first layer
        atts_mat = self.gat_lyr(data)  # attention matrix, dense(nodes, nodes)
        occ_conv1 = torch.matmul(atts_mat, data)  # (b, n, s)
        occ_conv1 = self.dropout(self.LeakyReLU(self.gcn(occ_conv1)))

        # second layer
        atts_mat2 = self.gat_lyr(occ_conv1)  # attention matrix, dense(nodes, nodes)
        occ_conv2 = torch.matmul(atts_mat2, occ_conv1)  # (b, n, s)
        occ_conv2 = self.dropout(self.LeakyReLU(self.gcn(occ_conv2)))

        occ_conv1 = (1 - self.alpha) * occ_conv1 + self.alpha * data
        occ_conv2 = (1 - self.alpha) * occ_conv2 + self.alpha * occ_conv1
        occ_conv1 = occ_conv1.view(b * n, self.seq)
        occ_conv2 = occ_conv2.view(b * n, self.seq)

        x = torch.stack([occ_conv1, occ_conv2], dim=2)  # best
        lstm_out, (_, _) = self.lstm(x)  # b*n, s, 2

        # TPA
        ht = lstm_out[:, -1, :]  # ht
        hw = lstm_out[:, :-1, :]  # from h(t-1) to h1
        hw = torch.transpose(hw, 1, 2)
        Hc = self.fc1(hw)
        Hn = self.fc2(Hc)
        ht = torch.unsqueeze(ht, dim=2)
        a = torch.bmm(Hn, ht)
        a = torch.sigmoid(a)
        a = torch.transpose(a, 1, 2)
        vt = torch.matmul(a, Hc)
        ht = torch.transpose(ht, 1, 2)
        hx = torch.cat((vt, ht), dim=2)
        y = self.fc3(hx)
        y = y.view(b, n)
        return y


# class GCN_v1(nn.Module):
#     def __init__(self, seq, n_fea, adj_dense):
#         super(GCN_v1, self).__init__()
#         self.nodes = adj_dense.shape[0]
#         self.seq_len = seq
#         self.num_features = n_fea
#
#         # Initialize GCN layers
#         self.gcn_l1 = GCNConv(in_channels=seq * self.num_features, out_channels=64)
#         self.gcn_l2 = GCNConv(in_channels=64, out_channels=32)
#
#         # Fully connected layers
#         self.fc1 = nn.Linear(32, 16)
#         self.fc2 = nn.Linear(16, 1)
#
#         # Preprocess the adjacency matrix to create edge_index and edge_weight
#         self.edge_index = self.create_edge_index(adj_dense)
#         self.edge_weight = adj_dense[adj_dense > 0]
#
#     def create_edge_index(self, adj_dense):
#         # Convert dense adjacency matrix to sparse edge index format
#         edge_index = torch.nonzero(adj_dense, as_tuple=False).t().contiguous()
#         return edge_index
#
#     def forward(self, occ, prc):
#         # Combine occ and prc features
#         # x = torch.stack([occ, prc], dim=-1)  # x.shape = (batch, node, seq, 2)
#         x = occ.unsqueeze(-1)
#         # Reshape x to (batch, node, seq * feature_num)
#         batch_size = x.size(0)
#         x = x.view(batch_size, self.nodes, self.seq_len * self.num_features)
#
#         # Reshape x to (batch * node, seq * feature_num)
#         x = x.view(batch_size * self.nodes, self.seq_len * self.num_features)
#
#         # Pass through GCN layers
#         x = F.relu(self.gcn_l1(x, self.edge_index, self.edge_weight))
#         x = F.relu(self.gcn_l2(x, self.edge_index, self.edge_weight))
#
#         # Reshape x back to (batch, node, feature)
#         x = x.view(batch_size, self.nodes, -1)
#
#         # Pass through fully connected layers
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#
#         # Squeeze to match the required output shape (batch, node)
#         x = torch.squeeze(x, dim=-1)
#         return x


class FCNN(nn.Module):
    def __init__(self, seq=6, n_fea=1,  hidden_dim=16, num_layers=3):
        """
        Fully Connected Neural Network (FCNN) for time series prediction.

        Args:
            seq (int): Sequence length (the length of the time window used for prediction).
            n_fea (int): Number of features (e.g., occupancy and prc features).
            node (int): Number of nodes (e.g., spatial units or charging stations).
            hidden_dim (int): Number of hidden units in each layer.
            num_layers (int): Number of fully connected layers.
        """
        super(FCNN, self).__init__()
        self.seq_len = seq
        self.num_features = n_fea
        self.hidden_dim = hidden_dim

        # Define the input layer
        self.input_layer = nn.Linear(self.seq_len * self.num_features, self.hidden_dim)

        # Define the hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        # Define the output layer
        self.output_layer = nn.Linear(self.hidden_dim, 1)

    def forward(self, input_data):
        x = input_data[:, 0:6]

        # Reshape x to (batch, node, seq * feature_num)

        x = torch.reshape(x, [1,x.shape[0], x.shape[1]])
        # x = x.contiguous().view(batch_size, self.nodes, self.seq_len * self.num_features)

        # Apply the input layer
        x = F.relu(self.input_layer(x))

        # Apply the hidden layers
        for layer in self.hidden_layers:
            x=layer(x)
            x = F.relu(x)

        # Apply the output layer
        x = self.output_layer(x)  # shape (batch, node, 1)

        # Squeeze to remove the last dimension, resulting in (batch, node)
        x = torch.reshape(x, [x.shape[1], 1])

        return x



class FGN(nn.Module):
    def __init__(self, pre_length=1, embed_size=16,
                 feature_size=0, seq_length=16, hidden_size=8, hard_thresholding_fraction=1, hidden_size_factor=1,
                 sparsity_threshold=0.01):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_frequency = 1
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.encoder = nn.Linear(6, 1)
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor))
        self.b3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_length, 8))
        self.fc = nn.Sequential(
            nn.Linear(8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )
        self.to('cuda:0')

    def tokenEmb(self, x):
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y

    # FourierGNN
    def fourierGC(self, x, B, N, L):
        o1_real = torch.zeros([B, (N * L) // 2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([B, (N * L) // 2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        o3_real = torch.zeros(x.shape, device=x.device)
        o3_imag = torch.zeros(x.shape, device=x.device)

        o1_real = F.relu(
            torch.einsum('bli,ii->bli', x.real, self.w1[0]) - \
            torch.einsum('bli,ii->bli', x.imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('bli,ii->bli', x.imag, self.w1[0]) + \
            torch.einsum('bli,ii->bli', x.real, self.w1[1]) + \
            self.b1[1]
        )

        # 1 layer
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)

        o2_real = F.relu(
            torch.einsum('bli,ii->bli', o1_real, self.w2[0]) - \
            torch.einsum('bli,ii->bli', o1_imag, self.w2[1]) + \
            self.b2[0]
        )

        o2_imag = F.relu(
            torch.einsum('bli,ii->bli', o1_imag, self.w2[0]) + \
            torch.einsum('bli,ii->bli', o1_real, self.w2[1]) + \
            self.b2[1]
        )

        # 2 layer
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = x + y

        o3_real = F.relu(
            torch.einsum('bli,ii->bli', o2_real, self.w3[0]) - \
            torch.einsum('bli,ii->bli', o2_imag, self.w3[1]) + \
            self.b3[0]
        )

        o3_imag = F.relu(
            torch.einsum('bli,ii->bli', o2_imag, self.w3[0]) + \
            torch.einsum('bli,ii->bli', o2_real, self.w3[1]) + \
            self.b3[1]
        )

        # 3 layer
        z = torch.stack([o3_real, o3_imag], dim=-1)
        z = F.softshrink(z, lambd=self.sparsity_threshold)
        z = z + x
        z = torch.view_as_complex(z)
        return z

    def forward(self, input_data):
        x = input_data[:, 0:6]
        x = self.encoder(x)
        x = torch.squeeze(x)

        B=1
        N=x.shape[0]
        L=1
        # B*N*L ==> B*NL
        x = x.reshape(B, -1)
        # embedding B*NL ==> B*NL*D
        x = self.tokenEmb(x)

        # FFT B*NL*D ==> B*NT/2*D
        x = torch.fft.rfft(x, dim=1, norm='ortho')

        x = x.reshape(B, (N * L) // 2 + 1, self.frequency_size)

        bias = x

        # FourierGNN
        x = self.fourierGC(x, B, N, L)

        x = x + bias

        x = x.reshape(B, (N * L) // 2 + 1, self.embed_size)

        # ifft
        x = torch.fft.irfft(x, n=N * L, dim=1, norm="ortho")

        x = x.reshape(B, N, L, self.embed_size)
        x = x.permute(0, 1, 3, 2)  # B, N, D, L

        x=x.squeeze(-1)

        # projection
        x = torch.matmul(x, self.embeddings_10)
        x = x.reshape(B, N, -1)
        x = self.fc(x)
        x = x.squeeze(0)
        return x

