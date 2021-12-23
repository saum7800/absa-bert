import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAttModel(nn.Module):
    def __init__(self, embedding_dim=300, hidden_dim=128, lstm_layer=2, dropout=0.2):
        super(LSTMAttModel, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim
        self.lstm_text = nn.LSTM(
            input_size=self.embedding_dim, hidden_size=hidden_dim, num_layers=lstm_layer, bidirectional=True,
            batch_first=True
        )
        self.lstm_aspects = nn.LSTM(
            input_size=self.embedding_dim, hidden_size=hidden_dim, num_layers=lstm_layer, bidirectional=True,
            batch_first=True
        )

        self.dense = nn.Linear(2 * hidden_dim, 3)

    def forward(self, text_vecs, aspect_vecs, labels):
        text_vecs = self.dropout(text_vecs)  # batch_size(16) x text_len(125) x text_embed_dim(300)
        out_text, (h_n, c_n) = self.lstm_text(text_vecs)  # batch_size(16) x text_len(125) x text_hidden_dim(256)

        aspect_vecs = self.dropout(aspect_vecs)  # batch_size(16) x text_len(125) x text_embed_dim(300)
        out_aspect, (h_n, c_n) = self.lstm_aspects(aspect_vecs)  # batch_size(16) x text_len(125) x text_hidden_dim(256)

        interaction_mat = torch.matmul(out_text, torch.transpose(out_aspect, 1, 2))
        alpha = F.softmax(interaction_mat, dim=1)  # col-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta = F.softmax(interaction_mat, dim=2)  # row-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta_avg = beta.mean(dim=1, keepdim=True)  # batch_size x 1 x (asp) seq_len
        gamma = torch.matmul(alpha, beta_avg.transpose(1, 2))  # batch_size x (ctx) seq_len x 1
        weighted_sum = torch.matmul(torch.transpose(out_text, 1, 2), gamma).squeeze(-1)  # batch_size x 2*hidden_dim
        out = self.dense(weighted_sum)  # batch_size x polarity_dim
        if type(labels) == list and len(labels) == 0:
            return out

        loss = nn.CrossEntropyLoss()(out, labels)

        return {
            'logits': out,
            'loss': loss
        }
