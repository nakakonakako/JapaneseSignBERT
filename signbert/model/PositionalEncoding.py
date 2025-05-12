import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    入力のシーケンスに位置エンコーディングを追加するためのPyTorchモジュール
    
    Positional encodingは、transformerにシーケンス内の要素の順序を意識させるために使用されます。
    これは、要素の相対位置が重要な意味を持つタスクで特に必要です。
    
    Attributes:
    dropout (nn.Dropout): 正則化のためのドロップアウトレイヤー
    pe (Tensor): 位置エンコーディングテンソル
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        PositionalEncodingモジュールを初期化します。

        Parameters:
        d_model (int):埋め込みの次元（したがって位置エンコーディング）
        dropout (float): ドロップアウト率。デフォルトは0.1です
        max_len (int): 入力シーケンスの最大長。デフォルトは5000です
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # サイン波関数を使用して位置エンコーディング行列を作成
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # パラメータとして扱われないようにpeをバッファとして登録する
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        PositionalEncodingモジュールのフォワードパス。

        Parameters:
        x (Tensor): 位置エンコーディングが追加される入力テンソル。

        Returns:
        Tensor: 位置エンコーディングが追加された入力テンソル
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)