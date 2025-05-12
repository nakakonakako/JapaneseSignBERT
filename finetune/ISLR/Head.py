import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """
    孤立した手話認識タスクのためのヘッドとして機能するPyTorchモジュール。

    このモジュールは、右手と左手のデータから特徴を組み合わせ、
    時間的マージングを適用し、線形層を使用してクラスラベルを予測するためのヘッドとして機能します。

    Attributes:
    temporal_merging (nn.Sequential): 時間的特徴をマージするためのシーケンシャルモデル。
    classifies (nn.Linear): 分類のための線形層。
    """
    def __init__(self, in_channels, num_classes):
        """
        Headモジュールを初期化します。
        
        Parameters:
        in_channels (int): 右手と左手の特徴の入力チャンネル数。
        num_classes (int): 分類タスクのクラス数。
        """
        super().__init__()
        # 右手と左手の特徴の連結を考慮してin_channelsを調整します
        in_channels = in_channels * 2
        # 時間的マージング層を定義
        self.temporal_merging = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Softmax(dim=1)
        )
        # 分類層を定義
        self.classifier = nn.Linear(in_channels, num_classes)
    
    def forward(self, rhand, lhand):
        """
        Headモジュールのフォワードパス。

        Parameters:
        rhand (Tensor): 右手の特徴の入力テンソル。
        lhand (Tensor): 左手の特徴の入力テンソル。

        Returns:
        Tensor: 分類後の出力テンソル。
        """
        # 右手と左手の特徴を連結
        x = torch.concat((rhand, lhand), axis=2)
        # 時間的マージングを連結された特徴に適用
        x = self.temporal_merging(x) * x
        # 時間次元に対してmax-poolingを適用
        x = F.max_pool1d(x.mT, kernel_size=x.shape[1]).squeeze()
        # 最終的なロジットを取得するために分類器を通過
        x = self.classifier(x)

        return x
        