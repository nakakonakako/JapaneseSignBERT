import torch
import torch.nn as nn

from signbert.model.thirdparty.st_gcn.net.st_gcn import HeadlessModel as STGCN


class ArmsExtractor(nn.Module):
    """
    STGCNを使用して腕のキーポイントを抽出および処理するためのPyTorchモジュール
    このモジュールは、特に腕に焦点を当てたキーポイントのシーケンスを処理するために設計されています
    STGCNを使用して特徴を抽出し、これらの特徴にmax poolingを適用する

    Attributes:
    stgcn (STGCN): 特徴抽出のための空間的時間畳み込みネットワーク
    """
    def __init__(
        self,
        in_channels,
        hid_dim,
        dropout
    ):
        """
        ArmsExtractorモジュールを初期化する

        Parameters:
        in_channels (int): STGCNの入力チャンネル数
        hid_dim (int): STGCNの隠れ層の次元数
        dropout (float): STGCNのドロップアウト率
        """
        super().__init__()
        self.stgcn = STGCN(
            in_channels=in_channels,
            num_hid=hid_dim,
            graph_args={'layout': 'mmpose_arms'},
            edge_importance_weighting=False,
            dropout=dropout
        )

    def forward(self, x):
        """
        ArmsExtractorモジュールのフォワードパス
        
        Parameters:
        x (Tensor): キーポイントのシーケンスを表すテンソル

        Returns:
        tuple: 処理された右腕と左腕のキーポイントを表すテンソルのタプル
        """
        # zero-paddingを除いたシーケンスの長さを計算
        lens = (x!=0.0).all(-1).all(-1).sum(1)
        # STGCNのために入力テンソルを並べ替えて再形成
        # (N, T, V, C) -> (N, C, T, V, 1)
        x = x.permute(0, 3, 1, 2).unsqueeze(-1)
        # STGCNを使用して入力を処理します
        x = self.stgcn(x, lens)
        # 両腕のキーポイントのインデックスを抽出
        rarm = x[:, :, :, (1,3,5)]
        larm = x[:, :, :, (0,2,4)]
        # 腕のキーポイントにmax poolingを適用
        rarm = torch.amax(rarm, dim=3, keepdim=True)
        larm = torch.amax(larm, dim=3, keepdim=True)

        return (rarm, larm)