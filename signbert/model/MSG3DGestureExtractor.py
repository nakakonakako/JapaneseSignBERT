import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from signbert.model.thirdparty.MS_G3D.model.msg3d import HeadlessModel as MSG3D
from signbert.model.MediapipeHandPooling import MediapipeHandPooling
from signbert.model.thirdparty.st_gcn.net.st_gcn import HeadlessModel as STGCN


class Hands17Graph:
    """
    21のキーポイントを持つ手のグラフ構造を表すクラス

    このクラスは、ノードがキーポイントを表し、エッジがこれらのキーポイント間の接続を表す手のグラフ表現を作成します。
    
    Attributes:
    num_nodes (int): グラフ内のノード（キーポイント）の数
    edges (list): キーポイント間の接続を表すエッジのリスト    
    self_loops (list): A list of self-loops for each node in the graph.
    self_loops (list): 各ノードに対する自己ループのリスト
    A_binary (ndarray): 自己ループなしのグラフのバイナリ隣接行列
    A_binary_with_I (ndarray): 自己ループを持つグラフのバイナリ隣接行列
    """
    def __init__(self, *args, **kwargs):
        num_node = 21 # 手のキーポイントの合計数
        self.num_nodes = num_node
        # 手首、手のひら、親指、人差し指、中指、薬指、小指の各部位に対応するキーポイントのインデックスを定義
        inward = [
            (0, 1), (0, 5), (0, 17), # Wrist
            (5, 9), (9, 13), (13, 17), # Palm
            (1, 2), (2, 3), (3, 4), # Thumb
            (5, 6), (6, 7), (7, 8), # Index
            (9, 10), (10, 11), (11, 12), # Middle
            (13, 14), (14, 15), (15, 16), # Ring
            (17, 18), (18, 19), (19, 20) # Pinky
        ]
        # 外向きのエッジは、内向きのエッジを逆にしたもの
        outward = [(j, i) for (i, j) in inward]
        neighbor = inward + outward
        # Save the edges and self-loops
        # エッジと自己ループを保存
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        # バイナリ隣接行列を生成
        self.A_binary = self.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = self.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)

    def get_adjacency_matrix(self, edges, num_nodes):
        """
        グラフの隣接行列を生成

        Parameters:
        edges (list): グラフ内のエッジのリスト
        num_nodes (int): グラフ内のノードの数
        Returns:
        ndarray: グラフを表すバイナリ隣接行列
        """
        # Initialize an adjacency matrix with zeros
        # ゼロで初期化された隣接行列を初期化
        A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for edge in edges:
            # 隣接行列を埋める：エッジが存在する場合は1、それ以外は0
            A[edge] = 1.
        return A
    

class PretrainGraph:
    """
    42キーポイントを持つ左右の手のグラフ構造を表すクラス

    このクラスは、ノードがキーポイントを表し、エッジがこれらのキーポイント間の接続を表す手のグラフ表現を作成します。

    最初に右手が来ることに注意してください。

    Attributes:
    num_nodes (int): グラフ内のノード（キーポイント）の数
    edges (list): キーポイント間の接続を表すエッジのリスト
    self_loops (list): 各ノードに対する自己ループのリスト
    A_binary (ndarray): 自己ループなしのグラフのバイナリ隣接行列
    A_binary_with_I (ndarray): 自己ループを持つグラフのバイナリ隣接行列
    """
    def __init__(self, *args, **kwargs):
        num_node = 42 # 両手のキーポイントの合計数
        self.num_nodes = num_node
        rhand_inward = [
            (0, 1), (0, 5), (0, 17), # Wrist
            (5, 9), (9, 13), (13, 17), # Palm
            (1, 2), (2, 3), (3, 4), # Thumb
            (5, 6), (6, 7), (7, 8), # Index
            (9, 10), (10, 11), (11, 12), # Middle
            (13, 14), (14, 15), (15, 16), # Ring
            (17, 18), (18, 19), (19, 20) # Pinky
        ]
        # 左手のエッジを定義し、21でオフセット
        lhand_inward = np.array(rhand_inward) + 21
        lhand_inward = list(map(tuple, lhand_inward))
        # 外向きのエッジは、内向きのエッジを逆にしたもの
        rhand_outward = [(j, i) for (i, j) in rhand_inward]
        lhand_outward = [(j, i) for (i, j) in lhand_inward]
        # すべてのエッジを結合
        neighbor = rhand_inward + rhand_outward + lhand_inward + lhand_outward
        # エッジと自己ループを保存
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        # バイナリ隣接行列を生成
        self.A_binary = self.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = self.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)

    def get_adjacency_matrix(self, edges, num_nodes):
        """
        グラフの隣接行列を生成

        Parameters:
        edges (list): グラフ内のエッジのリスト
        num_nodes (int): グラフ内のノードの数
        Returns:
        ndarray: グラフを表すバイナリ隣接行列
        """
        # ゼロで初期化された隣接行列を初期化
        A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for edge in edges:
            # 隣接行列を埋める：エッジが存在する場合は1、それ以外は0
            A[edge] = 1.
        return A


class PretrainGestureExtractor(nn.Module):
    """
    MSG3DとSTGCNを使用して手のジェスチャデータから特徴を抽出するためのPyTorchモジュール

    このモジュールは、最初にMSG3Dモデルを使用して手のジェスチャを表すキーポイントを処理し、
    追加の特徴抽出のためにクラスタリングとプーリングを適用することができます。

    下のGestureExtractorクラスと同様に機能しますが、事前トレーニング中に存在する両手のケースを処理します。

    Attributes:
    model (MSG3D): マルチスケール空間時間グラフ畳み込みネットワーク
    maxpool1 (MediapipeHandPooling): 初期特徴抽出のための手のプーリングレイヤー
    stgcn (STGCN): クラスタリングされたデータ用の空間的時間グラフ畳み込みネットワーク
    maxpool2 (MediapipeHandPooling): さらなる特徴抽出のための追加の手のプーリングレイヤー
    dropout (nn.Dropout): 正則化のためのドロップアウトレイヤー
    """
    def __init__(
            self,
            num_point,
            num_gcn_scales,
            num_g3d_scales,
            hid_dim,
            in_channels,
            do_cluster,
            msg_3d_dropout=0.0,
            st_gcn_dropout=0.0,
            dropout=0.0,
            relu_between=False,
            input_both_hands=False
        ):
        super().__init__()
        self.do_cluster = do_cluster
        self.relu_between = relu_between
        self.input_both_hands = input_both_hands
        # MSG3Dモデルを初期化
        self.model = MSG3D(
            num_point,
            num_gcn_scales,
            num_g3d_scales,
            PretrainGraph(), # Pretrain graph
            hid_dim,
            msg_3d_dropout,
            in_channels,
        )
        # クラスタリングとプーリングコンポーネントを初期化
        if do_cluster:
            self.maxpool1 = MediapipeHandPooling(last=False)
            self.stgcn = STGCN(
                in_channels=hid_dim[-1],
                num_hid=hid_dim[-1],
                graph_args={'layout': 'mediapipe_six_hand_cluster'},
                edge_importance_weighting=False,
                dropout=st_gcn_dropout
            )
            self.maxpool2 = MediapipeHandPooling(last=True)
        # 正則化のためのドロップアウトを初期化
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        入力キーポイントデータを処理するためのフォワードパス

        Parameters:
        x (Tensor): キーポイントデータを含む入力テンソル

        Returns:
        tuple: 処理された右手と左手の特徴
        """
        # シーケンスの長さを計算（ゼロパディングを除く）
        lens = (x!=0.0).all(-1).all(-1).sum(1)
        # MSG3Dは、データを（N、C、T、V、M）形式で受け取る
        x = x.permute(0, 3, 1, 2).unsqueeze(-1)
        x = self.model(x, lens)
        # クラスタリングとプーリングを適用する
        if self.do_cluster:
            rhand = x[...,:21]
            lhand = x[...,21:]
            # 最初のmaxプーリングを適用
            rhand = self.maxpool1(rhand)
            lhand = self.maxpool1(lhand)
            # 有効になっている場合は、間に非線形性を適用します
            if self.relu_between:
                rhand = F.relu(rhand) # Add M dimension
                lhand = F.relu(lhand) # Add M dimension
            rhand = rhand.unsqueeze(-1)
            lhand = lhand.unsqueeze(-1)
            # STGCNで特徴を抽出
            rhand = self.stgcn(rhand, lens)
            lhand = self.stgcn(lhand, lens)
            rhand = rhand.squeeze(1)
            lhand = lhand.squeeze(1)
            # 2番目のmaxプーリングを適用
            rhand = self.maxpool2(rhand)
            lhand = self.maxpool2(lhand)
            # 有効になっている場合は、間に非線形性を適用します
            if self.relu_between:
                rhand = F.relu(rhand)
                lhand = F.relu(lhand)
            rhand = rhand.unsqueeze(-1) # Add M dimension
            lhand = lhand.unsqueeze(-1) # Add M dimension
            x = torch.concat((rhand, lhand), dim=3)
        else:
            x = x.unsqueeze(-1) # Add M dimension
        # ドロップアウトを適用
        x = self.dropout(x)
        rhand = x[:, :, :, 0]
        lhand = x[:, :, :, 1]

        return (rhand, lhand)


class GestureExtractor(nn.Module):
    """
    MSG3DとSTGCNを使用して手のジェスチャデータから特徴を抽出するためのPyTorchモジュール

    このモジュールは、最初にMSG3Dモデルを使用して手のジェスチャを表すキーポイントを処理し、
    追加の特徴抽出のためにクラスタリングとプーリングを適用することができます。

    Attributes:
    model (MSG3D): マルチスケール空間時間グラフ畳み込みネットワーク
    maxpool1 (MediapipeHandPooling): 初期特徴抽出のための手のプーリングレイヤー
    stgcn (STGCN): クラスタリングされたデータ用の空間的時間グラフ畳み込みネットワーク
    maxpool2 (MediapipeHandPooling): さらなる特徴抽出のための追加の手のプーリングレイヤー
    dropout (nn.Dropout): 正則化のためのドロップアウトレイヤー
    """
    def __init__(
            self,
            num_point,
            num_gcn_scales,
            num_g3d_scales,
            hid_dim,
            in_channels,
            do_cluster,
            msg_3d_dropout=0.0,
            st_gcn_dropout=0.0,
            dropout=0.0,
            relu_between=False,
            input_both_hands=False
        ):
        super().__init__()
        self.do_cluster = do_cluster
        self.relu_between = relu_between
        self.input_both_hands = input_both_hands
        # MSG3Dモデルを初期化
        self.model = MSG3D(
            num_point,
            num_gcn_scales,
            num_g3d_scales,
            Hands17Graph(), # Feasibility study graph
            hid_dim,
            msg_3d_dropout,
            in_channels,
        )
        # クラスタリングとプーリングコンポーネントを初期化
        if do_cluster:
            self.maxpool1 = MediapipeHandPooling(last=False)
            self.stgcn = STGCN(
                in_channels=hid_dim[-1],
                num_hid=hid_dim[-1],
                graph_args={'layout': 'mediapipe_six_hand_cluster'},
                edge_importance_weighting=False,
                dropout=st_gcn_dropout
            )
            self.maxpool2 = MediapipeHandPooling(last=True)
        # 正則化のためのドロップアウトを初期化
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        入力キーポイントデータを処理するためのフォワードパス

        Parameters:
        x (Tensor): キーポイントデータを含む入力テンソル

        Returns:
        Tensor: ネットワークを通過した後の処理された特徴
        """
        # シーケンスの長さを計算（ゼロパディングを除く）
        lens = (x!=0.0).all(-1).all(-1).sum(1)
        # MSG3Dは、データを（N、C、T、V、M）形式で受け取る
        x = x.permute(0, 3, 1, 2).unsqueeze(-1)
        x = self.model(x, lens)
        # クラスタリングとプーリングを適用する
        if self.do_cluster:
            # 最初のmaxプーリングを適用
            x = self.maxpool1(x)
            # 有効になっている場合は、間に非線形性を適用します
            if self.relu_between:
                x = F.relu(x)
            x = x.unsqueeze(-1)
            # STGCNで特徴を抽出
            x = self.stgcn(x, lens)
            x = x.squeeze(-1)
            # 2番目のmaxプーリングを適用
            x = self.maxpool2(x)
            # 有効になっている場合は、間に非線形性を適用します
            if self.relu_between:
                x = F.relu(x)            
            x = x.unsqueeze(-1) # Add M dimension
        else:
            x = x.unsqueeze(-1) # Add M dimension
        # ドロップアウトを適用
        x = self.dropout(x)

        return x