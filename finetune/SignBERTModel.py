# python finetune.py --config finetune/configs/ISLR_MSASL.yml --ckpt logs/pretrain/version_0/ckpts/last.ckpt
import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from torchmetrics import Accuracy

from finetune.ISLR.Head import Head
from signbert.model.PretrainSignBertModelManoTorch import SignBertModel as BaseModel


class SignBertModel(pl.LightningModule):
    """
    SignBERT+を実装するPyTorch Lightningモジュールです。

    このクラスは、事前学習済みのベースモデルと、手話認識の特定のタスクに合わせたカスタムヘッドを組み合わせます。
    
    Attributes:
    model (BaseModel): 事前学習済みのベースモデル
    head (Head): 手話認識のためにベースモデルに追加されたカスタムヘッド
    train_acc (Accuracy): 学習精度を追跡するためのメトリック
    val_acc (Accuracy): 検証精度を追跡するためのメトリック
    """

    def __init__(self, ckpt, lr, head_args):
        """
        SignBertModelを初期化

        Parameters:
        ckpt (str): 事前学習済みベースモデルのチェックポイントへのパス
        lr (float): オプティマイザの学習率
        head_args (dict): カスタムヘッドを初期化するための引数
        """
        super().__init__()
        self.lr = lr
        # ベースモデルをロードして初期化
        self.model = BaseModel.load_from_checkpoint(ckpt, map_location="cpu")
        self._init_base_model()
        # Extractorの出力に基づいてカスタムヘッドの入力チャンネルサイズを決定
        ge_hid_dim = self.model.hparams.gesture_extractor_args["hid_dim"]
        in_channels = ge_hid_dim[-1] if isinstance(ge_hid_dim, list) else ge_hid_dim
        # 手話認識のためのカスタムヘッドを初期化
        self.head = Head(in_channels=in_channels, **head_args)
        # 学習と検証の精度メトリックを初期化
        num_classes = head_args["num_classes"]
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def _init_base_model(self):
        """
        このメソッドは、事前学習済みのベースモデルから不要なコンポーネントを削除し、
        トレーニング中に重みが更新されるのを防ぐために重みを固定します。
        """
        # 現在のタスクに不要なベースモデルのコンポーネントを削除
        del self.model.pg
        del self.model.lhand_hd
        del self.model.rhand_hd
        del self.model.train_pck_20
        del self.model.train_pck_auc_20_40
        del self.model.val_pck_20
        del self.model.val_pck_auc_20_40
        # トレーニング中に重みが更新されるのを防ぐためにモデルを凍結
        self.model.freeze()

    def forward(self, arms, rhand, lhand):
        """
        SignBertModelのフォワードパス。

        Parameters:
        arms (Tensor): 腕のキーポイントの入力テンソル
        rhand (Tensor): 右手のキーポイントの入力テンソル
        lhand (Tensor): 左手のキーポイントの入力テンソル

        Returns:
        Tensor: モデルの出力予測
        """
        # 右手と左手のデータを連結
        x = torch.concat((rhand, lhand), dim=2)
        # ベースモデルのジェスチャーエクストラクタを使用して手のトークンを抽出
        rhand, lhand = self.model.ge(x)
        rhand = rhand.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        lhand = lhand.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        # ベースモデルの空間的時間処理を使用して腕トークンを抽出
        rarm, larm = self.model.stpe(arms)
        rarm = rarm.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        larm = larm.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        N, T, C, V = rhand.shape
        # 右手と左手のデータを連結
        rhand = rhand + rarm 
        lhand = lhand + larm 
        # さらなる処理のために手のデータを再形成
        rhand = rhand.view(N, T, C*V)
        lhand = lhand.view(N, T, C*V)
        # 右手と左手のデータに位置エンコーディングを適用
        rhand = self.model.pe(rhand) 
        lhand = self.model.pe(lhand) 
        # 右手と左手のデータをトランスフォーマーエンコーダーを通して処理
        rhand = self.model.te(rhand)
        lhand = self.model.te(lhand)
        # 最終的な予測のためにデータをカスタムヘッドに渡します
        x = self.head(rhand, lhand)

        return x

    def training_step(self, batch):
        """
        SignBertModelのトレーニングステップ
        データの単一バッチを処理し、損失を計算し、モデルを更新し、メトリックを記録する
        Parameters:
        batch (dict): データのバッチ。腕、左手、右手、クラスラベルのテンソルを含む

        Returns:
        torch.Tensor: バッチの計算された損失
        """
        # バッチから腕、左手、右手、ラベルを抽出
        arms = batch["arms"]
        lhand = batch["lhand"]
        rhand = batch["rhand"]
        labels = batch["class_id"]
        # モデルの出力予測を計算
        logits = self(arms, rhand, lhand)
        # 損失を計算
        loss = F.cross_entropy(logits, labels)
        # 精度を計算
        acc = self.train_acc(logits, labels) 
        # 損失と精度を記録
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True, prog_bar=False)

        return loss

    def validation_step(self, batch, dataloader_idx=0):
        """
        SignBertModelの検証ステップ

        評価データの単一バッチを処理し、損失を計算し、モデルのパフォーマンスを評価し、監視用のメトリックを記録します。

        Parameters:
        batch (dict): 検証データのバッチ。腕、左手、右手、クラスラベルのテンソルを含む
        dataloader_idx (int, optional): データローダーのインデックス。デフォルトは0です。

        Returns:
        None: このメソッドは検証損失と精度を記録しますが、何も返しません。
        """
        # 検証バッチから腕、左手、右手、ラベルを抽出
        arms = batch["arms"]
        lhand = batch["lhand"]
        rhand = batch["rhand"]
        labels = batch["class_id"]
        # モデルの出力予測を計算
        logits = self(arms, rhand, lhand)
        # 損失を計算
        loss = F.cross_entropy(logits, labels)
        # 精度を計算
        acc = self.val_acc(logits, labels) 
        # 損失と精度を記録
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=False)
        self.log("val_acc", acc, on_step=True, on_epoch=True, logger=True, prog_bar=True)

    def test_step(self, batch, dataloader_idx=0):
        # テストバッチから腕、左手、右手、ラベルを抽出
        arms = batch["arms"]
        lhand = batch["lhand"]
        rhand = batch["rhand"]
        labels = batch["class_id"]
        # モデルの出力予測を計算
        logits = self(arms, rhand, lhand)
        # 損失を計算
        loss = F.cross_entropy(logits, labels)
        # 精度を計算
        acc = self.val_acc(logits, labels)  # You can reuse the validation accuracy metric
        # 損失と精度を記録
        self.log("test_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=False)
        self.log("test_acc", acc, on_step=True, on_epoch=True, logger=True, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer