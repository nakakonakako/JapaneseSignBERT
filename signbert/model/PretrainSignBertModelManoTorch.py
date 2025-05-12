import os

import torch
import lightning.pytorch as pl

from signbert.utils import my_import
from signbert.model.PositionalEncoding import PositionalEncoding
from signbert.metrics.PCK import PCK, PCKAUC
from manotorch.manolayer import ManoLayer, MANOOutput


class SignBertModel(pl.LightningModule):
    """
    手話認識のためのSignBERTを実装するPyTorchモジュール

    このクラスは、ジェスチャー抽出、位置エンコーディング、空間的時間処理、トランスフォーマーエンコーダー、
    MANOレイヤーを組み合わせて、手話ジェスチャーを処理し解釈します
    
    事前学習中に使用される両手構成を処理します。

    Attributes:
    in_channels, num_hid, num_headsなどいくつかのパラメータから構成
    ge (GestureExtractor): ジェスチャー抽出器
    pe (PositionalEncoding): 位置エンコーディング
    stpe (SpatialTemporalProcessing): 空間的時間処理
    te (TransformerEncoder: トランスフォーマーエンコーダー
    pg (Linear): 予測のための線形層
    rhand_hd, lhand_hd (ManoLayer): 詳細な手形の推定を行うMANOレイヤー
    学習中と検証中のPCKとPCKAUCメトリクス
    """
    def __init__(
            self, 
            in_channels, 
            num_hid, 
            num_heads,
            tformer_n_layers,
            tformer_dropout,
            eps, 
            lmbd, 
            weight_beta, 
            weight_delta,
            lr,
            hand_cluster,
            n_pca_components,
            gesture_extractor_cls,
            gesture_extractor_args,
            arms_extractor_cls,
            arms_extractor_args,
            total_steps=None,
            normalize_inputs=False,
            use_pca=True,
            flat_hand=False,
            weight_decay=0.01,
            use_onecycle_lr=False,
            pct_start=None,
            *args,
            **kwargs,
        ):
        super().__init__()
        self.save_hyperparameters()
        # 自動最適化は無効になっているので、各バッチを独立して逆伝播できるように、手動で行う必要があります
        self.automatic_optimization = False

        self.in_channels = in_channels
        self.num_hid = num_hid
        self.num_heads = num_heads
        self.tformer_n_layers = tformer_n_layers
        self.tformer_dropout = tformer_dropout
        self.eps = eps
        self.lmbd = lmbd
        self.weight_beta = weight_beta
        self.weight_delta = weight_delta
        self.total_steps = total_steps
        self.lr = lr
        self.hand_cluster = hand_cluster
        self.n_pca_components = n_pca_components
        self.gesture_extractor_cls = my_import(gesture_extractor_cls)
        self.gesture_extractor_args = gesture_extractor_args
        self.arms_extractor_cls = my_import(arms_extractor_cls)
        self.arms_extractor_args = arms_extractor_args
        self.normalize_inputs = normalize_inputs
        self.use_pca = use_pca
        self.flat_hand = flat_hand
        self.weight_decay = weight_decay
        self.use_onecycle_lr = use_onecycle_lr
        self.pct_start = pct_start
        # もしクラスタリングが有効になっている場合、入力チャンネルを動的に制御する変数
        num_hid_mult = 1 if hand_cluster else 21
        # モデルのコンポーネントの初期化
        self.ge = self.gesture_extractor_cls(**gesture_extractor_args)
        self.pe = PositionalEncoding(
            d_model=num_hid*num_hid_mult,
            dropout=0.1,
            max_len=1000,
        )
        self.stpe = self.arms_extractor_cls(**arms_extractor_args)
        el = torch.nn.TransformerEncoderLayer(d_model=num_hid*num_hid_mult, nhead=num_heads, batch_first=True, dropout=tformer_dropout)
        self.te = torch.nn.TransformerEncoder(el, num_layers=tformer_n_layers)
        self.pg = torch.nn.Linear(
            in_features=num_hid*num_hid_mult,
            out_features=(
                n_pca_components + 3 + # 手のポーズ(シータ) + 3D手の中心
                10 + # 手の形状(ベータ)
                9 + # 回転行列
                2 + # オフセット
                1 # スケール
            )
        )
        # MANOの初期化（右手と左手の両方）    
        mano_assets_root = os.path.split(__file__)[0]
        mano_assets_root = os.path.join(mano_assets_root, "thirdparty", "mano_assets")
        assert os.path.isdir(mano_assets_root), "Download MANO files, check README."
        self.rhand_hd = ManoLayer(
            center_idx=0,
            flat_hand_mean=flat_hand,
            mano_assets_root=mano_assets_root,
            use_pca=use_pca,
            ncomps=n_pca_components,
        )
        self.lhand_hd = ManoLayer(
            center_idx=0,
            flat_hand_mean=flat_hand,
            mano_assets_root=mano_assets_root,
            use_pca=use_pca,
            ncomps=n_pca_components,
            side="left"
        )
        # PCK and PCKAUC metrics for training and validation
        # 学習と評価におけるPCKとPCKAUCメトリクス
        self.train_pck_20 = PCK(thr=20)
        self.train_pck_auc_20_40 = PCKAUC(thr_min=20, thr_max=40)
        self.val_pck_20 = PCK(thr=20)
        self.val_pck_auc_20_40 = PCKAUC(thr_min=20, thr_max=40)
        # プレースホルダー
        self.mean_loss = []
        self.mean_pck_20 = []

    def forward(self, arms, rhand, lhand):
        #ここがposition embeddingの部分
        # 右手と左手のデータを連結
        x = torch.concat((rhand, lhand), dim=2)
        # ジェスチャー抽出器を使用して手のトークンを抽出
        rhand, lhand = self.ge(x)
        rhand = rhand.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        lhand = lhand.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        # 空間的時間アーム抽出器を使用してアームトークンを抽出
        rarm, larm = self.stpe(arms)
        rarm = rarm.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        larm = larm.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        N, T, C, V = rhand.shape
        # 右手と左手のトークンを空間的時間的な位置トークンと組み合わせる        
        rhand = rhand + rarm 
        lhand = lhand + larm 
        # 右手と左手のデータを処理するためにreshape
        rhand = rhand.view(N, T, C*V)
        lhand = lhand.view(N, T, C*V)
        # ここがpositional encodingの部分
        # 位置エンコーディングを適用
        rhand = self.pe(rhand) 
        lhand = self.pe(lhand) 
        # ここがtransformer encoderの部分
        # データをtransformer encoderに通す
        rhand = self.te(rhand)
        lhand = self.te(lhand)
        # 右手と左手の手とカメラパラメータを予測
        rhand_params = self.pg(rhand)
        lhand_params = self.pg(lhand)
        # 右手と左手のパラメータを抽出
        offset = self.n_pca_components + 3
        rhand_pose_coeffs = rhand_params[...,:offset]
        rhand_betas = rhand_params[...,offset:offset+10]
        lhand_pose_coeffs = lhand_params[...,:offset]
        lhand_betas = lhand_params[...,offset:offset+10]
        offset += 10
        rhand_R = rhand_params[...,offset:offset+9]
        rhand_R = rhand_R.view(N, T, 3, 3)
        lhand_R = lhand_params[...,offset:offset+9]
        lhand_R = lhand_R.view(N, T, 3, 3)
        offset +=9
        rhand_O = rhand_params[...,offset:offset+2]
        lhand_O = lhand_params[...,offset:offset+2]
        offset += 2
        rhand_S = rhand_params[...,offset:offset+1]
        lhand_S = lhand_params[...,offset:offset+1]
        rhand_pose_coeffs = rhand_pose_coeffs.view(N*T, -1)
        lhand_pose_coeffs = lhand_pose_coeffs.view(N*T, -1)
        rhand_betas = rhand_betas.view(N*T, -1)
        lhand_betas = lhand_betas.view(N*T, -1)
        # MANOモデルを適用して3Dジョイントと頂点を取得
        rhand_mano_output: MANOOutput = self.rhand_hd(rhand_pose_coeffs, rhand_betas)
        lhand_mano_output: MANOOutput = self.lhand_hd(lhand_pose_coeffs, lhand_betas)
        # 右手と左手のMANO出力を抽出してreshape
        rhand_vertices = rhand_mano_output.verts
        rhand_joints_3d = rhand_mano_output.joints
        rhand_pose_coeffs = rhand_pose_coeffs.view(N, T, -1)
        rhand_betas = rhand_betas.view(N, T, -1)
        rhand_vertices = rhand_vertices.view(N, T, 778, 3).detach().cpu()
        rhand_center_joint = rhand_mano_output.center_joint.detach().cpu()
        rhand_joints_3d = rhand_joints_3d.view(N, T, 21, 3)
        lhand_vertices = lhand_mano_output.verts
        lhand_joints_3d = lhand_mano_output.joints
        lhand_pose_coeffs = lhand_pose_coeffs.view(N, T, -1)
        lhand_betas = lhand_betas.view(N, T, -1)
        lhand_vertices = lhand_vertices.view(N, T, 778, 3).detach().cpu()
        lhand_center_joint = lhand_mano_output.center_joint.detach().cpu()
        lhand_joints_3d = lhand_joints_3d.view(N, T, 21, 3)
        # 2D画像座標を取得するために3Dジョイントに直交投影を適用
        rhand = torch.matmul(rhand_R, rhand_joints_3d.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        rhand = rhand[...,:2]
        rhand *= rhand_S.unsqueeze(-1)
        rhand += rhand_O.unsqueeze(2)
        lhand = torch.matmul(lhand_R, lhand_joints_3d.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        lhand = lhand[...,:2]
        lhand *= lhand_S.unsqueeze(-1)
        lhand += lhand_O.unsqueeze(2)
        # 処理されたデータを返す
        return {
            "rhand": (rhand, rhand_pose_coeffs, rhand_betas, rhand_vertices, rhand_R, rhand_S, rhand_O, rhand_center_joint, rhand_joints_3d),
            "lhand": (lhand, lhand_pose_coeffs, lhand_betas, lhand_vertices, lhand_R, lhand_S, lhand_O, lhand_center_joint, lhand_joints_3d)
        }

    def training_step(self, batch, batch_idx):
        # 最適化とスケジューラを取得（手動最適化の一部）
        opt = self.optimizers()
        sch = self.lr_schedulers()
        # バッチ内の<key-value>ペアを処理する（<データセット名：バッチデータ>）
        for k, v in batch.items():
            # Unpack the batch data
            (seq_idx, 
            arms,
            rhand, 
            rhand_masked,
            rhand_masked_frames_idx,
            rhand_scores,
            lhand, 
            lhand_masked,
            lhand_masked_frames_idx,
            lhand_scores) = v
            # モデルを通してデータを処理
            hand_data = self(arms, rhand_masked, lhand_masked)
            # 右手と左手の出力からロジット、ポーズ係数、ベータを抽出
            (rhand_logits, rhand_theta, rhand_beta, _, _, _, _, _, _) = hand_data["rhand"]
            (lhand_logits, lhand_theta, lhand_beta, _, _, _, _, _, _) = hand_data["lhand"]
            # マスクされたフレームのみに損失を適用
            rhand_valid_idxs = torch.where(rhand_masked_frames_idx != -1.)
            rhand_logits = rhand_logits[rhand_valid_idxs]
            rhand = rhand[rhand_valid_idxs]
            rhand_scores = rhand_scores[rhand_valid_idxs]
            lhand_valid_idxs = torch.where(lhand_masked_frames_idx != -1.)
            lhand_logits = lhand_logits[lhand_valid_idxs]
            lhand = lhand[lhand_valid_idxs]
            lhand_scores = lhand_scores[lhand_valid_idxs]
            # 両手のLRec (Loss of Reconstruction)とLReg (Loss of Regularization)を計算
            rhand_lrec = torch.norm(rhand_logits - rhand, p=1, dim=2)
            rhand_scores = torch.where(rhand_scores >= self.eps, 1., rhand_scores)
            rhand_lrec = (rhand_lrec * rhand_scores).sum()
            lhand_lrec = torch.norm(lhand_logits - lhand, p=1, dim=2)
            lhand_scores = torch.where(lhand_scores >= self.eps, 1., lhand_scores)
            lhand_lrec = (lhand_lrec * lhand_scores).sum()
            rhand_beta_t_minus_one = torch.roll(rhand_beta, shifts=1, dims=1)
            rhand_beta_t_minus_one[:, 0] = 0.
            rhand_lreg = torch.norm(rhand_theta, 2) + self.weight_beta * torch.norm(rhand_beta, 2) + \
                self.weight_delta * torch.norm(rhand_beta - rhand_beta_t_minus_one, 2)
            # 右手のLRecとLRegを組み合わせる
            rhand_loss = rhand_lrec + (self.lmbd * rhand_lreg)
            lhand_lrec = torch.norm(lhand_logits[lhand_scores>self.eps] - lhand[lhand_scores>self.eps], p=1, dim=1).sum()
            lhand_beta_t_minus_one = torch.roll(lhand_beta, shifts=1, dims=1)
            lhand_beta_t_minus_one[:, 0] = 0.
            lhand_lreg = torch.norm(lhand_theta, 2) + self.weight_beta * torch.norm(lhand_beta, 2) + \
                self.weight_delta * torch.norm(lhand_beta - lhand_beta_t_minus_one, 2)
            # 左手のLRecとLRegを組み合わせる
            lhand_loss = lhand_lrec + (self.lmbd * lhand_lreg)
            # 両手の損失を組み合わせる
            loss = rhand_loss + lhand_loss
            # バックワードパスを手動で行う
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
            if isinstance(sch, torch.optim.lr_scheduler.OneCycleLR):
                sch.step()
            if self.normalize_inputs:
                # データセットの平均と標準偏差がtrainerと同じデバイスにあるか確認
                if self.device != self.trainer.datamodule.means[k].device:
                   self.trainer.datamodule.means[k] = self.trainer.datamodule.means[k].to(self.device)
                if self.device != self.trainer.datamodule.stds[k].device:
                    self.trainer.datamodule.stds[k] = self.trainer.datamodule.stds[k].to(self.device)
                # 平均0と標準偏差1の正規化を逆変換して2D画像座標を取得
                means = self.trainer.datamodule.means[k]
                stds = self.trainer.datamodule.stds[k]
                rhand_logits = (rhand_logits * stds) + means
                lhand_logits = (lhand_logits * stds) + means
                rhand = (rhand * stds) + means
                lhand = (lhand * stds) + means
            # PCKメトリクスを計算
            rhand_pck_20 = self.train_pck_20.update(preds=rhand_logits, target=rhand)
            rhand_pck_20 = self.train_pck_20.compute()
            self.train_pck_20.reset()
            lhand_pck_20 = self.train_pck_20(preds=lhand_logits, target=lhand)
            lhand_pck_20 = self.train_pck_20.compute()
            self.train_pck_20.reset()
            rhand_pck_auc_20_40 = self.train_pck_auc_20_40(preds=rhand_logits, target=rhand)
            rhand_pck_auc_20_40 = self.train_pck_auc_20_40.compute()
            self.train_pck_auc_20_40.reset()
            lhand_pck_auc_20_40 = self.train_pck_auc_20_40(preds=lhand_logits, target=lhand)
            lhand_pck_auc_20_40 = self.train_pck_auc_20_40.compute()
            self.train_pck_auc_20_40.reset()
            # メトリクスをログに記録
            self.log(f"{k}_train_loss", loss, prog_bar=False)
            self.log(f"{k}_train_rhand_PCK_20", rhand_pck_20, prog_bar=False)
            self.log(f"{k}_train_lhand_PCK_20", lhand_pck_20, prog_bar=False)
            self.log(f"{k}_train_rhand_PCK_auc_20_40", rhand_pck_auc_20_40, prog_bar=False)
            self.log(f"{k}_train_lhand_PCK_auc_20_40", lhand_pck_auc_20_40, prog_bar=False)
        
    def validation_step(self, batch, batch_idx, dataloader_idx):
        # データローダーのインデックスに基づいてデータセットキーを特定
        dataset_key = list(self.trainer.datamodule.val_dataloaders.keys())[dataloader_idx]
        # バッチデータを展開
        (seq_idx, 
        arms,
        rhand, 
        rhand_masked,
        rhand_masked_frames_idx,
        rhand_scores,
        lhand, 
        lhand_masked,
        lhand_masked_frames_idx,
        lhand_scores) = batch
        # モデルを通してデータを処理
        hand_data = self(arms, rhand_masked, lhand_masked)
        # 右手と左手の出力からロジット、ポーズ係数、ベータを抽出
        (rhand_logits, rhand_theta, rhand_beta, _, _, _, _, _, _) = hand_data["rhand"]
        (lhand_logits, lhand_theta, lhand_beta, _, _, _, _, _, _) = hand_data["lhand"]
        # マスクされたフレームのみに損失を適用
        rhand_valid_idxs = torch.where(rhand_masked_frames_idx != -1.)
        rhand_logits = rhand_logits[rhand_valid_idxs]
        rhand = rhand[rhand_valid_idxs]
        rhand_scores = rhand_scores[rhand_valid_idxs]
        lhand_valid_idxs = torch.where(lhand_masked_frames_idx != -1.)
        lhand_logits = lhand_logits[lhand_valid_idxs]
        lhand = lhand[lhand_valid_idxs]
        lhand_scores = lhand_scores[lhand_valid_idxs]
        # 両手のLRec (Loss of Reconstruction)とLReg (Loss of Regularization)を計算
        rhand_lrec = torch.norm(rhand_logits - rhand, p=1, dim=2)
        rhand_scores = torch.where(rhand_scores >= self.eps, 1., rhand_scores)
        rhand_lrec = (rhand_lrec * rhand_scores).sum()
        lhand_lrec = torch.norm(lhand_logits - lhand, p=1, dim=2)
        lhand_scores = torch.where(lhand_scores >= self.eps, 1., lhand_scores)
        lhand_lrec = (lhand_lrec * lhand_scores).sum()
        rhand_beta_t_minus_one = torch.roll(rhand_beta, shifts=1, dims=1)
        rhand_beta_t_minus_one[:, 0] = 0.
        rhand_lreg = torch.norm(rhand_theta, 2) + self.weight_beta * torch.norm(rhand_beta, 2) + \
            self.weight_delta * torch.norm(rhand_beta - rhand_beta_t_minus_one, 2)
        rhand_loss = rhand_lrec + (self.lmbd * rhand_lreg)
        lhand_lrec = torch.norm(lhand_logits[lhand_scores>self.eps] - lhand[lhand_scores>self.eps], p=1, dim=1).sum()
        lhand_beta_t_minus_one = torch.roll(lhand_beta, shifts=1, dims=1)
        lhand_beta_t_minus_one[:, 0] = 0.
        lhand_lreg = torch.norm(lhand_theta, 2) + self.weight_beta * torch.norm(lhand_beta, 2) + \
            self.weight_delta * torch.norm(lhand_beta - lhand_beta_t_minus_one, 2)
        lhand_loss = lhand_lrec + (self.lmbd * lhand_lreg)
        # 両手の損失を組み合わせる
        loss = rhand_loss + lhand_loss
        # メトリクスを計算
        if self.normalize_inputs:
            # trainerと同じデバイスに平均と標準偏差があるか確認
            if self.device != self.trainer.datamodule.means[dataset_key].device:
                self.trainer.datamodule.means[dataset_key] = self.trainer.datamodule.means[dataset_key].to(self.device)
            if self.device != self.trainer.datamodule.stds[dataset_key].device:
                self.trainer.datamodule.stds[dataset_key] = self.trainer.datamodule.stds[dataset_key].to(self.device)
            # 逆正規化
            means = self.trainer.datamodule.means[dataset_key]
            stds = self.trainer.datamodule.stds[dataset_key]
            rhand_logits = (rhand_logits * stds) + means
            lhand_logits = (lhand_logits * stds) + means
            rhand = (rhand * stds) + means
            lhand = (lhand * stds) + means
        # PCKメトリクスを計算
        rhand_pck_20 = self.val_pck_20.update(preds=rhand_logits, target=rhand)
        rhand_pck_20 = self.val_pck_20.compute()
        self.val_pck_20.reset()
        lhand_pck_20 = self.val_pck_20(preds=lhand_logits, target=lhand)
        lhand_pck_20 = self.val_pck_20.compute()
        self.val_pck_20.reset()
        rhand_pck_auc_20_40 = self.val_pck_auc_20_40(preds=rhand_logits, target=rhand)
        rhand_pck_auc_20_40 = self.val_pck_auc_20_40.compute()
        self.val_pck_auc_20_40.reset()
        lhand_pck_auc_20_40 = self.val_pck_auc_20_40(preds=lhand_logits, target=lhand)
        lhand_pck_auc_20_40 = self.val_pck_auc_20_40.compute()
        self.val_pck_auc_20_40.reset()
        # メトリクスをログに記録
        self.log(f"{dataset_key}_val_loss", loss, prog_bar=False)
        self.log(f"{dataset_key}_val_rhand_pck_20", rhand_pck_20, prog_bar=False)
        self.log(f"{dataset_key}_val_lhand_pck_20", lhand_pck_20, prog_bar=False)
        self.log(f"{dataset_key}_val_rhand_pck_auc_20_40", rhand_pck_auc_20_40, prog_bar=False)
        self.log(f"{dataset_key}_val_lhand_pck_auc_20_40", lhand_pck_auc_20_40, prog_bar=False)
        # エポックレベルの平均結果を保存
        self.mean_loss.append(loss.detach().cpu())
        self.mean_pck_20.append(rhand_pck_20)
        self.mean_pck_20.append(lhand_pck_20)
    
    def on_validation_epoch_end(self):
        # エポックの最後にメトリクスの平均を計算
        mean_epoch_loss = torch.stack(self.mean_loss).mean()
        mean_epoch_pck_20 = torch.stack(self.mean_pck_20).mean()
        self.log("val_loss", mean_epoch_loss, prog_bar=False)
        self.log("val_PCK_20", mean_epoch_pck_20, prog_bar=False)
        self.mean_loss.clear()
        self.mean_pck_20.clear()

    def configure_optimizers(self):
        toret = {}
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.use_onecycle_lr:
            lr_scheduler_config = dict(
                scheduler=torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, 
                    max_lr=self.lr,
                    total_steps=self.trainer.estimated_stepping_batches * 5,
                    pct_start=self.pct_start,
                    anneal_strategy='linear'
                )
            )
        toret['optimizer'] = optimizer
        if self.use_onecycle_lr:
            toret['lr_scheduler'] = lr_scheduler_config

        return toret