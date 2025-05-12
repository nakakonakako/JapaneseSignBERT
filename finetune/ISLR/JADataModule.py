import os
import re

import torch
import numpy as np
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


import pandas as pd

class JADataModule(pl.LightningDataModule):
    """
    日本語データセット用のPyTorch Lightning DataModule。

    このクラスは、日本語データセットのトレーニング、バリデーション、テスト用のデータローディングと前処理を処理します。
    さまざまなデータファイルやディレクトリへのパスを整理し、モデルのためのデータローダーをセットアップします。
    
    Class Attributes:
    データのディレクトリやファイルへのパスを定義するクラス属性。
    VIDEO_ID_PATTERN: 動画IDを抽出するための正規表現パターン。
    PADDING_VALUE: シーケンスのパディングに使用される値。

    Instance Attributes:
    batch_size (int): データローダーのバッチサイズ。
    normalize (bool): データを正規化するかどうかを示すフラグ。
    """
    # データファイルやディレクトリへのパスを定義するクラス属性
    DPATH = '/mnt/d-drive/user/kimlab/Nakanishi/npy/ja'
    PREPROCESS_DPATH = '/home/user/kimlab/Nakanishi/SignBERT/SLR/ja'
    SKELETON_DPATH = DPATH
    CLASSES_JSON_FPATH = '/home/user/kimlab/Nakanishi/SignBERT/finetune/ISLR/wordslist.json'
    TRAIN_SKELETON_DPATH = os.path.join(SKELETON_DPATH, 'train')
    VAL_SKELETON_DPATH = os.path.join(SKELETON_DPATH, 'val')
    MEANS_FPATH = os.path.join(PREPROCESS_DPATH, 'means.npy')
    STDS_FPATH = os.path.join(PREPROCESS_DPATH, 'stds.npy')
    VIDEO_ID_PATTERN = r"(?<=v\=).{11}"
    PADDING_VALUE = 0.0

    def __init__(self, batch_size, normalize):
        """
        JADataModuleを初期化

        Parameters:
        batch_size (int): データローダーのバッチサイズ。
        normalize (bool): データを正規化するかどうかを示すフラグ。
        """
        super().__init__()
        self.batch_size = batch_size
        self.normalize = normalize

    def setup(self, stage):
        """
        ステージ（'fit'または'test'）に対してデータセットを準備します。

        Parameters:
        stage (str): 通常はトレーニングとバリデーションのための 'fit'
        """
        # クラスラベルをJSONファイルから読み込む
        classes = pd.read_json(JADataModule.CLASSES_JSON_FPATH, encoding='utf-8')
        classes = classes["単語"].tolist()
        # テキストファイルから欠落している動画IDのリストを読み込む
        if stage == "fit":
            train_list = [ 
                os.path.splitext(f)[0] 
                for f in os.listdir(JADataModule.TRAIN_SKELETON_DPATH) 
                if f.endswith('.npy') and os.path.isfile(os.path.join(JADataModule.TRAIN_SKELETON_DPATH, f)) 
            ]
            train_list = [
                item 
                for item in train_list
                if not (match := re.search(r"_(.*?)[\s・_]", item)) or not match.group(1).startswith("A")
            ]
            
            val_list = [
                os.path.splitext(f)[0] 
                for f in os.listdir(JADataModule.VAL_SKELETON_DPATH) 
                if f.endswith('.npy') and os.path.isfile(os.path.join(JADataModule.VAL_SKELETON_DPATH, f)) 
            ]
            val_list = [
                item 
                for item in val_list
                if not (match := re.search(r"_(.*?)[\s・_]", item)) or not match.group(1).startswith("A")
            ]

            self.train_dataset = JADataset(
                train_list, 
                classes,
                JADataModule.TRAIN_SKELETON_DPATH, 
                self.normalize,
                np.load(JADataModule.MEANS_FPATH),
                np.load(JADataModule.STDS_FPATH)
            )
            self.val_dataset = JADataset(
                val_list, 
                classes,
                JADataModule.VAL_SKELETON_DPATH, 
                self.normalize,
                np.load(JADataModule.MEANS_FPATH),
                np.load(JADataModule.STDS_FPATH)
            )

    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=True,
            collate_fn=my_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            collate_fn=my_collate_fn
        )


class JADataset(Dataset):
    """
    日本語データセットのデータを処理するためのPyTorchデータセット。

    このクラスは、各サンプルのスケルトンデータをロードし、必要に応じて正規化し、
    腕、左手、右手のキーポイントなどの特定の特徴を抽出します。

    Attributes:
    train_info (list): 各サンプルの情報を含む辞書のリスト。
    skeleton_dpath (str): スケルトンデータファイルが含まれるディレクトリへのパス。
    normalize (bool): データを正規化するかどうかを示すフラグ。
    normalize_mean (numpy.ndarray or None): 正規化のための平均値。
    normalize_std (numpy.ndarray or None): 正規化のための標準偏差。
    """
    def __init__(self, videotitle_info, classes, skeleton_dpath, normalize, normalize_mean=None, normalize_std=None):
        """
        JADataModuleを初期化

        Parameters:
        train_info (list): 各トレーニングサンプルの情報。
        skeleton_dpath (str): スケルトンデータファイルが含まれるディレクトリへのパス。
        normalize (bool): データを正規化するかどうかを示すフラグ。
        normalize_mean (numpy.ndarray, optional): 正規化のための平均値。 
        normalize_std (numpy.ndarray, optional): 正規化のための標準偏差。
        """
        super().__init__()
        self.video_info = videotitle_info
        self.classes = classes
        self.skeleton_dpath = skeleton_dpath
        self.normalize = normalize
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def __len__(self):
        """Returns the number of samples in the dataset."""
        """データセット内のサンプル数を返す。"""
        return len(self.video_info)

    def __getitem__(self, idx):
        """
        特定のインデックスのデータセットからサンプルを取得します。

        Parameters:
        idx (int): 取得するサンプルのインデックス。

        Returns:
        dict: サンプルデータを含む辞書。
        """
        sample = self.video_info[idx]
        match = re.search(r"_(.*?)[\s・_]", sample)
        class_id = self.classes.index(match.group(1))

        skeleton_video_fpath = os.path.join(self.skeleton_dpath, f"{sample}.npy")
        skeleton_data = np.load(skeleton_video_fpath) #[start_video:end_video]
        # スコア列が存在する場合は削除し、必要に応じてデータを正規化
        skeleton_data = skeleton_data[...,:2]
        if self.normalize:
            skeleton_data = (skeleton_data - self.normalize_mean) / self.normalize_std
            
        skeleton_data = skeleton_data.astype(np.float32)
        
        # 特定の特徴を抽出：腕、左手、右手のキーポイント
        arms = skeleton_data[:, 5:11]
        lhand = skeleton_data[:, 91:112]
        rhand = skeleton_data[:, 112:133]
        
        return {
            "sample_id": idx,
            "class_id": class_id,
            "arms": arms, 
            "lhand": lhand,
            "rhand": rhand
        }


def my_collate_fn(original_batch):
    """Custom collate DataLoader function."""
    sample_id = []
    class_id = []
    arms = []
    lhand = []
    rhand = []
    for ob in original_batch:
        sample_id.append(ob["sample_id"])
        class_id.append(ob["class_id"])
        arms.append(torch.from_numpy(ob["arms"]))
        lhand.append(torch.from_numpy(ob["lhand"]))
        rhand.append(torch.from_numpy(ob["rhand"]))
    arms = pad_sequence(arms, batch_first=True, padding_value=JADataModule.PADDING_VALUE)
    lhand = pad_sequence(lhand, batch_first=True, padding_value=JADataModule.PADDING_VALUE)
    rhand = pad_sequence(rhand, batch_first=True, padding_value=JADataModule.PADDING_VALUE)
    class_id = torch.tensor(class_id, dtype=torch.int64)
    sample_id = torch.tensor(sample_id, dtype=torch.int32)

    return {
        "sample_id": sample_id,
        "class_id": class_id,
        "arms": arms, 
        "lhand": lhand,
        "rhand": rhand
    }
