import os
import gc
import glob

import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from signbert.data_modules.PretrainMaskKeypointDataset import PretrainMaskKeypointDataset, mask_keypoint_dataset_collate_fn
from signbert.utils import dict_to_json_file

from IPython import embed


class MSASLDataModule(pl.LightningDataModule):
    DPATH = '/home/user/kimlab/Nakanishi/SignBERT/SLR/MSASL'
    SKELETON_DPATH = '/mnt/d-drive/user/kimlab/Nakanishi/npy/msasl'
    MISSING_VIDEOS_FPATH = os.path.join(DPATH, 'raw_videos', 'missing.txt')
    TRAIN_SKELETON_DPATH = os.path.join(SKELETON_DPATH, 'train')
    VAL_SKELETON_DPATH = os.path.join(SKELETON_DPATH, 'val')
    PREPROCESS_DPATH = os.path.join(DPATH, 'preprocess')
    MEANS_FPATH = os.path.join(PREPROCESS_DPATH, 'means.npy')
    STDS_FPATH = os.path.join(PREPROCESS_DPATH, 'stds.npy')
    TRAIN_FPATH = os.path.join(PREPROCESS_DPATH, 'train.npy')
    VAL_FPATH = os.path.join(PREPROCESS_DPATH, 'val.npy')
    TRAIN_NORM_FPATH = os.path.join(PREPROCESS_DPATH, 'train_norm.npy')
    VAL_NORM_FPATH = os.path.join(PREPROCESS_DPATH, 'val_norm.npy')
    TRAIN_IDXS_FPATH = os.path.join(PREPROCESS_DPATH, 'train_idxs.npy')
    VAL_IDXS_FPATH = os.path.join(PREPROCESS_DPATH, 'val_idxs.npy')
    TRAIN_MAPPING_IDXS_FPATH = os.path.join(PREPROCESS_DPATH, 'train_mapping_idxs.json')
    VAL_MAPPING_IDXS_FPATH = os.path.join(PREPROCESS_DPATH, 'val_mapping_idxs.json')
    SEQ_PAD_VALUE = 0.0

    def __init__(self, batch_size, normalize=False, R=0.3, m=5, K=8, max_disturbance=0.25):
        super().__init__()
        self.batch_size = batch_size
        self.normalize = normalize
        self.R = R
        self.m = m
        self.K = K
        self.max_disturbance = max_disturbance
        self.means_fpath = MSASLDataModule.MEANS_FPATH
        self.stds_fpath = MSASLDataModule.STDS_FPATH

    def prepare_data(self):
        # もし存在しない場合は、前処理パスを作成
        if not os.path.exists(MSASLDataModule.PREPROCESS_DPATH):
            os.makedirs(MSASLDataModule.PREPROCESS_DPATH)

        # もし存在しない場合は、平均と標準偏差を計算
        if not os.path.exists(MSASLDataModule.MEANS_FPATH) or \
            not os.path.exists(MSASLDataModule.STDS_FPATH):
            
            train_skeleton_fpaths = glob.glob(os.path.join(MSASLDataModule.TRAIN_SKELETON_DPATH, "*.npy"))
            train = [np.load(f) for f in train_skeleton_fpaths]
            self._generate_means_stds(train)
        
        if not os.path.exists(MSASLDataModule.TRAIN_FPATH) or \
            not os.path.exists(MSASLDataModule.VAL_FPATH) or \
            not os.path.exists(MSASLDataModule.TRAIN_NORM_FPATH) or \
            not os.path.exists(MSASLDataModule.VAL_NORM_FPATH) or \
            not os.path.exists(MSASLDataModule.TRAIN_IDXS_FPATH) or \
            not os.path.exists(MSASLDataModule.VAL_IDXS_FPATH) or \
            not os.path.exists(MSASLDataModule.TRAIN_MAPPING_IDXS_FPATH) or \
            not os.path.exists(MSASLDataModule.VAL_MAPPING_IDXS_FPATH):

            # ファイルパスを取得し、欠損をフィルタリング
            train_skeleton_fpaths = glob.glob(
                os.path.join(MSASLDataModule.TRAIN_SKELETON_DPATH, '*.npy')
            )
            train_idxs = [os.path.basename(f).split('.npy')[0] for f in train_skeleton_fpaths]
            train_skeleton_fpaths = [
                f 
                for f in train_skeleton_fpaths 
            ]  
            train_idxs = [idx for idx in train_idxs]
            val_skeleton_fpaths = glob.glob(
                os.path.join(MSASLDataModule.VAL_SKELETON_DPATH, '*.npy')
            )
            val_idxs = [os.path.basename(f).split('.npy')[0] for f in val_skeleton_fpaths]
            val_skeleton_fpaths = [
                f 
                for f in val_skeleton_fpaths 
            ]  
            val_idxs = [idx for idx in train_idxs]

            # Generate Numpy
            self._generate_preprocess_npy_arrays(
                train_idxs, 
                train_skeleton_fpaths, 
                MSASLDataModule.TRAIN_FPATH, 
                MSASLDataModule.TRAIN_NORM_FPATH,
                MSASLDataModule.TRAIN_IDXS_FPATH,
                MSASLDataModule.TRAIN_MAPPING_IDXS_FPATH
            )
            self._generate_preprocess_npy_arrays(
                val_idxs, 
                val_skeleton_fpaths, 
                MSASLDataModule.VAL_FPATH, 
                MSASLDataModule.VAL_NORM_FPATH,
                MSASLDataModule.VAL_IDXS_FPATH,
                MSASLDataModule.VAL_MAPPING_IDXS_FPATH
            )

            
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            X_train_fpath = MSASLDataModule.TRAIN_NORM_FPATH if self.normalize else MSASLDataModule.TRAIN_FPATH
            X_val_fpath = MSASLDataModule.VAL_NORM_FPATH if self.normalize else MSASLDataModule.VAL_FPATH

            self.setup_train = PretrainMaskKeypointDataset(
                MSASLDataModule.TRAIN_IDXS_FPATH, 
                X_train_fpath, 
                self.R, 
                self.m, 
                self.K, 
                self.max_disturbance
            )
            self.setup_val = PretrainMaskKeypointDataset(
                MSASLDataModule.VAL_IDXS_FPATH,
                X_val_fpath, 
                self.R, 
                self.m, 
                self.K, 
                self.max_disturbance
            )

    def train_dataloader(self):
        return DataLoader(self.setup_train, batch_size=self.batch_size, collate_fn=mask_keypoint_dataset_collate_fn, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.setup_val, batch_size=self.batch_size, collate_fn=mask_keypoint_dataset_collate_fn)

    def _generate_means_stds(self, train_data):
        """Compute mean and standard deviation for all x and y coordinates."""
        seq_concats = np.concatenate([s[..., :2] for s in train_data], axis=0)
        means = seq_concats.mean((0, 1))
        stds = seq_concats.std((0, 1))
        np.save(MSASLDataModule.MEANS_FPATH, means)
        np.save(MSASLDataModule.STDS_FPATH, stds)

    def _generate_preprocess_npy_arrays(
        self, 
        split_idxs, 
        skeleton_fpaths, 
        out_fpath,
        norm_out_fpath,
        idxs_out_fpath,
        idxs_mapping_out_fpath,
        max_seq_len=500
    ):
        """
        シーケンスデータを処理して保存します

        この関数は、シーケンスが最大長を超える場合の分割、正規化、一様な長さへのパディング、
        および処理されたシーケンスを元のインデックスにマッピングすることを処理し、
        それらの処理されたシーケンスを効率的なアクセスのために.npy形式で保存します。

        Parameters:
        split_idxs (list): シーケンスを分割する位置を示すインデックス
        skeleton_fpaths (list): 生のスケルトンシーケンスのファイルパス
        out_fpath (str): 処理されたシーケンスを保存するファイルパス
        norm_out_fpath (str): 正規化されたシーケンスを保存するファイルパス
        idxs_mapping_out_fpath (str): マッピングインデックスを保存するファイルパス
        idxs_out_fpath (str): シーケンスインデックスを保存するファイルパス
        max_seq_len (int): 分割前のシーケンスの最大長。デフォルトは500です。
        """
        seqs = []
        # 前のシーケンスの順序を追跡
        sequential_idx = [] 
        # 新しいインデックスを元のシーケンスインデックスにマッピング
        counter = 0  
        # マッピングインデックスを保存する辞書
        mapping_idxs = {}  
        # ファイルパスとそれに対応するインデックスを反復処理
        for idx, f in zip(split_idxs, skeleton_fpaths):
            seq = np.load(f)
            # もしシーケンスが最大長を超えて分割が必要な場合
            if seq.shape[0] > max_seq_len:
                split_indices = list(range(max_seq_len, seq.shape[0], max_seq_len))
                seq = np.array_split(seq, split_indices, axis=0)
                for s in seq:
                    seqs.append(s)
                    sequential_idx.append(counter)
                    # 新しいインデックスを元のインデックスにマッピング
                    mapping_idxs[counter] = idx  
                    counter += 1
            else:
                seqs.append(seq)
                sequential_idx.append(counter)
                mapping_idxs[counter] = idx
                counter += 1
        # 正規化されたシーケンスを取得
        seqs_norm = self._normalize_seqs(seqs)
        # シーケンスを一様な長さにパディング
        seqs = self._pad_seqs_by_max_len(seqs)
        seqs_norm = self._pad_seqs_by_max_len(seqs_norm)
        # シーケンスをfloat32形式に変換
        seqs = seqs.astype(np.float32)
        seqs_norm = seqs_norm.astype(np.float32)
        # シーケンスインデックスをint32形式に変換
        seqs_idxs = np.array(sequential_idx, dtype=np.int32)
        # 保存されたファイルは.npy形式で、効率的なアクセスを提供します
        np.save(out_fpath, seqs)
        np.save(norm_out_fpath, seqs_norm)
        np.save(idxs_out_fpath, seqs_idxs)
        dict_to_json_file(mapping_idxs, idxs_mapping_out_fpath)  # Save the mapping as a JSON file
        # メモリを解放するためにクリーンアップ
        del seqs
        del seqs_norm
        del seqs_idxs
        gc.collect()

    
    def _normalize_seqs(self, seqs):
        """
        事前計算された平均値と標準偏差を使用してシーケンスを正規化します。

        この関数は、提供されたシーケンスのリスト（seqs）の各シーケンスを、各要素について平均を引いて標準偏差で割ることで正規化します。
        これは、多くの機械学習タスクで一般的な前処理ステップであり、データを平均が 0 で標準偏差が 1 に標準化します。
        
        Parameters:
        seqs (list): 正規化されたシーケンスのリスト。
        
        Returns:
        list: 正規化されたシーケンスのリスト。
        """
        # 正規化のために事前計算された平均値と標準偏差をロード
        means = np.load(MSASLDataModule.MEANS_FPATH)
        stds = np.load(MSASLDataModule.STDS_FPATH)

        means = np.concatenate((means, [0]), -1)
        stds = np.concatenate((stds, [1]), -1)
        # リスト内の各シーケンスを正規化
        # 正規化は、各要素について平均を引いて標準偏差で割ることによって行われる
        seqs_norm = [(s - means) / stds for s in seqs]

        return seqs_norm

    def _pad_seqs_by_max_len(self, seqs):
        """
        同じ最大長になるように、リスト内のすべてのシーケンスをパディングします。

        この関数は、リスト内の各シーケンスをパディングして、すべてが同じ長さになるようにします。
        これは、機械学習モデルのバッチ処理に特に有用であり、すべての入力が同じサイズであることを要求します。

        Parameters:
        seqs (list): パディングされるシーケンスのリスト（numpy 配列）。

        Returns:
        numpy.ndarray: 同じ最大長になるようにパディングされたすべてのシーケンスの numpy 配列。
        """
        # リスト内の各シーケンスの長さを計算
        seqs_len = [len(t) for t in seqs]
        # 最大シーケンス長を見つける
        max_seq_len = max(seqs_len)
        # パディング構成を生成するためのラムダ関数を定義します
        # この関数は、シーケンス長次元にのみパディングが適用され、定数値で埋められます
        lmdb_gen_pad_seq = lambda s_len: ((0,max_seq_len-s_len), (0,0), (0,0))
        # リスト内の各シーケンスをパディングします
        # パディングはシーケンス長次元にのみ適用され、定数値で埋められます
        seqs = np.stack([
            np.pad(
                array=t, 
                pad_width=lmdb_gen_pad_seq(seqs_len[i]),
                mode='constant',
                constant_values=MSASLDataModule.SEQ_PAD_VALUE
            ) 
            for i, t in enumerate(seqs)
        ])

        return seqs


if __name__ == '__main__':

    d = MSASLDataModule(
        batch_size=32,
        normalize=True,
    )
    d.prepare_data()
    d.setup()
    dl = d.train_dataloader()
    sample = next(iter(dl))
    embed()
