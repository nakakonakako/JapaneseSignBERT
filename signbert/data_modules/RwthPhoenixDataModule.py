import os
import gc
import glob

import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from signbert.data_modules.PretrainMaskKeypointDataset import PretrainMaskKeypointDataset, mask_keypoint_dataset_collate_fn

from IPython import embed


class RwthPhoenixDataModule(pl.LightningDataModule):
    DPATH = '/mnt/d-drive/user/kimlab/Nakanishi/npy/phoenix'
    DPATH_T = '/mnt/d-drive/user/kimlab/Nakanishi/npy/ja'
    PREPROCESS_DPATH = '/home/user/kimlab/Nakanishi/SignBERT/SLR/phoenix'
    PREPROCESS_DPATH_T = '/home/user/kimlab/Nakanishi/SignBERT/SLR/ja'
    TRAIN_DPATH = os.path.join(DPATH, 'train')
    DEV_DPATH = os.path.join(DPATH, 'val')
    TRAIN_DPATH_T = os.path.join(DPATH_T, 'train')
    DEV_DPATH_T = os.path.join(DPATH_T, 'val')
    SEQ_PAD_VALUE = 0.0

    def __init__(self, batch_size, normalize=False, R=0.3, m=5, K=8, max_disturbance=0.25, ja=False):
        super().__init__()
        self.batch_size = batch_size
        self.normalize = normalize
        self.R = R
        self.m = m
        self.K = K
        self.max_disturbance = max_disturbance
        self.ja  = ja 
        self.dpath = RwthPhoenixDataModule.DPATH_T if ja else RwthPhoenixDataModule.DPATH
        self.train_dpath = RwthPhoenixDataModule.TRAIN_DPATH_T if ja else RwthPhoenixDataModule.TRAIN_DPATH
        self.dev_dpath = RwthPhoenixDataModule.DEV_DPATH_T if ja else RwthPhoenixDataModule.DEV_DPATH
        self.preprocess_dpath = RwthPhoenixDataModule.PREPROCESS_DPATH_T if ja else RwthPhoenixDataModule.PREPROCESS_DPATH 
        self.train_fpath = os.path.join(self.preprocess_dpath, 'train.npy')
        self.val_fpath = os.path.join(self.preprocess_dpath, 'val.npy')
        self.train_norm_fpath = os.path.join(self.preprocess_dpath, 'train_norm.npy')
        self.val_norm_fpath = os.path.join(self.preprocess_dpath, 'val_norm.npy')
        self.train_idxs_fpath = os.path.join(self.preprocess_dpath, 'train_idxs.npy')
        self.val_idxs_fpath = os.path.join(self.preprocess_dpath, 'val_idxs.npy')
        self.means_fpath = os.path.join(self.preprocess_dpath, 'means.npy')
        self.stds_fpath = os.path.join(self.preprocess_dpath, 'stds.npy')

    def prepare_data(self):
        # もし存在しない場合は前処理パスを作成
        if not os.path.exists(self.preprocess_dpath):
            os.makedirs(self.preprocess_dpath)
        
        # 学習キーポイントのxとyの平均を計算
        if not os.path.exists(self.means_fpath) or \
            not os.path.exists(self.stds_fpath):
            self._generate_train_means_stds()
            
        # train、validation、テストのNumpy配列が存在するかどうかを確認
        if not os.path.exists(self.train_fpath) or \
            not os.path.exists(self.train_norm_fpath) or \
            not os.path.exists(self.val_fpath) or \
            not os.path.exists(self.val_norm_fpath):
            self._generate_preprocess_npy_arrays(
                self.train_dpath, 
                self.train_fpath, 
                self.train_norm_fpath
            )
            self._generate_preprocess_npy_arrays(
                self.dev_dpath, 
                self.val_fpath, 
                self.val_norm_fpath
            )

        
        # インデックスNumpy配列が存在するかどうかを確認
        if not os.path.exists(self.train_idxs_fpath) or \
            not os.path.exists(self.val_idxs_fpath):
            self._generate_idxs()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            X_train_fpath = self.train_norm_fpath if self.normalize else self.train_fpath
            X_val_fpath = self.val_norm_fpath if self.normalize else self.val_fpath

            self.setup_train = PretrainMaskKeypointDataset(
                self.train_idxs_fpath, 
                X_train_fpath, 
                self.R, 
                self.m, 
                self.K, 
                self.max_disturbance
            )
            self.setup_val = PretrainMaskKeypointDataset(
                self.val_idxs_fpath,
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
    
    def _generate_idxs(self):
        train_idxs = np.arange(len(np.load(self.train_fpath)))
        val_idxs = np.arange(
            start=len(train_idxs), 
            stop=len(train_idxs) + len(np.load(self.val_fpath))
        )

        np.save(self.train_idxs_fpath, train_idxs)
        np.save(self.val_idxs_fpath, val_idxs)
    
    def _generate_train_means_stds(self):
        """Compute mean and standard deviation for all x and y coordinates."""
        npy_files = glob.glob(os.path.join(self.train_dpath, '*.npy'))
        npy_concats = np.concatenate([np.load(f)[...,:2] for f in npy_files])
        means = np.mean(npy_concats, axis=(0,1))
        stds = np.std(npy_concats, axis=(0,1))
        np.save(self.means_fpath, means)
        np.save(self.stds_fpath, stds)

    def _generate_preprocess_npy_arrays(self, dpath, out_fpath, norm_out_fpath):
        """
        numpy形式のデータのシーケンスを処理して保存します。

        ディスクから生のシーケンスをロードし、それらを平均と標準偏差で正規化し、
        一定の長さにパディングし、最後に処理されたシーケンスをディスクに保存します。

        Parameters:
        dpath (str): 生のシーケンスが保存されているディレクトリパス。
        out_fpath (str): 処理された（ただし正規化されていない）シーケンスを保存するファイルパス。
        norm_out_fpath (str): 正規化されたシーケンスを保存するファイルパス。
        """
        # ディレクトリから生のシーケンスをロード
        seqs = self._load_raw_seqs(dpath)
        # 平均と標準偏差でシーケンスを正規化
        seqs_norm = self._normalize_seqs(seqs)
        # パディングして、すべてのシーケンスが同じ長さにする
        seqs = self._pad_seqs_by_max_len(seqs)
        seqs_norm = self._pad_seqs_by_max_len(seqs_norm)
        # 保存されたファイルは、後でデータモジュールのセットアップで使用
        np.save(out_fpath, seqs)
        np.save(norm_out_fpath, seqs_norm)
        # ガベージコレクションを呼び出して、大きなシーケンス変数を削除し、メモリを解放します
        del seqs
        del seqs_norm
        gc.collect()
 
    def _load_raw_seqs(self, dpath):
        """
        生のシーケンスを指定されたディレクトリの.npyファイルからロードします。

        この関数は、指定されたディレクトリ内の.npyファイルをスキャンし、
        各ファイルからデータをリストにロードします。

        Parameters:
        dpath (str): .npyファイルが保存されているディレクトリパス。

        Returns:
        list: .npyファイルからロードされた各配列を含むリスト。
        """
        # 特定のディレクトリ内のすべての.npyファイルを見つけるためにglobを使用
        npy_files = glob.glob(os.path.join(dpath, '*.npy'))
        # .npyファイルをロードし、その内容をリストに追加します
        seqs = [np.load(f) for f in npy_files]

        return seqs

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
        means = np.load(self.means_fpath)
        stds = np.load(self.stds_fpath)

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
                constant_values=RwthPhoenixDataModule.SEQ_PAD_VALUE
            ) 
            for i, t in enumerate(seqs)
        ])

        return seqs

if __name__ == '__main__':

    d = RwthPhoenixDataModule(
        32,
        True,
        phoenix_T=False
    )
    d.prepare_data()
    d.setup()
    dl = d.train_dataloader()
    sample = next(iter(dl))
    embed()
