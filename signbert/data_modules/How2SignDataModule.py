import os
import gc
import glob
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from signbert.data_modules.PretrainMaskKeypointDataset import PretrainMaskKeypointDataset, mask_keypoint_dataset_collate_fn
from signbert.utils import read_json

from sklearn.model_selection import train_test_split

from IPython import embed


class How2SignDataModule(pl.LightningDataModule):
    DPATH = '/home/user/kimlab/Nakanishi/SignBERT/SLR/How2Sign'
    SKELETON_DPATH = '/mnt/d-drive/user/kimlab/Nakanishi/npy/how2sign'
    PREPROCESS_DPATH = os.path.join(DPATH, 'preprocess')
    MEANS_FPATH = os.path.join(PREPROCESS_DPATH, 'means.npy')
    STDS_FPATH = os.path.join(PREPROCESS_DPATH, 'stds.npy')
    TRAIN_FPATH = os.path.join(PREPROCESS_DPATH, 'train.npy')
    VAL_FPATH = os.path.join(PREPROCESS_DPATH, 'val.npy')
    TRAIN_NORM_FPATH = os.path.join(PREPROCESS_DPATH, 'train_norm.npy')
    VAL_NORM_FPATH = os.path.join(PREPROCESS_DPATH, 'val_norm.npy')
    TRAIN_IDXS_FPATH = os.path.join(PREPROCESS_DPATH, 'train_idxs.npy')
    VAL_IDXS_FPATH = os.path.join(PREPROCESS_DPATH, 'val_idxs.npy')
    SEQ_PAD_VALUE = 0.0

    def __init__(self, batch_size, normalize=False, R=0.3, m=5, K=8, max_disturbance=0.25):
        super().__init__()
        self.batch_size = batch_size
        self.normalize = normalize
        self.R = R
        self.m = m
        self.K = K
        self.max_disturbance = max_disturbance
        self.means_fpath = How2SignDataModule.MEANS_FPATH
        self.stds_fpath = How2SignDataModule.STDS_FPATH

    def prepare_data(self):

        # もし How2SignDataModule.PREPROCESS_DPATH が存在しない場合は作成
        if not os.path.exists(How2SignDataModule.PREPROCESS_DPATH):
            os.makedirs(How2SignDataModule.PREPROCESS_DPATH)

        all_data_fpaths = glob.glob(os.path.join(How2SignDataModule.SKELETON_DPATH, "*.npy"))
        all_data = [np.load(f) for f in all_data_fpaths]
        
        train, val = train_test_split(all_data, test_size=0.25, random_state=42)
        
        # 平均値と標準偏差を計算
        if not os.path.exists(How2SignDataModule.MEANS_FPATH) or \
            not os.path.exists(How2SignDataModule.STDS_FPATH):
                self._generate_means_stds(train)
        # train, val それぞれの Numpy ファイルが存在しない場合は作成
        if not os.path.exists(How2SignDataModule.TRAIN_FPATH) or \
            not os.path.exists(How2SignDataModule.VAL_FPATH) or \
            not os.path.exists(How2SignDataModule.TRAIN_NORM_FPATH) or \
            not os.path.exists(How2SignDataModule.VAL_NORM_FPATH) or \
            not os.path.exists(How2SignDataModule.TRAIN_IDXS_FPATH) or \
            not os.path.exists(How2SignDataModule.VAL_IDXS_FPATH):

            self._generate_preprocess_npy_arrays(
                range(len(train)), 
                train, 
                How2SignDataModule.TRAIN_FPATH, 
                How2SignDataModule.TRAIN_NORM_FPATH,
                How2SignDataModule.TRAIN_IDXS_FPATH,
            )
            del train
            gc.collect()

            self._generate_preprocess_npy_arrays(
                range(len(val)), 
                val, 
                How2SignDataModule.VAL_FPATH, 
                How2SignDataModule.VAL_NORM_FPATH,
                How2SignDataModule.VAL_IDXS_FPATH,
            )
            del val
            gc.collect()
            
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            X_train_fpath = How2SignDataModule.TRAIN_NORM_FPATH if self.normalize else How2SignDataModule.TRAIN_FPATH
            X_val_fpath = How2SignDataModule.VAL_NORM_FPATH if self.normalize else How2SignDataModule.VAL_FPATH

            self.setup_train = PretrainMaskKeypointDataset(
                How2SignDataModule.TRAIN_IDXS_FPATH, 
                X_train_fpath, 
                self.R, 
                self.m, 
                self.K, 
                self.max_disturbance
            )
            self.setup_val = PretrainMaskKeypointDataset(
                How2SignDataModule.VAL_IDXS_FPATH,
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

    def _read_openpose_split(self, split_fpath):
        """Read Openpose prediction output file in parallel."""
        # split_fpath 内のすべての JSON ファイルを取得（サブディレクトリを無視）
        json_files = glob.glob(os.path.join(split_fpath, '*.json')) 
        executor = ProcessPoolExecutor(max_workers=os.cpu_count())
        futures = []
        
        # 各ファイルを並列で処理
        for json_file in json_files:
            future = executor.submit(self._read_openpose_json_out, json_file)
            futures.append(future)
        
        results = [f.result() for f in futures] 
        executor.shutdown()

        return results

    def _read_openpose_json_out(self, fpath):
        """Parse Openpose output from JSON to a Numpy array."""
        data = []
        
        # JSON ファイルを処理
        raw_data = read_json(fpath)['people'][0]
        face_kps = np.array(raw_data['face_keypoints_2d']).reshape(-1, 3)
        pose_kps = np.array(raw_data['pose_keypoints_2d']).reshape(-1, 3)
        lhand_kps = np.array(raw_data['hand_left_keypoints_2d']).reshape(-1, 3)
        rhand_kps = np.array(raw_data['hand_right_keypoints_2d']).reshape(-1, 3)
        kps = np.concatenate((face_kps, pose_kps, lhand_kps, rhand_kps))
        data.append(kps)
        
        # Numpy 配列を返す
        data = np.stack(data)
        return data

    def _generate_means_stds(self, train_data):
        """Compute mean and standard deviation for all x and y coordinates."""
        seq_concats = np.concatenate([s[..., :2] for s in train_data], axis=0)
        means = seq_concats.mean((0, 1))
        stds = seq_concats.std((0, 1))
        np.save(How2SignDataModule.MEANS_FPATH, means)
        np.save(How2SignDataModule.STDS_FPATH, stds)

    def _generate_preprocess_npy_arrays(
            self, 
            split_idxs, 
            skeleton_fpaths, 
            out_fpath,
            norm_out_fpath,
            idxs_out_fpath,
            max_seq_len=500
        ):
        """
        学習中に使用できるように、生のスケルトンシーケンスを処理します。
                
        この関数は、スケルトンファイルパスのリストを受け取り、それらを処理し、
        トレーニング中に簡単にアクセスできるように .npy 形式のファイルに結果を保存します。

        Parameters:
        split_idxs (list): 現在の実装では使用されません。
        skeleton_fpaths (list): 生のスケルトンシーケンスを含むファイルパスのリスト。
        out_fpath (str): 処理されたシーケンスを保存するファイルパス。
        norm_out_fpath (str): 正規化されたシーケンスを保存するファイルパス。
        idxs_out_fpath (str): シーケンスのインデックスを保存するファイルパス。
        max_seq_len (int): シーケンスをより小さなシーケンスに分割する前の最大長。デフォルトは 500 です。
        """
        # 処理されたシーケンスを保存するための空のリストを初期化
        seqs = []
        # 各スケルトンファイルパスに対して繰り返す
        for seq in skeleton_fpaths:
            # 最大シーケンス長を超えるかどうかを確認
            if seq.shape[0] > max_seq_len:
                # シーケンスを最大シーケンス長を超えないように分割するためのインデックスを計算
                split_indices = list(range(max_seq_len, seq.shape[0], max_seq_len))
                # 計算されたインデックスに基づいてシーケンスをより小さなシーケンスに分割
                seq = np.array_split(seq, split_indices, axis=0)
                # 分割されたシーケンスをリストに追加
                for s in seq:
                    seqs.append(s)
            else:
                # 最大長を超えない場合は、シーケンスをそのまま追加
                seqs.append(seq)
        # シーケンスのインデックスを生成
        seqs_idxs = range(len(seqs)) 
        # 正規化されたシーケンスを生成
        seqs_norm = self._normalize_seqs(seqs)
        # 同じ最大長になるようにシーケンスをパディング
        seqs = self._pad_seqs_by_max_len(seqs)
        seqs_norm = self._pad_seqs_by_max_len(seqs_norm)
        seqs = seqs.astype(np.float32)
        seqs_norm = seqs_norm.astype(np.float32)
        seqs_idxs = np.array(seqs_idxs, dtype=np.int32)
        # それぞれのファイルパスに処理されたシーケンス、正規化されたシーケンス、インデックスを保存
        np.save(out_fpath, seqs)
        np.save(norm_out_fpath, seqs_norm)
        np.save(idxs_out_fpath, seqs_idxs)
        # 大きな変数を削除し、ガベージコレクションを呼び出すことでメモリを解放
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
        means = np.load(How2SignDataModule.MEANS_FPATH)
        stds = np.load(How2SignDataModule.STDS_FPATH)

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
                constant_values=How2SignDataModule.SEQ_PAD_VALUE
            ) 
            for i, t in enumerate(seqs)
        ])

        return seqs


if __name__ == '__main__':

    d = How2SignDataModule(
        batch_size=32,
        normalize=True,
    )
    d.prepare_data()
    d.setup()
    dl = d.train_dataloader()
    sample = next(iter(dl))
    embed()
