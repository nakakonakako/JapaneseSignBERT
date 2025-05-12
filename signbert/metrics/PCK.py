import torch
from torch import Tensor
from torch.nn import ModuleList
from torchmetrics import Metric

from IPython import embed; from sys import exit


class PCK(Metric):
    """
    PCK(Percentage of Correct Keypoints)メトリッククラス

    このクラスは、PCKメトリックを計算するためにPyTorchのMetricクラスを拡張します。
    PCKは、予測されたキーポイントのうち、真のキーポイントから一定のしきい値距離以内にある割合を測定します。
    
    Attributes:
    threshold (float): キーポイントが正しく予測されたと見なされる距離しきい値
    correct (Tensor): しきい値内に正しく予測されたキーポイントの数
    total (Tensor): 予測されたキーポイントの総数
    """
    def __init__(self, thr: float = 20.):
        """
        PCkメトリックを初期化
        
        Parameters:
        thr (float): キーポイントが正しく予測されたと見なされるしきい値。デフォルトは20
        """
        super().__init__()
        self.threshold = thr
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """
        新しい予測とターゲットでメトリックの状態を更新します。

        Parameters:
        preds (Tensor): 予測されたキーポイント
        target (Tensor): 真のキーポイント
        """
        assert preds.shape == target.shape
        # 予測とターゲットの間のL2距離を計算
        distances = torch.norm(target - preds, dim=-1)
        # しきい値距離内にある予測の数をカウント
        correct = (distances < self.threshold).sum()
        self.correct += correct
        self.total += distances.numel()

    def compute(self): 
        """
        PCK値を計算します。

        Returns:
        float: 正しく予測されたキーポイントの割合
        """
        return self.correct.float() / self.total


class PCKAUC(Metric):
    """
    PCK(Percentage of Correct Keypoints)メトリックのAUC(Area Under Curve)を計算するためのクラスです。

    このクラスは、一連のしきい値にわたるPCKメトリックのAUCを計算します。

    Attributes:
    metrics (ModuleList): 異なるしきい値を持つPCKメトリックのリスト
    thresholds (Tensor): PCKを計算するために使用されるしきい値のテンソル
    diff (Tensor): 最小しきい値と最大しきい値の差
    """
    def __init__(self, thr_min: float = 20, thr_max: float = 40):
        """
        PCKAUCメトリックを初期化します。

        Parameters:
        thr_min (float): PCK計算の最小しきい値。デフォルトは20です。
        thr_max (float): PCK計算の最大しきい値。デフォルトは40です。
        """
        super().__init__()
        assert thr_min < thr_max
        step = 1
        thresholds = torch.arange(thr_min, thr_max+step, step)
        # 各しきい値に対してPCKメトリックを作成
        self.metrics = ModuleList([PCK(thr) for thr in thresholds])
        self.add_state("thresholds", default=thresholds, dist_reduce_fx=None)
        self.add_state("diff", default=torch.tensor(thr_max-thr_min), dist_reduce_fx=None)
    
    def update(self, preds: Tensor, target: Tensor):
        """
        新しい予測とターゲットでメトリックの状態を更新します。

        Parameters:
        preds (Tensor): 予測されたキーポイント
        target (Tensor): 真のキーポイント
        """
        assert preds.shape == target.shape
        # 新しい予測とターゲットで各PCKメトリックを更新
        for m in self.metrics: m.update(preds, target)

    def compute(self):
        """
        PCKAUC値を計算します。

        Returns:
        Tensor: しきい値の範囲全体でのPCKメトリックのAUC
        """
        # 各しきい値のPCKを計算し、結果を連結
        result = torch.cat([m.compute().reshape(1) for m in self.metrics])
        # 曲線下の面積（AUC）を計算し、[0,1]の間に正規化します
        return torch.trapz(result, self.thresholds) / self.diff
    
    def reset(self):
        # 各PCKメトリックをリセット
        self._update_count = 0
        self._forward_cache = None
        self._computed = None
        # 中間状態をリセット
        self._cache = None
        self._is_synced = False
        for m in self.metrics: m.reset()
        self.auc = 0.


if __name__ == '__main__':

    gt = torch.rand(16, 2361, 21, 2)
    pred = torch.rand(16, 2361, 21, 2)

    pck = PCK()
    pck.update(gt, pred)
    print(f'{pck.compute()=}')

    pck_auc = PCKAUC()
    pck_auc.update(gt, pred)
    print(f'{pck_auc.compute()=}')
    embed(); exit()