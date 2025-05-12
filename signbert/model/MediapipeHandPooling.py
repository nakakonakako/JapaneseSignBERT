import torch
from torch import nn


class MediapipeHandPooling(nn.Module):
    """
    MediaPipeのハンドトラッキング構成に従って手のキーポイントをプーリングするためのPyTorchモジュール

    このモジュールは、手の異なる部分（手のひら、親指、人差し指など）に対応するキーポイントをプールし、
    MediaPipeの21の手のキーポイント形式と連携するように設計されています。

    Attributes:
    PALM_IDXS, THUMB_IDXS, INDEX_IDXS, MIDDLE_IDXS, RING_IDXS, PINKY_IDXS: 手の部位ごとのキーポイントのインデックスを含むタプル
    """
    # MediaPipeのキーポイントに従って手の異なる部分のインデックスを定義
    PALM_IDXS = (0, 1, 5, 9, 13, 17)
    THUMB_IDXS = (2, 3, 4)
    INDEX_IDXS = (6, 7, 8)
    MIDDLE_IDXS = (10, 11, 12)
    RING_IDXS = (14, 15, 16)
    PINKY_IDXS = (18, 19, 20)

    def __init__(self, last=False):
        """
        Initialize the pooling module.
        プーリングモジュールを初期化

        Parameters:
        last (bool): Trueの場合、以前のプーリングが行われたため、新しいプーリングは最後の次元のみに適用されます。
        デフォルトはFalseです。
        """
        super().__init__()
        self.last = last

    def forward(self, x):
        """
        Apply pooling operation to hand keypoints.
        手のキーポイントにプーリング操作を適用
        
        Parameters:
        x (Tensor): 手のキーポイントを含んだ入力テンソル

        Returns:
        Tensor: プーリングされた手のキーポイントを含んだテンソル
        """
        if self.last: # もし以前のプーリングが行われた場合
            # 6つのクラスターのみがあることを確認
            assert x.shape[3] == 6
            return torch.amax(x, 3, keepdim=True)
        else:
            # Ensure that there are 21 hand keypoints
            # 手のキーポイントが21個あることを確認
            assert x.shape[3] == 21
            # キーポイントのグループ（手のひら、親指、指）ごとにmax poolingを適用します。合計6つ
            return torch.cat((
                torch.amax(x[:, :, :, MediapipeHandPooling.PALM_IDXS], 3, keepdim=True),
                torch.amax(x[:, :, :, MediapipeHandPooling.THUMB_IDXS], 3, keepdim=True),
                torch.amax(x[:, :, :, MediapipeHandPooling.INDEX_IDXS], 3, keepdim=True),
                torch.amax(x[:, :, :, MediapipeHandPooling.MIDDLE_IDXS], 3, keepdim=True),
                torch.amax(x[:, :, :, MediapipeHandPooling.RING_IDXS], 3, keepdim=True),
                torch.amax(x[:, :, :, MediapipeHandPooling.PINKY_IDXS], 3, keepdim=True),
            ), dim=3)