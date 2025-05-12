import numpy as np

def mask_transform_identity(seq, R, max_disturbance, no_mask_joint, K, m):
    """
    連続したフレームに対して異なる種類のマスキング変換を適用します。    

    この関数は、与えられたフレームシーケンスに対していくつかのマスキング操作のうちの1つをランダムに適用します。
    マスキングの種類には、ジョイントマスキング、フレームマスキング、クリップマスキング、およびアイデンティティ（変更なし）が含まれます。
    
    Parameters:
    seq (numpy.ndarray): マスキングを適用するフレームのシーケンス。    

    Returns:
    tuple:
        - numpy.ndarray: マスキングが適用された変換されたシーケンス。
        - numpy.ndarray: マスクが適用されたフレームのインデックス。
    """
    # 元のシーケンスを変更しないために、入力シーケンスのコピーを作成します
    toret = seq.copy()
    # マスキングされていないフレームの数を計算します
    n_frames = (toret != 0.0).all((1,2)).sum()
    # 事前に定義された比率Rに基づいてマスクするフレームの合計数を計算します
    n_frames_to_mask = int(np.ceil(R * n_frames))
    # マスクするフレームのインデックスをランダムに選択します
    frames_to_mask = np.random.choice(n_frames, size=n_frames_to_mask, replace=False)
    clipped_masked_frames = []
    for f in frames_to_mask:
        # マスクするフレームを取得します
        curr_frame = toret[f]
        # マスキング操作のタイプをランダムに選択します
        op_idx = np.random.choice(4) # 0: joint, 1: frame, 2: clip, 3: identity
        
        if op_idx == 0:
            # ジョイントマスキングを適用します
            curr_frame = mask_joint(curr_frame, max_disturbance, no_mask_joint, m)
            toret[f] = curr_frame
        elif op_idx == 1:
            # フレームマスキングを適用します
            curr_frame[:] = 0.
            toret[f] = curr_frame
        elif op_idx == 2:
            # クリップマスキングを適用します
            curr_frame, masked_frames_idx = mask_clip(f, toret, n_frames, K)
            clipped_masked_frames.extend(masked_frames_idx)
        else:
            # 何も変更しない
            pass
    # 損失計算に使用するすべてのマスクされたフレームのリストをコンパイルします
    masked_frames_idx = np.unique(np.concatenate((frames_to_mask, clipped_masked_frames)))
    
    return toret, masked_frames_idx

def mask_transform(seq, R, max_disturbance, no_mask_joint, K, m):
    """
    連続したフレームに対して異なる種類のマスキング変換を適用します。

    この関数は、与えられたフレームシーケンスに対していくつかのマスキング操作のうちの1つをランダムに適用します。
    マスキングの種類には、ジョイントマスキング、フレームマスキング、クリップマスキングが含まれます。

    Parameters:
    seq (numpy.ndarray): マスキングを適用するフレームのシーケンス。

    Returns:
    tuple:
        - numpy.ndarray: マスキングが適用された変換されたシーケンス。
        - numpy.ndarray: マスクが適用されたフレームのインデックス。
    """
    # 元のシーケンスを変更しないために、入力シーケンスのコピーを作成します
    toret = seq.copy()
    # マスキングされていないフレームの数を計算します
    n_frames = (toret != 0.0).all((1,2)).sum()
    # 事前に定義された比率Rに基づいてマスクするフレームの合計数を計算します
    n_frames_to_mask = int(np.ceil(R * n_frames))
    # マスクするフレームのインデックスをランダムに選択します
    frames_to_mask = np.random.choice(n_frames, size=n_frames_to_mask, replace=False)
    clipped_masked_frames = []
    for f in frames_to_mask:
        # マスクするフレームを取得します
        curr_frame = toret[f]
        # マスキング操作のタイプをランダムに選択します
        op_idx = np.random.choice(3) # 0: joint, 1: frame, 2: clip
        if op_idx == 0:
            # ジョイントマスキングを適用します
            curr_frame = mask_joint(curr_frame, max_disturbance, no_mask_joint, m)
            toret[f] = curr_frame
        elif op_idx == 1:
            # フレームマスキングを適用します
            curr_frame[:] = 0.
            toret[f] = curr_frame
        else:
            # クリップマスキングを適用します
            curr_frame, masked_frames_idx = mask_clip(f, toret, n_frames, K)
            clipped_masked_frames.extend(masked_frames_idx)
    # 損失計算に使用するすべてのマスクされたフレームのリストをコンパイルします
    masked_frames_idx = np.unique(np.concatenate((frames_to_mask, clipped_masked_frames)))
    
    return toret, masked_frames_idx

def mask_clip(frame_idx, seq, n_frames, K):
    """
    連続したフレームにクリップマスキングを適用します。
    
    クリップマスキングには、シーケンス内の連続したフレームの一部をゼロに設定することが含まれます。
    この関数は、マスキングを適用するフレームインデックスを中心に、マスクするクリップの長さをランダムに決定し、
    次にマスキングを適用します。

    Parameters:
    frame_idx (int): クリップの中心になるフレームのインデックス。
    seq (numpy.ndarray): マスキングを適用するフレームのシーケンス。
    n_frames (int): シーケンス内の総フレーム数。

    Returns:
    tuple:
        - numpy.ndarray: マスキングが適用されたシーケンス。
        - list: マスクされたフレームのインデックス。
    """
    # Kフレームまでのランダムなフレーム数をマスクすることをランダムに決定します
    n_frames_to_mask = np.random.randint(2, K+1)
    n_frames_to_mask_half = n_frames_to_mask // 2
    # マスクするクリップの開始と終了インデックスを計算します
    start_idx = frame_idx - n_frames_to_mask_half
    end_idx = frame_idx + (n_frames_to_mask - n_frames_to_mask_half)
    # 連続したフレームの一部がシーケンスの境界を超える場合は、境界に合わせて調整します
    if start_idx < 0:
        diff = abs(start_idx)
        start_idx = 0
        end_idx += diff
    if end_idx > n_frames:
        diff = end_idx - n_frames
        end_idx = n_frames
        start_idx -= diff
    # マスクするフレームのインデックスのリストを生成します
    masked_frames_idx = list(range(start_idx, end_idx))
    # 選択されたフレームをゼロに設定することでマスキングを適用します
    seq[masked_frames_idx] = 0.0

    return seq, masked_frames_idx

def mask_joint(frame, max_disturbance, no_mask_joint, m):
    """
    フレーム内の特定のジョイントにマスキングを適用します。

    この関数は、与えられたフレーム内の特定のジョイントを選択し、これらのジョイントにゼロマスキングまたは空間的な乱れを適用します。
    ゼロマスキングはジョイント座標をゼロに設定し、空間的な乱れは座標にランダムなオフセットを追加します。

    Parameters:
    frame (numpy.ndarray): マスキングを適用するフレーム（ジョイント座標の配列）

    Returns:
    numpy.ndarray: マスキングが適用されたフレーム。
    """
    # 空間的な乱れによる関数を定義
    def spatial_disturbance(xy):
        # [-max_disturbance, max_disturbance]の範囲内でランダムな乱れを追加
        return xy + [np.random.uniform(-max_disturbance, max_disturbance), np.random.uniform(-max_disturbance, max_disturbance)]
    
    # m個のジョイントをマスクするかどうかをランダムに決定します
    m = np.random.randint(1, m+1)
    # マスクするジョイントのインデックスをランダムに選択します
    joint_idxs_to_mask = np.random.choice(21, size=m, replace=False)
    # ゼロマスキングまたは空間的な乱れのどちらを適用するかをランダムに決定します
    op_idx = np.random.binomial(1, p=0.5, size=m).reshape(-1, 1)
    # 選択されたジョイントに選択されたマスキング操作を適用します
    frame[joint_idxs_to_mask] = np.where(
        op_idx, 
        spatial_disturbance(frame[joint_idxs_to_mask]), 
        spatial_disturbance(frame[joint_idxs_to_mask]) if no_mask_joint else 0.0
    )
    return frame