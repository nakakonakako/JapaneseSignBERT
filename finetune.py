"""
手話学習モデルのFineTuningを行うためのスクリプト

基本的な実行コマンド
python finetune.py --config finetune/configs/ISLR_JA.yml --ckpt /path/to/ckpt.ckpt 
[--epochs 600] [--device 0] [--name test] [--precision 32-true]

"""

import os
import argparse
from pprint import pformat

import yaml
from lightning.pytorch import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint

from finetune.ISLR.JADataModule import JADataModule
from finetune.SignBERTModel import SignBertModel



class Config:
    """Stores configuration parameters"""
    def __init__(self, **config):
        self.__dict__.update(config)
    
    def __repr__(self):
        return pformat(vars(self), indent=2)


def main(args):
    with open(args.config, "r") as fid:
        cfg = yaml.load(fid, yaml.SafeLoader)
    # コマンドライン引数（args）で構成を更新します（argsはcfgをオーバーライドします）
    cfg.update(args.__dict__)
    # Configオブジェクトを作成します
    config = Config(**cfg)
    print(config)
    # バッチサイズと追加のdatamodule引数を使用してdatamoduleをインスタンス化します
    datamodule = JADataModule(
        batch_size=config.batch_size,
        **config.datamodule_args
    )
    # チェックポイント、学習率、およびヘッド引数を使用してモデルをインスタンス化します
    model = SignBertModel(ckpt=config.ckpt, lr=config.lr, head_args=config.head_args)
    # ロギングとチェックポイントディレクトリを設定します
    logs_dpath = os.path.join(os.getcwd(), "finetune_logs")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=logs_dpath, name=config.name)
    ckpt_dirpath = os.path.join(tb_logger.log_dir, "ckpts")
    # モデルチェックポイントコールバックを構成します
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dirpath, 
        save_top_k=5, 
        monitor="val_acc", 
        mode="max", 
        filename="epoch={epoch:02d}-step={step}-{val_acc:.4f}", 
        save_last=True
    )
    # Trainerをセットアップして構成します
    trainer = Trainer(
        accelerator="gpu",
        strategy="auto",
        devices=[config.device],
        max_epochs=config.epochs,
        logger=tb_logger,
        callbacks=[ckpt_cb],
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        precision=config.precision
    )
    trainer.fit(model, datamodule) # Start training
    trainer.test(model, datamodule) # Start testing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--epochs", default=600, type=int)
    parser.add_argument("--name", default="test", type=str)
    parser.add_argument("--precision", default="32-true", type=str)
    args = parser.parse_args()

    main(args)