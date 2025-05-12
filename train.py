"""
手話学習モデルの事前学習を行うためのスクリプト

基本的な実行コマンド
python train.py --config signbert/configs/pretrain.yml
[--ckpt /path/to/ckpt.ckpt] [--epochs 600] [--device 0] [--lr 1e-4] [--name test] [--val-interval 1]

"""

import os
import argparse
from pprint import pprint

import yaml
from lightning.pytorch import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from signbert.model.PretrainSignBertModelManoTorch import SignBertModel as PretrainSignBert
from signbert.data_modules.PretrainDataModule import PretrainDataModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--name', default='test', type=str)
    parser.add_argument('--val-interval', default=None, type=int)
    args = parser.parse_args()
    # コンフィグの読み込み
    with open(args.config, 'r') as fid:
        cfg = yaml.load(fid, yaml.SafeLoader)
    pprint(cfg)
    epochs = args.epochs if args.epochs is not None else 600
    lr = args.lr if args.lr is not None else cfg['lr'] # Preference over arguments
    batch_size = cfg['batch_size']
    normalize = cfg['normalize']
    datasets = cfg.get("datasets", None)
    
    assert datasets is not None
    # データモジュールの初期化
    datamodule = PretrainDataModule(
        datasets,
        batch_size=batch_size,
        normalize=normalize
    )
    # モデルの初期化
    model = PretrainSignBert(
        **cfg["model_args"],
        lr=lr, 
        normalize_inputs=normalize, 
    )
    
    trainer_config = dict(
        accelerator='gpu',
        strategy='auto',
        devices=[args.device],
        max_epochs=epochs
    )

    if args.ckpt: # チェックポイントが指定されている場合
        print('Resuming training from ckpt:', args.ckpt)
    # 学習率のログを出力
    lr_logger = LearningRateMonitor(logging_interval='step')
    # tensorboardのログを出力 
    logs_dpath = os.path.join(os.getcwd(), 'logs')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=logs_dpath, name=args.name)
    # チェックポイントのコールバック
    ckpt_dirpath = os.path.join(tb_logger.log_dir, 'ckpts')
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dirpath, save_top_k=10, monitor="val_PCK_20", mode='max', filename="epoch={epoch:02d}-step={step}-{val_PCK_20:.4f}", save_last=True)
    # 早期終了のコールバック
    early_stopping_callback = EarlyStopping(monitor="val_PCK_20", mode="max", patience=30, min_delta=1e-4)
    # Trainerの初期化 
    trainer = Trainer(
        **trainer_config,
        accumulate_grad_batches=cfg.get('accumulate_grad_batches', 1), 
        logger=tb_logger, 
        callbacks=[lr_logger, checkpoint_callback],#, early_stopping_callback],
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        precision=cfg.get('precision', '32-true'),
        check_val_every_n_epoch=args.val_interval
    )
    trainer.fit(model, datamodule, ckpt_path=args.ckpt)