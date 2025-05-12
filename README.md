# SignBERT
SignBERT+を参照した、自己教師あり学習による手話学習モデル

## 研究概要
**背景**  
日本手話の連続認識には大量のアノテーション済みデータが必要ですが、そのデータは不足しており、高精度モデルの構築が困難です。従来は LSTM・Conformer＋CTC などの教師あり手法で孤立手話の認識には成功しているものの、連続手話では遷移部分の識別が難しく、またアノテーションコストも大きいという課題がありました

**目的**  
少量のアノテーション済みデータで高精度を実現できる「自己教師あり学習」を活用し、既存モデルの問題点を解明・改良することで、手話認識精度を実用レベルまで向上させる。さらにアノテーションコストを低減し、手話学習の普及に貢献する。

**手法**  
1. **データ前処理**  
   - OpenPose／MediaPipe による動画からのキーポイント抽出  
   - 時系列データとして整形  
2. **自己教師あり事前学習**  
   - Transformer Encoder（BERT）を骨組みとし、  
     - Frame-MLM：ランダムフレーム全体をマスク  
     - Point-MLM：ランダムキーポイントをマスク  
   - 大規模非アノテーションデータから特徴を学習  
3. **Fine-tuning**  
   - 日本手話データ（約27,000サンプル）でモデルをタスク特化学習  
4. **モデル改良**  
   - ST-GCN／MS-G3D などの Graph Convolution  
   - Sinusoidal Positional Encoding  
   - Hand-Aware Decoder（MANO メッシュ）導入  
5. **データ拡張**  
   - ドイツ手話（RWTH-PHOENIX, SIGNUM）、米国手話（MS-ASL, WLASL, How2Sign）を追加

**結果**  
- 非アノテーション事前学習で損失が安定的に低下  
- 孤立手話分類タスクで学習精度85%／検証精度55%（目標80%）達成  
- データ拡張により認識率が約1.2%→4.5%に向上

**今後の展望**  
- RGB 情報や追加データセットによるさらなる精度向上  
- 連続手話分類・翻訳タスクへの応用  
- 実環境でのリアルタイム認識システム構築

**キーワード**：手話認識／自己教師あり学習／BERT／Graph Convolution  


## ファイル構成
<dl>
	<dt>詳細はディレクトリ内にあるREADME.mdおよびスクリプト内を参照してください</dt>
	<dt>README.md</dt>
  <dd>当ファイル</dd>
  <dt>finetune/</dt>
	<dd>Fine-Tuning関連のディレクトリ</dd>
	<dt>signbert/</dt>
	<dd>事前学習関連のディレクトリ</dd>
	<dt>SLR/</dt>
	<dd>データセット毎の前処理を格納するディレクトリ</dd>
	<dt>finetune.py</dt>
	<dd>Fine-Tuningを実行するスクリプト</dd>
	<dt>train.py</dt>
	<dd>事前学習を実行するスクリプト</dd>
  <dt>requirements.txt</dt>
  <dd>パッケージのインストールファイル
</dl>

### データセット
```bash
Nakanishi/
    ├── how2sign        How2Signの元データセット
    ├── msasl		        MSASLの元データセット
    ├── npy             npy形式にしたファイルを格納したディレクトリ
    │   ├── how2sign    
    │   ├── ja            ja5とja6をまとめて1つにしたデータセット
    │   ├── ja5           NASにある手話検定5級のデータセットをMMPoseでnpy形式にしたデータセット
    │   ├── ja6           NASにある手話検定6級のデータセットをMMPoseでnpy形式にしたデータセット
    │   ├── msasl
    │   ├── phoenix       
    │   └── wlasl
    ├── output          MMPoseの姿勢推定結果を格納したディレクトリ
    └── wlasl           WLASLの元データセット
```

## 課題
<dl>
<dt>モデルの改良</dt>
<dd>現在、姿勢推定モデルから得られたキーポイントを用いて特徴量抽出を行っている</dd>
<dd>これに画像自体のRGBを特徴量とした学習を加えることで精度の向上を図ることができる</dd>

## 連絡先
mail: kotamina2016@gmail.com