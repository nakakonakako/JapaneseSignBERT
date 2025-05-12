# SignBERT
SignBERT+を参照した、自己教師あり学習による手話学習モデル
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

## 研究の引継ぎ
### venvの構築
実行環境であるSupermicroでは、pyenv+venvにおける環境構築が推奨されています  
このディレクトリで以下を実行することで環境構築ができます

```bash
$ pyenv install 3.12
$ pyenv local 3.12
$ python -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
$ pip install signbert/model/thirdparty/manotorch
```

### chumpyの変更
使用するパッケージであるchumpyにおいてバグがあるため、モジュールを修正します
   - `venv/lib/python3.12/site-packages/chumpy/ch.py`における1203行目の
`getargspec`を`getfullargspec`に変更
   - `venv/lib/python3.12/site-packages/chumpy/__init__.py`における11行目をすべて削除

### MANOファイルのダウンロード
1. [MANO website](http://mano.is.tue.mpg.de/)でアカウントを作る(無料)
2. `Models & Code`をダウンロード
3. 解凍したフォルダの中身を`signbert/model/thirdparty/mano_assets`の中に入れる
```bash
mano_assets/
    ├── ._.DS_Store
    ├── .DS_Store		
    ├── __init__.py
    ├── LICENSE.txt
    ├── models
    │   ├── info.txt
    │   ├── LICENSE.txt
    │   ├── MANO_LEFT.pkl
    │   ├── MANO_RIGHT.pkl
    │   ├── SMPLH_female.pkl
    │   └── SMPLH_male.pkl
    └── webuser
        └── ...
```

### スクリプトの実行コマンドおよびオプション
それぞれの実行スクリプト内に記述があるので、そちらを参照してください

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