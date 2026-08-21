---
exam: E資格
concept: VAE
slug: vae
tier: A
area: 深層学習
summary: 入力から潜在変数の分布を推定するエンコーダと、潜在変数からデータを生成するデコーダを組み合わせ、変分下限を再パラメータ化して学習する生成モデルです。
updated: 2026-08-22
sources:
  - title: "Auto-Encoding Variational Bayes"
    url: https://arxiv.org/abs/1312.6114
---

## ひとことで言うと

VAE（Variational Auto-Encoder）は、入力データを1個のコードに圧縮するのではなく、潜在変数 $\mathbf{z}$ の分布を推定してからデータを生成するモデルです。エンコーダは入力 $\mathbf{x}$ から近似事後分布 $q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})$ を出力し、デコーダはそのサンプルを使って生成分布 $p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z})$ を表します。$\boldsymbol{\phi}$ はエンコーダ、$\boldsymbol{\theta}$ は生成モデルのパラメータです。

<div class="analogy">

写真を「1枚の代表ラベル」に置き換えるのではなく、「この写真はこの範囲の特徴を持つ」という封筒に入れて渡す仕組みです。デコーダは封筒から1点を取り出して、元らしい写真を生成します。封筒の広がりを無視しないため、潜在空間からのサンプリングが学習の中心になります。

</div>

## なぜ必要か

通常のオートエンコーダは、入力をエンコーダで潜在表現に写し、デコーダで復元します。しかし、決定論的な1点の表現だけでは、潜在空間の途中の点を選んで妥当なデータを生成できるとは限りません。復元誤差だけを小さくすることと、生成モデルとして潜在変数を扱えることは同じではありません。

VAEでは、生成側の事後分布 $p_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})$ を直接求める代わりに、エンコーダが $q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})$ を近似します。原論文の狙いは、連続潜在変数を持ち、真の事後分布が扱いにくいモデルでも、認識モデルをデータごとに最適化する高価な反復推論なしに学習しやすくすることです。

| モデル | 潜在表現 | 学習の見方 |
|---|---|---|
| 通常のオートエンコーダ | 1点のコード | 復元誤差を小さくする |
| VAE | 条件付き分布 $q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})$ | 復元と事前分布への整合を同時に扱う |

## 仕組み

まず、入力 $\mathbf{x}$ と潜在変数 $\mathbf{z}$ の生成モデルを考えます。データ1点に対する対数周辺尤度 $\log p_{\boldsymbol{\theta}}(\mathbf{x})$ は、近似事後分布を導入すると次のように分解できます。

$$
\log p_{\boldsymbol{\theta}}(\mathbf{x}) = \mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi};\mathbf{x}) + D_{KL}\left(q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})\,\|\,p_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})\right)
$$

KLダイバージェンスは0以上なので、$\mathcal{L}$ は対数周辺尤度の下限です。実際に最大化するELBOは、生成モデルの事前分布 $p_{\boldsymbol{\theta}}(\mathbf{z})$ を使って次の2項に書けます。

$$
\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi};\mathbf{x}) = \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})}\left[\log p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z})\right] - D_{KL}\left(q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})\,\|\,p_{\boldsymbol{\theta}}(\mathbf{z})\right)
$$

第1項の期待対数尤度は再構成項です。サンプルした $\mathbf{z}$ からデコーダが元の $\mathbf{x}$ を高い確率で出せるようにするので、データへの適合を担います。第2項のKL項は、入力ごとの近似事後分布を事前分布から離れすぎないようにする正則化です。したがって損失として実装すると、再構成損失にKLダイバージェンスを加えた形になります。KL項を単なる飾りのペナルティと見ると、生成時に潜在空間からサンプルできる理由を取り逃がします。

ガウス事前分布と対角共分散のガウス近似事後分布を使う場合、エンコーダは平均 $\boldsymbol{\mu}(\mathbf{x})$ と標準偏差 $\boldsymbol{\sigma}(\mathbf{x})$ を出力します。サンプリングをそのまま書くと勾配計算の経路が確率変数で切れます。そこで、標準正規乱数 $\boldsymbol{\epsilon}$ を外から与え、次の決定論的な変換にします。

$$
\mathbf{z} = \boldsymbol{\mu}(\mathbf{x}) + \boldsymbol{\sigma}(\mathbf{x}) \odot \boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})
$$

$\odot$ は要素ごとの積、$\mathbf{0}$ は零ベクトル、$\mathbf{I}$ は単位行列です。乱数そのものではなく $\boldsymbol{\mu}$ と $\boldsymbol{\sigma}$ を通る計算として $\mathbf{z}$ を表すため、デコーダの再構成項からエンコーダまで連鎖律で勾配を流せます。これが再パラメータ化トリックです。乱数を消したのではなく、勾配を追える形に乱数の位置を移した、と捉えると実装上の意図が明確です。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| ELBOの2項の意味 | 再構成項はデータへの適合、KL項は近似事後と事前分布の距離 | KL項を再構成誤差の一部とする |
| エンコーダの出力 | 潜在変数の分布のパラメータ（平均・分散など） | 決定論的な潜在ベクトル1個だけを出す |
| 再パラメータ化の目的 | サンプリングを $\boldsymbol{\mu},\boldsymbol{\sigma}$ と外生乱数の変換にし、勾配を流す | サンプル数を増やすための技巧とする |
| ELBOを最大化する理由 | 対数周辺尤度の下限であり、KL項が非負 | ELBOが対数尤度そのものだとする |
| 生成時の潜在変数 | 事前分布から $\mathbf{z}$ をサンプルしてデコーダへ渡す | 学習時の入力ごとの $\mathbf{x}$ を必ず必要とする |

## 実装で確かめる

次の最小例では、標準正規分布から乱数を取り出し、エンコーダの平均と標準偏差を通して $\mathbf{z}$ を作ります。対角ガウス同士のKL項も同じサンプルについて計算します。再構成項は、デコーダが返す対数尤度として別に評価する設計です。

```python
import numpy as np

rng = np.random.default_rng(7)
mu = np.array([0.4, -0.2])
log_var = np.array([-0.3, 0.5])
sigma = np.exp(0.5 * log_var)
epsilon = rng.normal(size=mu.shape)
z = mu + sigma * epsilon

# q(z|x)=N(mu, diag(exp(log_var))) と N(0, I) のKL
kl = 0.5 * np.sum(np.exp(log_var) + mu**2 - 1.0 - log_var)
print("z:", z)
print("kl:", kl)
```

ここで乱数を固定しても、$\mathbf{z}$ は $\boldsymbol{\mu}$ と $\boldsymbol{\sigma}$ の関数として計算されています。実際の学習では、デコーダが出した分布の対数尤度を再構成項にし、\`kl\` を足した負のELBOを最小化します。対角ガウスの場合、KL項はサンプリングせず解析的に計算できるため、乱数推定のノイズをこの項に持ち込みません。

## 取り違えやすいもの

| 用語 | VAEとの切り分け |
|---|---|
| [通常のオートエンコーダ](/learn/e-shikaku/autoencoder/) | 復元する1点のコードが中心。VAEはコードを条件付き分布として扱い、KL項を含む下限を最適化する |
| 変分推論 | 近似事後を最適化する枠組み。VAEはその認識モデルにニューラルネットを使う構成 |
| 再構成誤差 | ELBOの再構成項に対応するが、VAEの目的関数全体ではない |
| [GAN](/learn/e-shikaku/gan/) | 生成器と識別器の競合で学習するモデルで、VAEのELBOや再パラメータ化とは別の仕組み |
| [拡散モデル](/learn/e-shikaku/diffusion-model/) | ノイズ除去過程を学習する生成モデルで、VAEの潜在分布と同じものではない |

## 想起チェック

<details class="recall">
<summary>ELBOを構成する2項は、それぞれ何を押し上げ、何を抑えるか</summary>

再構成項は、潜在変数から元データを説明する対数尤度を押し上げます。KL項は、入力ごとの近似事後分布が事前分布から離れすぎることを抑えます。

</details>

<details class="recall">
<summary>VAEのエンコーダは、通常のオートエンコーダと違って何を出力するか</summary>

潜在変数 $\mathbf{z}$ の近似事後分布を定めるパラメータです。対角ガウスなら平均 $\boldsymbol{\mu}(\mathbf{x})$ と標準偏差または対数分散を出力します。

</details>

<details class="recall">
<summary>再パラメータ化トリックは、なぜサンプリングを置き換えるのか</summary>

乱数を $\boldsymbol{\mu}(\mathbf{x})+\boldsymbol{\sigma}(\mathbf{x})\odot\boldsymbol{\epsilon}$ の外生入力に分離し、再構成項からエンコーダのパラメータまで勾配を流せるようにするためです。

</details>

<details class="recall">
<summary>ELBOを最大化することと、対数周辺尤度を最大化することの関係は</summary>

ELBOは対数周辺尤度の下限です。両者の差は近似事後分布と真の事後分布のKLダイバージェンスで、0以上になります。

</details>
