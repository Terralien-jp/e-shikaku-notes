---
exam: E資格
concept: 拡散モデル
slug: diffusion-model
tier: A
area: 深層学習
summary: データに段階的にノイズを加える過程を定め、その逆向きに各段階のノイズを予測して画像を生成する確率モデル。
updated: 2026-08-22
sources:
  - title: "Denoising Diffusion Probabilistic Models"
    url: https://arxiv.org/abs/2006.11239
---

## ひとことで言うと

拡散モデルは、データを少しずつ壊して最終的にガウスノイズへ近づける**前向き過程**と、その逆向きにノイズを取り除いてデータへ戻す**逆過程**を組み合わせた生成モデルです。DDPM では、逆過程を直接「きれいな画像」へ写すのではなく、各時刻で加わったノイズ $\boldsymbol{\epsilon}$ をニューラルネットワーク $\boldsymbol{\epsilon}_{\theta}(\mathbf{x}_t,t)$ に予測させます。

<div class="analogy">

写真を何段階も薄い霧で覆い、最後には何も見えなくする工程を考えます。学習では、霧の濃さを指定して「今回加えた霧は何だったか」を当てさせます。生成時は完全な霧から始め、当てた霧を一段ずつ取り除きます。1回で復元しないことが、拡散モデルの見た目と計算量を決めます。

</div>

## なぜ必要か

画像を一発で生成する写像を学習させると、複雑なデータ分布を一度の予測に押し込む必要があります。拡散モデルは、ノイズ量の異なる多数の復元問題へ分解します。入力画像 $\mathbf{x}_0$ に時刻 $t$ のノイズを加えた $\mathbf{x}_t$ を作り、ネットワークは「この段階で混ざったノイズ」を予測します。教師信号は自分で加えたノイズなので、画像どうしの対応を用意する必要はありません。

これは GAN のように生成器と識別器を敵対的に最適化する枠組みではありません。変分推論に基づく確率モデルとして逆過程を学習し、サンプリングはその逆過程を順に実行します。VAE のような潜在変数モデルとの接点はありますが、ここで扱う DDPM の $\mathbf{x}_t$ はデータと同じ次元の中間状態です。潜在空間で同じ考え方を行う潜在拡散や、条件を入力に加える条件付き生成は別の拡張です。

| 観点 | DDPM |
|---|---|
| 学習の教師 | 自分で加えたノイズ |
| 生成の開始点 | 標準正規ノイズ |
| 生成の進み方 | 時刻を逆向きに一段ずつ更新 |

## 仕組み

前向き過程は、データ $\mathbf{x}_0$ から時刻 $T$ のノイズ $\mathbf{x}_T$ までをマルコフ連鎖で作ります。$\beta_t$ は時刻 $t$ に加えるノイズの分散、$\mathbf{I}$ は単位行列です。

$$
q(\mathbf{x}_t\mid\mathbf{x}_{t-1}) = \mathcal{N}\left(\sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\,\beta_t\mathbf{I}\right)
$$

$\alpha_t=1-\beta_t$、$\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s$ と置くと、途中の状態は一段ずつ生成せず直接サンプルできます。

$$
\mathbf{x}_t=\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon},\qquad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})
$$

$\mathbf{x}_0$ は元データ、$\mathbf{x}_t$ は時刻 $t$ のノイズ画像、$\boldsymbol{\epsilon}$ は標準正規ノイズです。$t$ が進むほど信号の係数 $\sqrt{\bar{\alpha}_t}$ は小さくなり、最終状態を標準正規分布に近づけます。学習時は画像と時刻 $t$ を選び、この式で $\mathbf{x}_t$ を作ります。

逆過程は、ノイズから始めて $t=T,T-1,\ldots,1$ の順に戻ります。モデルが表す一段の遷移を $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ とし、平均をニューラルネットワークで決めます。DDPM の実装で中心になるのは、平均をノイズ予測でパラメータ化することです。

$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\right)
$$

$\boldsymbol{\mu}_\theta$ は逆過程の平均、$\theta$ はネットワークの学習パラメータです。学習の重い変分下限を、実装では次の単純なノイズ予測損失として扱います。

$$
L_{\mathrm{simple}}=\mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}\left[\left\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\right\|^2\right]
$$

つまり、学習の1回は「時刻を選ぶ→ノイズを加える→加えたノイズを予測する→二乗誤差を最小化する」です。生成時には予測したノイズから逆過程の平均を計算し、必要な分散のランダムノイズを加えて次の状態を作ります。これを何段も繰り返すため、生成は1回の順伝播で終わらず遅くなります。ここで $t$ をネットワークに渡し忘れると、ノイズの強さが異なる復元問題を区別できません。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| 前向き過程の説明 | データに小さいガウスノイズを段階的に加え、最終的にノイズへ近づける | 前向きが画像を生成する過程だとする |
| 逆過程の説明 | ノイズから始め、時刻を $T$ から0へ戻す | 時刻を0から $T$ へ進める |
| 学習対象の説明 | $\boldsymbol{\epsilon}$ そのものではなく、入力 $(\mathbf{x}_t,t)$ から加えたノイズを予測するネットワーク | 画像を直接回帰する、とだけ説明する |
| 生成が遅い理由 | 多数の逆拡散ステップで順伝播を繰り返す | 学習データの読み込みが主因とする |
| GAN との違い | 識別器との敵対的学習ではなく、ノイズ予測の損失で逆過程を学習する | 必ず識別器を同時に更新するとする |

## 実装で確かめる

前向き過程の閉形式を NumPy で確認します。`x0` に対して時刻を変えると、同じ乱数の形でも信号係数とノイズ係数の組が変わります。

```python
import numpy as np

rng = np.random.default_rng(0)
x0 = np.array([1.0, -0.5, 0.25])
betas = np.array([0.01, 0.05, 0.10])
alphas = 1.0 - betas
alpha_bar = np.cumprod(alphas)
epsilon = rng.normal(size=x0.shape)

for t in range(len(betas)):
    xt = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * epsilon
    print(t + 1, np.round(xt, 4))
```

このコードで作られる $\mathbf{x}_t$ は、各時刻のノイズを独立に何度も足した結果と同じ分布です。実装では `alpha_bar` の添字をデータの時刻と取り違えやすく、$t=1$ を配列の0番目に対応させるかを最初に固定します。

<div class="caution">

前向き過程の $β_t$ と $\bar{\alpha}_t$ は逆過程の式にも現れます。学習時と生成時でスケジュール、時刻の範囲、画像のスケーリングを変えると、ノイズ予測が合っていても逆過程の入力分布がずれます。

</div>

## 取り違えやすいもの

| 用語 | 拡散モデルとの切り分け |
|---|---|
| 前向き過程 | 学習用にデータを壊す固定の確率過程。通常、ここをニューラルネットワークが学習するわけではない |
| 逆過程 | ノイズからデータへ戻す学習対象の確率過程。ネットワークは時刻ごとのノイズや平均を予測する |
| ノイズ予測 | 逆過程を実装しやすくするパラメータ化。ネットワークの出力をそのまま画像と解釈しない |
| [VAE](/learn/e-shikaku/vae/) | エンコーダ・デコーダで潜在表現を一度に扱う。DDPM は多数の時刻の中間状態を使う |
| [GAN](/learn/e-shikaku/gan/) | 生成器と識別器を敵対的に学習する。DDPM の基本学習はノイズ予測の回帰損失である |
| [自己回帰モデル](/learn/e-shikaku/autoregressive-generation/) | 画素やトークンなどを順序づけて生成する。DDPM も多段だが、時刻ごとのノイズ除去を行う |

## 想起チェック

<details class="recall">
<summary>拡散モデルの前向き過程と逆過程は、それぞれ何をするか</summary>

前向き過程はデータに段階的にガウスノイズを加え、逆過程はノイズから始めてその手順を逆向きにたどります。

</details>

<details class="recall">
<summary>DDPM のネットワークは各時刻で何を予測するか</summary>

入力 $\mathbf{x}_t$ と時刻 $t$ から、前向き過程で加えたノイズ $\boldsymbol{\epsilon}$ を予測します。その二乗誤差が単純化された学習目的です。

</details>

<details class="recall">
<summary>生成が遅くなりやすい理由は何か</summary>

完全なノイズから画像まで、逆過程の遷移を時刻 $T$ から0へ多数回実行するためです。各段階でネットワークの予測が必要になります。

</details>

<details class="recall">
<summary>GAN と比べたときの基本的な学習の違いは何か</summary>

GAN のような生成器と識別器の敵対的最適化ではなく、DDPM は加えたノイズを予測する損失で逆過程を学習します。

</details>
