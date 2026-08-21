---
exam: E資格
concept: 活性化関数
slug: activation-functions
tier: A
area: 深層学習
summary: 各層の出力に非線形性を入れる関数群。導関数の値域から勾配消失・出力の0中心・dying ReLU・計算コストが機械的に読める。
updated: 2026-08-22
sources:
  - title: "Deep Learning: Deep Feedforward Networks"
    url: https://www.deeplearningbook.org/contents/mlp.html
  - title: "PyTorch nn package"
    url: https://docs.pytorch.org/docs/stable/nn.html
---

## ひとことで言うと

活性化関数は、各層の**アフィン変換 $z=Wx+b$ の出力に非線形性を挿入する関数**です。線形変換だけを何層重ねても、合成すれば係数行列の積になるだけで全体は1つの線形変換に潰れます。活性化関数が無いと、層を深くする操作そのものに意味が無くなります。

<div class="analogy">

各層を「フィルタを通す工程」だと考えると、線形なフィルタだけをいくら直列につないでも、結局は1つの巨大な線形フィルタと同じ結果にしかなりません。工程の間に非線形な検査（しきい値で弾く・符号で挙動を変える）を挟んで初めて、工程を増やすこと自体に意味が出てきます。

</div>

## なぜ必要か

活性化関数の主流は「勾配をどれだけ通すか」で移り変わってきました。sigmoid・tanh の時代は出力を有界な値や確率に押し込めることを優先しましたが、深い層を重ねると勾配が層を経るたびに小さくなり学習が止まる問題（勾配消失）が表面化しました（Goodfellow et al., "Deep Feedforward Networks"）。

| 世代 | 代表 | 優先したこと | 代償 |
|---|---|---|---|
| 第1世代 | sigmoid, tanh | 出力を有界にする・確率解釈 | 飽和域で勾配がほぼ0 |
| 第2世代 | ReLU とその一族 | 勾配を大きく・一定に保つ | 負領域で勾配ゼロ（dying ReLU） |
| 第3世代 | GELU, Swish | ReLU の折れ目を滑らかにする | 計算コスト増（exp・tanh を含む） |

隠れ層と出力層では選ぶ基準が違う点にも注意が要ります。隠れ層は勾配の通りやすさで選びますが、**出力層はタスクと損失関数の組で決まります**（二値分類なら sigmoid＋交差エントロピー、多クラスなら softmax＋交差エントロピー、回帰なら恒等関数）。

## 仕組み

各関数を $g(z)$、導関数を $g'(z)$ として、値域と導関数の値域を並べます。**導関数の値域がそのまま逆伝播で層をまたいで掛かる係数の範囲**になるため（誤差逆伝播法の $\delta^{(l)}$ の漸化式を参照）、ここが勾配消失・爆発を判定する直接の材料になります。

$$
\sigma(z) = \frac{1}{1+e^{-z}}, \qquad \sigma'(z) = \sigma(z)(1-\sigma(z))
$$

$$
\tanh(z) = 2\sigma(2z) - 1, \qquad \tanh'(z) = 1 - \tanh^2(z)
$$

ReLU 系は負領域の傾きを $\alpha$ として統一的に書けます（$\alpha=0$ が通常の ReLU、$\alpha=0.01$ 程度の固定値が Leaky ReLU、$\alpha$ を学習させると PReLU）。

$$
g(z) = \max(0, z) + \alpha \min(0, z)
$$

ELU は負領域を指数で滑らかにつなぎます（既定は $\alpha=1$）。

$$
\mathrm{ELU}(z) = \begin{cases} z & z \geq 0 \\ \alpha(e^{z}-1) & z < 0 \end{cases}
$$

Swish（PyTorch では SiLU）は入力そのものを sigmoid でゲートする自己ゲート型です。

$$
\mathrm{Swish}(z) = z \cdot \sigma(z)
$$

GELU は標準正規分布の累積分布関数 $\Phi$ を使い $\mathrm{GELU}(z) = z\Phi(z)$ と定義され、実装では $\tanh$ を使った近似式が使われます。

| 関数 | 値域 | 導関数の値域 | 0中心 | 飽和 |
|---|---|---|---|---|
| sigmoid | $(0,1)$ | $(0, 0.25]$ | ✕ | 両側 |
| tanh | $(-1,1)$ | $(0, 1]$ | ○ | 両側 |
| ReLU | $[0,\infty)$ | $\{0,1\}$ | ✕ | 負側のみ（0で固定） |
| Leaky ReLU | $(-\infty,\infty)$ | $\{\alpha,1\}$（既定 $\alpha=0.01$） | ほぼ○ | しない |
| ELU | $(-\alpha,\infty)$ | $(0,1]$ | ほぼ○ | 負側で漸近 |
| GELU / Swish | ほぼ $(-0.17,\infty)$ ／ $(-0.28,\infty)$ | 非単調 | ほぼ○ | しない |

sigmoid の導関数の最大値が 0.25 という点が最重要です。層を重ねるたびにこの値が掛かるので、$n$ 層では最大でも $0.25^n$ まで縮み、10層程度で実用上ゼロになります。ReLU 系は活性化している素子では係数が1のまま伝わるため、この指数的な縮小が起きません。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| sigmoid が隠れ層で避けられる理由 | 導関数の最大値が0.25で、層を重ねるほど勾配が指数的に縮む | 「計算コストが高いから」だけを理由にする（副次的な理由にすぎない） |
| dying ReLU の発生条件 | 入力が恒常的に負になり、その素子の導関数がずっと0になる | 「学習率が高すぎるから起きる」に一般化する（誘因の一つで定義ではない） |
| Leaky ReLU と PReLU の違い | Leaky ReLU は負側の傾き $\alpha$ が固定値、PReLU は学習可能パラメータ | 「PReLU は Leaky ReLU の別名」 |
| 出力層の活性化関数の選び方 | タスク（二値／多クラス／回帰）と損失関数の組で決まる | 隠れ層と同じ基準（勾配の通りやすさ）で選ぶ |
| tanh と sigmoid の関係 | $\tanh(z) = 2\sigma(2z)-1$。0中心かどうかが主な違い | 「まったく別系統の関数」として扱う |

## 実装で確かめる

各関数の導関数を実測し、上の表の値域を裏取りします。

```python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))
def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)
def d_tanh(z): return 1 - np.tanh(z) ** 2
def d_leaky_relu(z, alpha=0.01): return np.where(z >= 0, 1.0, alpha)
def swish(z): return z * sigmoid(z)

z = np.linspace(-10, 10, 200_001)
print("sigmoid' の最大値:", d_sigmoid(z).max())
print("tanh' の最大値:", d_tanh(z).max())
print("leaky_relu' の最小値:", d_leaky_relu(z).min())
print("swish の最小値:", swish(z).min())
```

実行すると次の値が出ます。

```
sigmoid' の最大値: 0.25
tanh' の最大値: 1.0
leaky_relu' の最小値: 0.01
swish の最小値: -0.2784645426241578
```

sigmoid の 0.25 と tanh の 1.0 の差が、そのまま「tanh のほうが勾配消失に強い」の根拠です。Leaky ReLU は最小でも 0.01 が流れるため、導関数が完全に0で固定される dying ReLU を構造的に避けています。

<div class="caution">

PyTorch では `nn.ReLU` `nn.LeakyReLU` `nn.ELU` `nn.GELU` `nn.SiLU`（Swish に相当）`nn.Softmax` がそれぞれ独立したモジュールとして提供されています（PyTorch nn package）。**Swish という名前のクラスは無く**、`SiLU` を使う点に注意してください。

</div>

## 取り違えやすいもの

| 用語 | 活性化関数との関係 |
|---|---|
| softmax | 出力層専用。**隠れ層の活性化関数の一覧には並べない**。多クラス分類でクラス間の確率を正規化する関数で、単一素子ではなくベクトル全体に作用する |
| バッチ正規化 | 活性化関数の**前**に入力の分布を整える手法。関数そのものではなく前処理層 |
| dying ReLU | 現象の名前であって関数の名前ではない。原因は ReLU の負領域で導関数が0になること |
| PReLU | Leaky ReLU の一般化。傾き $\alpha$ が固定値ではなく学習パラメータになったもの |
| GELU / Swish | 別々に提案された関数だが、滑らかな非単調・自己ゲート型という形が近い。式は異なる（$z\Phi(z)$ と $z\sigma(z)$） |

## 想起チェック

<details class="recall">
<summary>sigmoid を隠れ層に使うのを避ける理由を、導関数の値から説明すると</summary>

sigmoid の導関数は最大でも 0.25 です。層を重ねるたびにこの値が掛け合わされるため、深いネットワークでは勾配が指数的に縮み、学習がほぼ止まります。

</details>

<details class="recall">
<summary>dying ReLU とは何か</summary>

ReLU の入力が恒常的に負になった素子で、その導関数がずっと0になり、勾配降下法で二度と更新されなくなる現象です。Leaky ReLU や ELU は負領域にも小さな傾きを残すことでこれを避けます。

</details>

<details class="recall">
<summary>tanh が sigmoid よりも隠れ層で好まれてきた理由</summary>

$\tanh(0)=0$ で0中心の出力になるのに対し、$\sigma(0)=0.5$ で出力が正の側に偏ります。0中心のほうが後段の重み更新が特定の方向に偏りにくく収束が安定します。導関数の最大値も tanh は1.0、sigmoid は0.25で、tanh のほうが勾配を大きく通します。

</details>

<details class="recall">
<summary>出力層の活性化関数は何で決まるか</summary>

隠れ層のように「勾配の通りやすさ」では選びません。タスクと損失関数の組み合わせで決まります。二値分類は sigmoid、多クラス分類は softmax、回帰は恒等関数が既定です。

</details>
