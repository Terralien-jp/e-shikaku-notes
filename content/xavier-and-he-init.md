---
exam: E資格
concept: XavierとHe初期化
slug: xavier-and-he-init
tier: B
area: 深層学習
summary: 層を重ねても信号と勾配のスケールが崩れにくいよう、活性化関数の性質に合わせて重みの初期分散を決める方法です。
updated: 2026-08-22
sources:
  - title: "Understanding the difficulty of training deep feedforward neural networks"
    url: https://proceedings.mlr.press/v9/glorot10a.html
  - title: "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
    url: https://arxiv.org/abs/1502.01852
---

## ひとことで言うと

Xavier初期化とHe初期化は、学習前の重みを「適当に小さく」置くのではなく、層の入力数と活性化関数に応じて分散を決める方法です。狙いは、順伝播の出力と逆伝播の勾配が層をまたぐたびに縮小・増幅し続けないことです。

<div class="analogy">

長い配管に同じ倍率のポンプを何台も置くと、倍率が1を少し外れるだけでも末端では水圧が極端になります。初期化は、各段のポンプ倍率を最初から調整して、信号と勾配の流量を保つ設計です。

</div>

## なぜ必要か

重みの分散が大きすぎると、層ごとの出力や勾配が増幅され、逆に小さすぎると消えて学習が止まります。深いネットワークではこの差が層数に応じて積み重なるため、初期値の時点でスケールが崩れていると、更新則や学習率を調整しても最初から手詰まりになります。ゼロ初期化も解決になりません。同じ層のニューロンが同じ出力・同じ勾配になり、対称性が壊れないため、ニューロンごとの役割分担が始まらないからです。

| 初期分散の状態 | 層を重ねたときの帰結 |
|---|---|
| 大きすぎる | 活性と勾配が増幅し、発散側へ寄る |
| 小さすぎる | 活性と勾配が縮小し、学習が停滞する |
| 0 | 対称性が残り、各ニューロンが同じ更新になる |

GlorotとBengioは、線形に近い領域で順伝播の活性と逆伝播の勾配の分散を保つ条件を考え、入力数（fan-in）と出力数（fan-out）の両方を使う初期化を提案しました。これが一般にXavier初期化と呼ばれます。Heらは、ReLUでは負側が0になり、線形の場合の仮定をそのまま使えないとして、rectifierの性質を含めた初期化を導きました。

## 仕組み

入力数を $n_{in}$、出力数を $n_{out}$、重みを $W$ とします。Xavierは、順伝播では $n_{in}\operatorname{Var}(W)$、逆伝播では $n_{out}\operatorname{Var}(W)$ が効くことから、両者の折衷として次を使います。

$$
\operatorname{Var}(W)=\frac{2}{n_{in}+n_{out}}
$$

これは線形、またはtanhの中心付近のように導関数が概ね1の領域を前提にした設計です。順伝播と逆伝播のどちらか一方だけを完全に固定するのではなく、両方向のスケールを同時に大きく外さない点が判断の要所です。

He初期化では、ReLUの導関数が0または1で、対称な入力なら半分ずつになることを分散計算に入れます。順伝播の各層でおおむね半分が0になるため、その損失を補う条件は次です。$n_{in}$ は1つの出力が受け取る重み数です。

$$
\operatorname{Var}(W)=\frac{2}{n_{in}}
$$

正規分布なら標準偏差は $\sqrt{2/n_{in}}$ です。論文の導出では、順伝播でも逆伝播でもReLU由来の $1/2$ が現れ、積が層数に応じて指数的に縮まないよう係数2で補います。したがって、XavierをReLUに機械的に流用するのではなく、活性化に合わせて選びます。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| 初期化の目的 | 順伝播の信号と逆伝播の勾配のスケールを保つ | 重みをすべて小さくすればよい |
| XavierとHeの対応 | Xavierは線形・tanh、HeはReLU系 | ReLUにも常に $1/(n_{in}+n_{out})$ |
| 分散の式 | Xavierは $2/(n_{in}+n_{out})$、Heは $2/n_{in}$ | 標準偏差と分散を取り違える |
| ゼロ初期化 | 対称性が残り、ニューロンが同じ更新になる | 「勾配が小さいから」だけで説明する |

## 実装で確かめる

fan-inだけを使うHe初期化では、層幅が変わっても理論上の重み分散は $2/n_{in}$ になります。乱数の実現値は揺れるので、ここでは複数要素の分散を確認します。

```python
import numpy as np
rng = np.random.default_rng(0)
fan_in = 256
w = rng.normal(0, np.sqrt(2 / fan_in), size=(fan_in, 128))
print(round(w.var(), 4), round(2 / fan_in, 4))
```

## 取り違えやすいもの

| 用語 | 初期化との切り分け |
|---|---|
| Xavier | fan-inとfan-outの折衷。線形・tanhを前提に分散を決める |
| He | ReLUの半分が0になる性質を補正し、主にfan-inで決める |
| 一様分布・正規分布 | 分布の形の選択。分散を合わせれば同じ設計思想で使える |
| バイアスの初期値 | 重みの分散設計とは別の値。He論文ではバイアスを0にしている |
| [正規化層](/learn/e-shikaku/normalization-layers/) | 学習中の出力分布を扱う別の仕組み。初期重みの設計そのものではない |

## 想起チェック

<details class="recall">
<summary>XavierとHeを活性化関数で切り分けると</summary>

線形・tanhならXavier、ReLU系ならHeです。決め手は、初期化の分散計算に活性化の性質を入れるかどうかです。

</details>

<details class="recall">
<summary>He初期化の係数2はどこから来るか</summary>

ReLUでは対称な入力の半分が0になり、順伝播・逆伝播の分散におおむね $1/2$ が掛かるためです。

</details>

<details class="recall">
<summary>重みを全部0にすると何が起きるか</summary>

同じ層のニューロンが対称なまま同じ勾配で更新され、異なる特徴を学ぶきっかけがありません。

</details>
