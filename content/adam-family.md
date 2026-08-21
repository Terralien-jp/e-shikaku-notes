---
exam: E資格
concept: Adam系最適化
slug: adam-family
tier: B
area: 深層学習
summary: 一次モーメントと二次モーメントで勾配を整え、パラメータごとに実効学習率を変える適応的な最適化手法です。
updated: 2026-08-22
sources:
  - title: "Adam: A Method for Stochastic Optimization"
    url: https://arxiv.org/abs/1412.6980
---

## ひとことで言うと

Adamは、勾配の向きの移動平均（一次モーメント）と、勾配の大きさの移動平均（二次モーメント）を併用して更新する確率的最適化手法です。分母で座標ごとの大きさをならすため、同じ学習率を指定してもパラメータごとの実効学習率は同じになりません。

<div class="analogy">

一次モーメントは「これまでの進行方向」、二次モーメントは「その方向の足場の荒さ」を記録します。Adamは進行方向を急に変えず、荒い座標では歩幅を抑え、静かな座標では相対的に大きく進みます。

</div>

## なぜ必要か

勾配の座標ごとにスケールが違うと、固定幅の更新では一部の座標だけが大きく動きます。Adamは一次モーメントで符号の揺れを平均化し、二次モーメントで更新量を正規化します。これにより、疎な勾配やノイズの大きい問題にも使いやすい設計になります。

ただし適応的に歩幅を変えることは、常に最終的な収束が有利という意味ではありません。Adam系では、訓練が速く進んでも、問題や設定によってはSGD系のほうが最終的な解に近づく場合があります。最初から「AdamならSGDより強い」と決めず、検証損失と更新の挙動を分けて見ます。

| 不便 | Adam系で見る量 | 更新への影響 |
|---|---|---|
| 勾配の向きが揺れる | 一次モーメント | 方向を平均化する |
| 座標ごとの大きさが違う | 二次モーメント | 分母で歩幅を調整する |

## 仕組み

時刻 $t$ の勾配を $\mathbf{g}_t$、パラメータを $\boldsymbol{\theta}_t$、一次・二次モーメントを $\mathbf{m}_t,\mathbf{v}_t$、減衰率を $\beta_1,\beta_2$、微小量を $\varepsilon$ とします。

$$
\mathbf{m}_t=\beta_1\mathbf{m}_{t-1}+(1-\beta_1)\mathbf{g}_t,\qquad
\mathbf{v}_t=\beta_2\mathbf{v}_{t-1}+(1-\beta_2)\mathbf{g}_t^2
$$

初期値を $\mathbf{m}_0=\mathbf{v}_0=\mathbf{0}$ とすると、初期の移動平均は0側に偏ります。そこで $\hat{\mathbf{m}}_t=\mathbf{m}_t/(1-\beta_1^t)$、$\hat{\mathbf{v}}_t=\mathbf{v}_t/(1-\beta_2^t)$ と補正します。更新は

$$
\boldsymbol{\theta}_{t+1}=\boldsymbol{\theta}_t-\alpha\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\varepsilon}
$$

です。$\alpha$ は基準学習率で、実際の座標別の歩幅は分母にも依存します。補正を省くと、特に初期ステップでモーメントを過小評価し、意図した更新量になりません。

| 観点 | Adam | AdamW |
|---|---|---|
| 重み減衰 | 勾配に正則化項を足してから適応化 | 適応化した更新と減衰を分離 |
| 実装上の注意 | 分母の影響を受ける | 減衰率を独立した更新として扱う |

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| モーメントの役割 | 一次は勾配の平均、二次は勾配二乗の平均 | 二次を「勾配の符号の平均」とする |
| バイアス補正の理由 | ゼロ初期化した移動平均の初期偏りを直す | 学習率を自動決定する補正とする |
| 実効学習率 | 分母によりパラメータ座標ごとに変わる | 全座標が同じ幅で更新されるとする |
| AdamとAdamW | 減衰を勾配に混ぜるか、更新で分離するか | 名前が違うだけで式は同じとする |
| 収束の比較 | 速い訓練と最終収束は別に評価する | Adamが常にSGDを上回ると断定する |

## 実装で確かめる

一次・二次モーメントとバイアス補正をNumPyでそのまま書きます。座標ごとに分母が変わるため、同じ $\alpha$ でも更新幅が変わることを確認できます。

```python
import numpy as np
g = np.array([1.0, 0.1])
theta = np.zeros(2); m = np.zeros(2); v = np.zeros(2)
alpha, b1, b2, eps = 0.001, 0.9, 0.999, 1e-8
for t in range(1, 3):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g**2
    mh = m / (1 - b1**t); vh = v / (1 - b2**t)
    step = alpha * mh / (np.sqrt(vh) + eps)
    theta -= step
print(theta)
```

## 取り違えやすいもの

| 用語 | 切り分け |
|---|---|
| [モメンタム](/learn/e-shikaku/momentum-optimization/) | 過去の勾配方向を蓄積する考え方。Adamでは一次モーメントとして使われる |
| [RMSProp系](/learn/e-shikaku/adaptive-optimizers/) | 二次モーメントで勾配をスケールする系統。Adamは一次モーメントと補正も組み合わせる |
| AdamW | 重み減衰を適応化された勾配更新から分離する実装上の系統 |
| [SGD](/learn/e-shikaku/sgd-and-minibatch/) | 座標別の二次モーメントによる正規化を持たない。Adamとの優劣は問題と評価時点で変わる |

## 想起チェック

<details class="recall">
<summary>Adamが使う二つのモーメントは何か</summary>

一次モーメントは勾配の移動平均、二次モーメントは勾配二乗の移動平均です。

</details>

<details class="recall">
<summary>初期ステップでバイアス補正が必要な理由は何か</summary>

モーメントをゼロから始めるため、初期の推定値がゼロ側に偏るからです。$1-\beta_1^t$ と $1-\beta_2^t$ で割って補正します。

</details>

<details class="recall">
<summary>AdamとAdamWの重み減衰の違いは何か</summary>

Adamは減衰項を勾配に混ぜて適応化し、AdamWは適応化した勾配更新と減衰の更新を分けます。

</details>
