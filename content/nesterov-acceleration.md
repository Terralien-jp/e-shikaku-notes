---
exam: E資格
concept: Nesterov加速
slug: nesterov-acceleration
tier: B
area: 深層学習
summary: モメンタムで先に進んだ位置の勾配を使い、行き過ぎを早めに補正する最適化手法。
updated: 2026-08-22
sources:
  - title: "SGD"
    url: https://docs.pytorch.org/docs/2.13/generated/torch.optim.SGD.html
  - title: "Practical recommendations for gradient-based training of deep architectures"
    url: https://arxiv.org/abs/1206.5533
---

## ひとことで言うと

Nesterov加速は、モメンタムの移動量で少し先へ進んだ位置を仮に作り、その位置で勾配を評価して更新する方法です。現在位置の勾配だけで進む通常のモメンタムより、進み過ぎへの補正が早く入ります。

<div class="analogy">

坂道を下る車が、いまいる場所だけでなく、惰性で進んだ先の路面を見てハンドルを切るイメージです。先の傾きを使うので、曲がり始めの判断が遅れません。

</div>

## なぜ必要か

@@
 Nesterov加速では、まず勢いによる予測位置を作り、その予測位置の勾配を次の更新に使います。したがって、モメンタムを捨てずに、先回りした勾配で行き過ぎを抑えます。これは学習率やモメンタム係数を別の最適化手法へ置き換える話ではなく、勾配をどこで評価するかの違いです。

| 手法 | 勾配を取る場所 | 補正のタイミング |
|---|---|---|
| 通常のモメンタム | 現在位置 | 勢いを加えた後 |
| Nesterov加速 | 勢いで進んだ先 | 更新前 |


Nesterov加速では、まず勢いによる予測位置を作り、その予測位置の勾配を次の更新に使います。したがって、モメンタムを捨てずに、先回りした勾配で行き過ぎを抑えます。これは学習率やモメンタム係数を別の最適化手法へ置き換える話ではなく、勾配をどこで評価するかの違いです。

## 仕組み

$\theta_t$ を時刻 $t$ のパラメータ、$\mathbf{v}_t$ を蓄積する速度、$\eta$ を学習率、$\mu$ をモメンタム係数、$L$ を損失とします。先読み位置 $\widetilde{\theta}_t$ を作り、そこで勾配を取ります。

$$
\widetilde{\theta}_t = \theta_t + \mu\mathbf{v}_t, \qquad
\mathbf{g}_t = \nabla L(\widetilde{\theta}_t)
$$

速度とパラメータを次のように更新します。

$$
\mathbf{v}_{t+1}=\mu\mathbf{v}_t-\eta\mathbf{g}_t, \qquad
\theta_{t+1}=\theta_t+\mathbf{v}_{t+1}
$$

$\mathbf{g}_t$ は先読み位置での損失勾配です。実装では、現在のパラメータを直接書き換えず、一時的な先読み値を作って勾配計算後に元の更新を行います。PyTorchのSGDでは `momentum` が0でないときに `nesterov=True` を指定する形で、この方式を選べます。係数の定義や初回ステップの扱いは実装の更新則を確認してください。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| 通常のモメンタムとの差 | 勾配を現在位置ではなく、勢いで進んだ先で評価する | 勢いを使わず毎回先読みするだけとする |
| 更新式の穴埋め | $\widetilde{\theta}_t=\theta_t+\mu\mathbf{v}_t$ を先に作る | $\theta_t-\mu\mathbf{v}_t$ と符号を逆にする |
| 実装設定の確認 | `momentum` を有効にして `nesterov=True` とする | Nesterovだけを単独で有効化できるとする |

## 実装で確かめる

1変数の二次関数で、先読み位置の勾配を使う更新をそのまま実装します。$\theta$ はパラメータ、$L(\theta)=\theta^2/2$ なので勾配は $\theta$ です。

```python
import numpy as np

theta, velocity = 4.0, 0.0
eta, mu = 0.1, 0.9
for _ in range(20):
    lookahead = theta + mu * velocity
    grad = lookahead
    velocity = mu * velocity - eta * grad
    theta += velocity
print(theta)
```

このコードを実行すると `0.4904749238276017` と表示されます。`grad = theta` に変えると、先読みをしない通常のモメンタムとの違いを同じ条件で比較できます。

## 取り違えやすいもの

| 用語 | 勾配を評価する位置 | 見分けるポイント |
|---|---|---|
| [通常のモメンタム](/learn/e-shikaku/momentum-optimization/) | 現在の $\theta_t$ | 現在の勾配に過去の速度を組み合わせる |
| Nesterov加速 | 先読みした $\widetilde{\theta}_t$ | 勢いを加えた仮の位置で勾配を取る |
| [SGD](/learn/e-shikaku/sgd-and-minibatch/) | 現在の $\theta_t$ | 速度の蓄積を使わず、勾配だけで更新する |
| [Adam系](/learn/e-shikaku/adam-family/) | 実装ごとの評価位置 | 一次・二次モーメントを使う別系統の更新則 |

## 想起チェック

<details class="recall">
<summary>Nesterov加速で、通常のモメンタムと最も違う計算箇所はどこか</summary>

勾配を現在位置ではなく、モメンタムによる移動を加えた先読み位置で評価する点です。

</details>

<details class="recall">
<summary>先読み位置を表す式は何か</summary>

$\widetilde{\theta}_t=\theta_t+\mu\mathbf{v}_t$ です。$\theta_t$ はパラメータ、$\mathbf{v}_t$ は速度、$\mu$ はモメンタム係数です。

</details>

<details class="recall">
<summary>PyTorchでNesterovを有効にする前提は何か</summary>

SGDの `momentum` を有効にしたうえで、`nesterov=True` を指定します。

</details>
