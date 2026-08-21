---
exam: E資格
concept: モメンタム法
slug: momentum-optimization
tier: B
area: 深層学習
summary: 過去の更新方向を速度として蓄積し、細長い谷での振動を抑えながら勾配降下を進める方法です。
updated: 2026-08-22
sources:
  - title: "Deep Learning: Optimization for Training Deep Models"
    url: https://www.deeplearningbook.org/contents/optimization.html
  - title: "SGD"
    url: https://docs.pytorch.org/docs/2.13/generated/torch.optim.SGD.html
---

## ひとことで言うと

モメンタム法は、現在の勾配だけでなく、過去の更新方向を速度として蓄積してパラメータを動かす最適化法です。勾配が左右に変わりやすい方向は相殺され、同じ向きが続く方向は加速されます。

<div class="analogy">

スキー板が細長い谷を滑る場面に似ています。谷を横切る傾斜は左右で反転するため、そのたびに受けた力を平均すれば揺れが小さくなります。一方、谷に沿う力は同じ向きに残るので、停止と再加速を繰り返さず進めます。

</div>

## なぜ必要か

通常の更新 $\theta_{t+1}=\theta_t-\eta g_t$ では、$\theta_t$ はパラメータ、$\eta$ は学習率、$g_t$ は時刻 $t$ の勾配です。

$$
\boldsymbol{\theta}_{t+1}=\boldsymbol{\theta}_t-\eta\mathbf{g}_{t+1}
$$

曲率の大きい方向と小さい方向が混在する悪条件の谷では、急な横方向の勾配に反応して左右へ振動し、谷に沿った前進が遅くなります。ミニバッチの勾配ノイズでも、毎回の勾配だけを信じると進行方向が不安定です。

モメンタムは勾配の履歴を1個のバッファにまとめます。反対向きの成分は減り、連続する成分は残るため、勾配を単純に小さくするのではなく、更新の向きを時間方向に平滑化します。ただし履歴を持つぶん、勾配が変わった直後もすぐには止まらず、学習率との組合せによっては行き過ぎます。

したがって損失が一時的に下がっても、更新幅や速度のノルムが増え続けていないかを別に確認します。谷の向きが変わる局面では、係数を下げるか学習率を見直す判断につながります。

## 仕組み

速度 $\mathbf{v}_t$ を、$\mathbf{g}_t$ を時刻 $t$ の勾配、$\mu$ をモメンタム係数、$\eta$ を学習率として次で更新します。

$$
\mathbf{v}_{t+1}=\mu\mathbf{v}_t+\mathbf{g}_{t+1},\qquad
\boldsymbol{\theta}_{t+1}=\boldsymbol{\theta}_t-\eta\mathbf{v}_{t+1}
$$

$\boldsymbol{\theta}_t$ はパラメータ、$\mathbf{v}_t$ は蓄積した速度です。$0\leq\mu<1$ なら、展開した速度は過去の勾配に $1,\mu,\mu^2,\ldots$ の重みを付けた和になります。したがって指数移動平均として見られ、$\mu$ が大きいほど長い履歴を保ちます。一定の勾配 $\mathbf{g}$ が続く理想化では、速度はおよそ $\mathbf{g}/(1-\mu)$ に近づきます。$\mu=0.9$ なら定常速度の倍率は10ですが、これは学習率を無条件に10倍にしてよいという意味ではありません。初期の速度は履歴がまだ短いため、序盤の挙動と定常状態を同じ感覚で読まないことも必要です。

実装では流儀に注意が要ります。PyTorchの `SGD` は上のように学習率を最後のパラメータ更新へ掛けますが、別の流儀では速度側に学習率を含めます。式を見ずに係数を移植すると、同じ `momentum=0.9` でも実効的な移動量が変わります。Nesterov版は、速度を先に適用した位置で勾配を測る形です。加速の理論的性質はここでは扱いません。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| 更新則の意味 | 速度に過去の速度と現在の勾配を混ぜる | 現在の勾配だけで更新する |
| 係数 $\mu$ の解釈 | 大きいほど履歴を長く保持する | 学習率そのものだとみなす |
| 谷での挙動 | 横方向の振動を抑え、長手方向を進みやすくする | どの方向の収束も必ず速くなると断定する |
| Nesterovとの違い | 先に進んだ位置で勾配を評価する | 通常のモメンタムと同じ位置で測る |

## 実装で確かめる

1次元の二次関数で、速度が勾配の履歴を蓄積する様子を確認します。mu=0 は通常の勾配降下に一致します。コードではパラメータを1個に絞り、速度の役割だけを追えるようにしています。

```python
import numpy as np

theta = np.array([3.0])
velocity = np.zeros_like(theta)
eta, mu = 0.1, 0.9
for _ in range(5):
    grad = theta                 # f(theta)=theta^2/2 の勾配
    velocity = mu * velocity + grad
    theta = theta - eta * velocity
print(theta, velocity)
```

ここで `velocity` を毎回ゼロに戻すと履歴が消え、モメンタムの効果も消えます。実際のライブラリでは、このバッファはパラメータごとの optimizer state として保持されます。PyTorchの仕様では初回バッファはゼロではなく初回勾配から始まり、dampening は2回目以降に適用されます。

## 取り違えやすいもの

| 手法 | モメンタム法との切り分け |
|---|---|
| [SGD](/learn/e-shikaku/sgd-and-minibatch/) | 現在の勾配だけで更新する。モメンタムはSGDに履歴バッファを加えた形 |
| [Nesterovモメンタム](/learn/e-shikaku/nesterov-acceleration/) | 先に速度を進めた位置で勾配を評価する。通常版との違いは評価位置 |
| [Adamなど](/learn/e-shikaku/adaptive-optimizers/) | 勾配の履歴に加えて成分ごとのスケーリングも行う。適応的最適化の論点 |
| 学習率減衰 | $\eta$ を時間で変える仕組みで、履歴を蓄積するモメンタムとは別の機構 |

## 想起チェック

<details class="recall">
<summary>モメンタムが細長い谷の横揺れを抑える理由は</summary>

反転しやすい横方向の勾配は履歴の加算で相殺され、同じ向きの谷に沿う成分は蓄積されるためです。

</details>

<details class="recall">
<summary>モメンタム係数を大きくすると何が変わるか</summary>

過去の勾配を長く保持します。応答が滑らかになる一方、方向転換後の停止は遅くなり、行き過ぎにも注意が要ります。

</details>

<details class="recall">
<summary>Nesterov版はどこで勾配を測るか</summary>

現在の速度を先に適用した位置で測ります。通常のモメンタムとの実装上の差は、勾配の評価位置です。

</details>
