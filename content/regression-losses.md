---
exam: E資格
concept: 回帰損失
slug: regression-losses
tier: B
area: 深層学習
summary: 回帰損失は外れ値への感度と、学習後に何を推定したいかで選びます。二乗誤差は条件付き平均、絶対誤差は条件付き中央値に対応し、Huber系は勾配の扱いやすさを両立します。
updated: 2026-08-22
sources:
  - title: "SmoothL1Loss"
    url: https://docs.pytorch.org/docs/2.13/generated/torch.nn.SmoothL1Loss.html
  - title: "Fast R-CNN"
    url: https://arxiv.org/abs/1504.08083
---

## ひとことで言うと

回帰損失は、予測値と正解値のずれをどのように罰するかを決める関数です。外れ値をどれだけ学習に効かせるかと、平均・中央値のどちらを推定したいかを同時に選びます。

<div class="analogy">

極端な一件を強く反映させるなら二乗誤差、距離一つ分として扱うなら絶対誤差です。Huber系は、近い誤差には二乗、遠い誤差には絶対値を使います。

</div>

## なぜ必要か

二乗誤差（MSE）は誤差 $e$ を $e^2$ として扱うので、外れ値の影響が急増します。勾配も誤差に比例して大きくなりますが、小さい領域では滑らかです。絶対誤差（MAE）は外れ値に鈍感で、勾配の大きさが概ね一定です。ただし $e=0$ で折れ曲がります。

損失の選択は推定対象まで変えます。入力を固定した真値の分布では、MSEを最小にする予測は条件付き平均、MAEを最小にする予測は条件付き中央値です。「典型的な値」か、二乗の意味で平均的な値かを先に決めます。

<div class="caution">

外れ値が観測ノイズならMAEやHuber型が候補ですが、予測対象の一部なら情報まで弱めます。

</div>

## 仕組み

予測を $\hat{y}$、正解を $y$、残差を $e=\hat{y}-y$ とします。

$$
L_{\mathrm{MSE}}=e^2,\qquad L_{\mathrm{MAE}}=|e|
$$

$\hat{y}$ は予測、$y$ は正解、$e$ は残差です。MSEの微分は $2e$、MAEの微分は $e\ne0$ で $\operatorname{sign}(e)$ です。

Huber型（Smooth L1）は、しきい値 $\beta$ を境に切り替えます。$\beta$ は二乗領域とL1領域の境界です。

$$
L_{\mathrm{Huber}}(e)=
\begin{cases}
\dfrac{e^2}{2\beta} & (|e|<\beta)\\
|e|-\dfrac{\beta}{2} & (|e|\ge\beta)
\end{cases}
$$

内側では二乗誤差の滑らかさ、外側ではL1の外れ値耐性を使います。境界で傾きがつながり、外側の勾配の大きさは1に抑えられます。PyTorchのSmoothL1Lossでも `beta` がL1と二乗項を切り替えるしきい値です。Fast R-CNNは境界ボックス回帰でこの損失を使い、L2より外れ値に敏感でなく、非有界な回帰対象での勾配爆発への感度を抑える理由を示しています。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| MSEとMAEの推定対象 | 条件付き平均／条件付き中央値 | 「どちらも平均」 |
| 外れ値と勾配 | MSEは残差に比例、MAEは大きさが飽和 | MAEの勾配も残差に比例 |
| Huberの領域 | 小さい誤差は二乗、大きい誤差はL1 | 全域で二乗、または全域でL1 |

## 実装で確かめる

次のコードは、同じ残差に対する3損失と勾配をNumPyで並べます。Huber型は `beta=1` とし、外れ値で勾配が増幅されないことを確認できます。

```python
import numpy as np

e = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
mse, gmse = e**2, 2 * e
mae, gmae = np.abs(e), np.sign(e)
beta = 1.0
huber = np.where(np.abs(e) < beta, e**2 / (2 * beta), np.abs(e) - beta / 2)
ghuber = np.where(np.abs(e) < beta, e / beta, np.sign(e))
print("MSE:", mse, "grad:", gmse)
print("MAE:", mae, "grad:", gmae)
print("Huber:", huber, "grad:", ghuber)
```

## 取り違えやすいもの

| 用語 | 切り分け |
|---|---|
| MSE | 外れ値を強く反映し、条件付き平均を推定する損失 |
| MAE | 外れ値に頑健で、条件付き中央値を推定する損失 |
| Huber／Smooth L1 | 残差の小さい領域を二乗、大きい領域をL1にする損失 |
| 評価指標 | 学習で勾配を作る損失とは目的が異なる。値の比較方法は別途設計する |

## 想起チェック

<details class="recall">
<summary>MSEとMAEは、それぞれ何を推定するか</summary>

MSEは条件付き平均、MAEは条件付き中央値です。

</details>

<details class="recall">
<summary>Huber型が外れ値に強い理由は何か</summary>

大きな残差ではL1領域に切り替わり、勾配の大きさが残差に比例して増え続けないためです。

</details>

<details class="recall">
<summary>beta は何を決めるか</summary>

$\beta$ は二乗誤差からL1へ切り替える残差のしきい値です。

</details>
