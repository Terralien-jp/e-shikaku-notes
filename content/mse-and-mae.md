---
exam: E資格
concept: MSEとMAE
slug: mse-and-mae
tier: B
area: 深層学習
summary: 二乗誤差と絶対誤差の性質を、外れ値・推定される代表値・勾配の違いから使い分ける。
updated: 2026-08-22
sources:
  - title: "MSELoss"
    url: https://docs.pytorch.org/docs/2.13/generated/torch.nn.MSELoss.html
  - title: "L1Loss"
    url: https://docs.pytorch.org/docs/2.13/generated/torch.nn.L1Loss.html
---

## ひとことで言うと

MSE（平均二乗誤差）は予測誤差を二乗して平均し、MAE（平均絶対誤差）は絶対値を取って平均する回帰損失です。MSE は大きな誤差を急激に重く扱い、MAE は誤差の大きさに比例して扱います。

<div class="analogy">

MSE は遅刻の罰金が遅刻時間の二乗で増える制度、MAE は遅刻時間に比例する制度です。1分の遅刻が増えたとき、前者はすでに大きく遅れた人をさらに強く追い立てます。

</div>

## なぜ必要か

回帰では、正負の誤差をそのまま足すと相殺されます。そこで誤差を常に非負にする必要があり、二乗と絶対値が代表的な選択肢になります。選択は「どちらが正しいか」ではなく、外れ値をどの程度学習に反映したいかで決まります。

同じデータでも、損失を変えるとモデルが寄せる先が変わります。条件付き分布の中心を平均で表したいなら MSE、中央値で表したいなら MAE です。目的変数の単位を保ちたい場合は MAE の解釈が直接的ですが、MAE は誤差ゼロで傾きが切り替わるため、最適化の扱いも確認します。

| 損失 | 誤差への重み | 外れ値の影響 | 推定する中心 |
|---|---|---|---|
| MSE | 二乗で増える | 大きい | 条件付き平均 |
| MAE | 絶対値に比例 | 比較的小さい | 条件付き中央値 |

## 仕組み

予測を $\hat{y}$、正解を $y$、誤差を $e=\hat{y}-y$、要素数を $N$ とします。PyTorch の既定の `mean` は要素ごとの損失を平均し、`sum` なら合計、`none` なら要素ごとの値を残します。

$$
L_{\mathrm{MSE}}=\frac{1}{N}\sum_{i=1}^{N}e_i^2,\qquad L_{\mathrm{MAE}}=\frac{1}{N}\sum_{i=1}^{N}|e_i|
$$

MSE の勾配は $\partial e_i^2/\partial e_i=2e_i$ なので、誤差が10倍なら勾配の大きさも10倍です。一方、MAE の勾配は $e_i>0$ で $1$、$e_i<0$ で $-1$ です。したがって MAE は大きな外れ値に勾配を占有されにくい反面、$e_i=0$ では微分できません。実装では原点の勾配を便宜的に0などと定めますが、そこが二乗誤差との明確な違いです。

一定値 $a$ を予測するとき、MSE の期待損失を $\mathbb{E}[(Y-a)^2\mid X=x]$ と書けます。$a$ で微分して0と置くと $a=\mathbb{E}[Y\mid X=x]$、つまり条件付き平均になります。MAE の期待損失 $\mathbb{E}[|Y-a|\mid X=x]$ は、左右にある確率質量が釣り合う点で最小になり、条件付き中央値を選びます。

Huber 損失は、しきい値 $\delta>0$（外れ値とみなす境界）を使い、$|e|\leq\delta$ では二乗、$|e|>\delta$ では絶対値に近い線形へ切り替えます。原点付近の滑らかさと、大きな誤差への頑健さを同時に欲しいときの折衷です。

$$
L_{\delta}(e)=\begin{cases}\frac{1}{2}e^2 & |e|\leq\delta\\ \delta\left(|e|-\frac{1}{2}\delta\right) & |e|>\delta\end{cases}
$$

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| 外れ値への反応 | MSE は二乗により強く反応、MAE は比例 | 「MSE のほうが頑健」 |
| 最小化で得る代表値 | MSE は条件付き平均、MAE は条件付き中央値 | どちらも平均とする |
| 微分の比較 | MAE は誤差0で微分不能、MSE は滑らか | MAE の勾配も誤差に比例とする |
| Huber の位置づけ | 小誤差は二乗、大誤差は線形 | 全域で二乗または全域で絶対値 |
| API の出力 | `mean` は要素平均、`sum` は合計、`none` は未集約 | バッチ平均と要素平均を無条件に同一視 |

## 実装で確かめる

外れ値を追加したとき、MSE の増加が MAE より大きくなることを NumPy で確かめます。`mean` を明示し、テンソルの要素数で割る場所を曖昧にしないのが実装上の要点です。

```python
import numpy as np

error = np.array([1.0, -1.0, 2.0])
outlier = np.append(error, 10.0)
mse = lambda e: np.mean(e ** 2)
mae = lambda e: np.mean(np.abs(e))
print(mse(error), mae(error))
print(mse(outlier), mae(outlier))
assert mse(outlier) - mse(error) > mae(outlier) - mae(error)
```

## 取り違えやすいもの

| 用語 | 切り分け |
|---|---|
| RMSE | MSE の平方根。目的変数と同じ単位になるが、最小化する代表値の性質は MSE と同じ |
| MAE | L1 損失とも呼ばれる。外れ値の影響はMSEより小さいが、原点で滑らかでない |
| Huber | 二乗と絶対値を誤差の大きさで接続する損失。しきい値 $\delta$ が挙動を決める |
| `reduction='sum'` | 損失の種類ではなく集約方法。バッチサイズや要素数で値と勾配の尺度が変わる |

## 想起チェック

<details class="recall">
<summary>MSE と MAE は外れ値をどう扱うか</summary>

MSE は誤差を二乗するため大きな誤差を強く重く扱い、MAE は絶対値に比例して扱います。

</details>

<details class="recall">
<summary>一定値予測で MSE と MAE が推定する中心は何か</summary>

MSE は条件付き平均、MAE は条件付き中央値です。

</details>

<details class="recall">
<summary>MAE の原点で最適化上の注意点は何か</summary>

誤差ゼロでは絶対値が微分できません。実装では勾配を便宜的に選び、Huber のように原点近くを二乗で滑らかにする選択肢もあります。

</details>
