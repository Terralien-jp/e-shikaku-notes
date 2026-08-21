---
exam: E資格
concept: 尤度とMAP推定
slug: likelihood-and-map
tier: B
area: 応用数学
summary: データを固定してパラメータを評価する尤度と、事前分布を加えて一点を選ぶMAP推定を、正則化との対応まで整理します。
updated: 2026-08-22
sources:
  - title: "Machine Learning Basics"
    url: https://www.deeplearningbook.org/contents/ml.html
  - title: "1.1. Linear Models"
    url: https://scikit-learn.org/stable/modules/linear_model.html
---

## ひとことで言うと

尤度は、観測したデータを固定し、パラメータを動かしたときにそのデータがどれだけ自然かを表す関数です。最尤推定（MLE）は尤度だけを最大にし、MAP推定は事前分布も掛けた事後分布の最大点を選びます。

<div class="analogy">

同じ観測記録を手元に置き、設定値の候補を一つずつ差し替えて「この設定なら、この記録はどれくらい納得できるか」を採点するのが尤度です。MAPはその採点に、設定値についての事前の好みも加えたものです。

</div>

## なぜ必要か

実装で混乱しやすいのは、確率分布の向きが入れ替わる点です。データが変数でパラメータを固定した $p(\mathcal{D}\mid\theta)$ はデータの確率モデルであり、データ $\mathcal{D}$ を固定して $\theta$ を変数と見る同じ式が尤度です。尤度はパラメータについて積分して1になる必要はありません。

MLEはデータへの適合だけで一点を決めます。データが少ないときや極端なパラメータを避けたいときは、パラメータに関する仮定を加えたいので、事前分布 $p(\theta)$ と尤度の積を最大にするMAPを使います。違いは目的関数に事前分布を含めるかどうかです。

| 読み方 | 固定するもの | 比較するもの |
|---|---|---|
| 確率 | $\theta$ | データ $\mathcal{D}$ |
| 尤度 | データ $\mathcal{D}$ | パラメータ $\theta$ |

## 仕組み

パラメータを $\theta$、観測データを $\mathcal{D}$、データの条件付き確率モデルを $p(\mathcal{D}\mid\theta)$ とします。尤度は

$$
L(\theta;\mathcal{D}) = p(\mathcal{D}\mid\theta)
$$

です。データを固定したまま $\theta$ の関数として読むのが要点です。通常は対数尤度 $\log L(\theta;\mathcal{D})$ を最大化します。独立な観測なら積が和になり、数値的にも扱いやすくなります。

ベイズの定理で事後分布は尤度と事前分布に比例するので、MAPは次を最大にする点です。

$$
\hat{\theta}_{\mathrm{MAP}} = \arg\max_{\theta}\; p(\mathcal{D}\mid\theta)p(\theta)
$$

対数を取り、最大化を最小化へ変えると、

$$
\hat{\theta}_{\mathrm{MAP}} = \arg\min_{\theta}\;[-\log p(\mathcal{D}\mid\theta)-\log p(\theta)]
$$

となります。ここで $\theta$ は推定したいパラメータ、$p(\theta)$ は事前分布です。正規分布の事前 $p(\theta)\propto\exp(-\lambda\lVert\theta\rVert_2^2/2)$ を置くと、$-\log p(\theta)$ は $\lambda\lVert\theta\rVert_2^2/2$ に定数を除いて一致します。したがってガウス事前のMAPは、負の対数尤度にL2型の項を加えた最適化として実装できます。これは正則化の一般論ではなく、事前分布を置いた結果として同じ目的関数が現れる、という対応です。

| 推定 | 最大化するもの | 事前分布 | 返すもの |
|---|---|---|---|
| MLE | $p(\mathcal{D}\mid\theta)$ | 使わない | 尤度最大の一点 |
| MAP | $p(\mathcal{D}\mid\theta)p(\theta)$ | 使う | 事後分布最大の一点 |
| 完全な事後推論 | $p(\theta\mid\mathcal{D})$ | 使う | 分布全体 |

MAPは事後分布の点推定にすぎません。事後分布の幅や複数の山を保持する推論とは同じではなく、最も高い一点以外の不確実性を捨てます。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| 尤度の変数を判定 | データを固定し、パラメータの関数として読む | 尤度はパラメータの確率分布なので積分1になるとする |
| MLEとMAPの比較 | MLEは尤度、MAPは尤度×事前分布を最大化する | MAPを事前分布だけの最大点とする |
| MAPを対数で書く | 負の対数尤度に負の対数事前分布を加える | 事前分布の項を足す符号を逆にする |
| ガウス事前の対応 | 負の対数事前分布が二乗ノルムの項になる | ガウス事前がL1型の項を直接生むとする |
| MAPの意味 | 事後分布の最大点という一点推定 | MAPが事後分布全体や予測分布そのものだとする |

## 実装で確かめる

観測値を固定し、平均パラメータ $\theta$ の候補を格子上で動かします。ガウス尤度だけの最大点がMLE、ガウス事前を加えた最大点がMAPです。コードでは確率の積が小さくなりすぎないよう対数で計算します。

```python
import numpy as np

y = np.array([0.8, 1.0, 1.2])
sigma, tau = 0.5, 1.0
theta = np.linspace(-1.0, 2.0, 3001)
log_likelihood = -((y[:, None] - theta) ** 2).sum(axis=0) / (2 * sigma**2)
log_prior = -(theta**2) / (2 * tau**2)
mle = theta[np.argmax(log_likelihood)]
map_est = theta[np.argmax(log_likelihood + log_prior)]
print(round(mle, 3), round(map_est, 3))
```

この出力は `1.0 0.923` です。データの平均が1.0なのでMLEはそこに置かれますが、0に近い値を好む事前を加えるとMAPは0側へ移ります。データが増えれば事前の影響は相対的に小さくなります。

## 取り違えやすいもの

| 用語 | 今回の切り分け |
|---|---|
| 確率 | パラメータを固定して、データの起こりやすさを読む |
| 尤度 | データを固定して、パラメータ候補を比較する |
| MLE | 事前分布なしで尤度を最大化する点推定 |
| MAP | 事前分布込みで事後分布の最大点を選ぶ点推定 |
| ガウス事前 | 負の対数を取ると二乗ノルム型の項になる |

## 想起チェック

<details class="recall">
<summary>尤度では何を固定し、何を動かして読むか</summary>

観測データ $\mathcal{D}$ を固定し、パラメータ $\theta$ を動かして $p(\mathcal{D}\mid\theta)$ を比較します。

</details>

<details class="recall">
<summary>MLEとMAPを分けるものは何か</summary>

MLEは尤度だけ、MAPは尤度に事前分布を掛けた事後分布を最大化します。

</details>

<details class="recall">
<summary>ガウス事前がMAPの目的関数に与える項は何か</summary>

負の対数を取ると、定数を除いて二乗ノルム型の項になります。そのためガウス事前のMAPはL2型の正則化を加えた最適化として書けます。

</details>
