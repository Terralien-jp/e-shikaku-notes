---
exam: E資格
concept: ベイズ推論
slug: bayesian-inference
tier: A
area: 応用数学
summary: パラメータを1点に決めず、事前分布をデータの尤度で更新して事後分布を得る推論の枠組みです。
updated: 2026-08-22
sources:
  - title: "Deep Learning: Probability and Information Theory"
    url: https://www.deeplearningbook.org/contents/prob.html
  - title: "Deep Learning: Machine Learning Basics"
    url: https://www.deeplearningbook.org/contents/ml.html
---

## ひとことで言うと

ベイズ推論は、未知のパラメータを確率変数として扱い、データを見る前の信念（事前分布）を、観測データがどれだけ起こりやすいか（尤度）で更新する枠組みです。更新後に得るのが事後分布であり、パラメータ推定の答えを1個の値ではなく不確実性込みで保持します。

<div class="analogy">

故障診断で、最初は各原因の候補に見込みを置き、センサーの観測が各原因のもとでどれだけ自然かを掛けて、観測後の候補の重みを更新するようなものです。原因を一つに決め打ちせず、候補の分布を次の判断へ渡します。

</div>

## なぜ必要か

最尤推定（MLE）は、観測データを最も生じやすくするパラメータを1点で選びます。データが少ない、ノイズが大きい、複数のパラメータが同程度に妥当、といった状況では、その一点だけを次の予測に使うと「どれくらい迷っているか」が消えます。

ベイズ推論では、データを見る前のパラメータの分布 $p(\theta)$ とデータの尤度 $p(\mathcal{D}\mid\theta)$ を組み合わせます。事前分布は、たとえば極端に大きい係数を好まないという仮定を表せます。データを観測すると、尤度の高い領域へ確率質量が移り、事後分布が得られます。

| 推定 | 持つもの | データが少ないときの特徴 |
|---|---|---|
| MLE | パラメータの一点 $\hat{\theta}_{\mathrm{MLE}}$ | 尤度だけで決める |
| MAP | 事後分布の最大点 $\hat{\theta}_{\mathrm{MAP}}$ | 事前分布の影響を一点に反映する |
| ベイズ推論 | パラメータの分布 $p(\theta\mid\mathcal{D})$ | 推定の不確実性を保持する |

つまり、MLEやMAPとの本質的な違いは「分布を持つか、点を持つか」です。MAPはベイズ的な分布を要約する点推定であって、事後分布そのものではありません。

## 仕組み

パラメータを $\theta$、観測データを $\mathcal{D}$ とします。ベイズの定理は次の形です。

$$
p(\theta\mid\mathcal{D}) = \frac{p(\mathcal{D}\mid\theta)p(\theta)}{p(\mathcal{D})}
$$

$p(\theta)$ は事前分布、$p(\mathcal{D}\mid\theta)$ は尤度、$p(\theta\mid\mathcal{D})$ は事後分布です。分母の $p(\mathcal{D})$ は周辺尤度（エビデンス）と呼ばれ、事後分布が確率分布として積分1になるように全体を正規化します。

$$
p(\mathcal{D}) = \int p(\mathcal{D}\mid\theta)p(\theta)\,d\theta
$$

離散パラメータなら積分は総和になります。分子は「データに合うこと」と「事前の好み」の積、分母はその積をパラメータ全域で足し合わせたものです。したがって $p(\mathcal{D})$ を計算せずに事後分布の形だけ欲しい場合は、$p(\theta\mid\mathcal{D}) \propto p(\mathcal{D}\mid\theta)p(\theta)$ と書けます。ただし、モデル比較や正規化された確率が必要なとき、周辺尤度を無視できません。

予測も点推定とは異なります。新しい入力に対応する値を $x_*$、未知の出力を $y_*$ とすると、予測分布は事後分布にわたる平均です。

$$
p(y_*\mid x_*,\mathcal{D}) = \int p(y_*\mid x_*,\theta)p(\theta\mid\mathcal{D})\,d\theta
$$

各パラメータが出す予測を、事後分布の重みで平均しています。MLEならこの積分を $\theta=\hat{\theta}_{\mathrm{MLE}}$ の一点で評価する形になります。ベイズ推論はパラメータの不確実性を予測へ伝播させるため、事後分布が広いと予測もその不確実性を含みます。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| ベイズの定理の各項 | 事後＝尤度×事前÷周辺尤度 | 分母を尤度や事前と取り違える |
| MLE・MAP・ベイズ推論の比較 | 前二者は点、ベイズ推論は事後分布を持つ | MAPを事後分布そのものとする |
| 周辺尤度の意味 | パラメータを積分消去したデータの確率、正規化項 | 観測データに条件付けた尤度と同一視する |
| ベイズ予測の式 | $p(y_*\mid x_*,\mathcal{D})$ を事後分布で積分する | 事後分布の平均パラメータだけで評価する |
| 計算量の説明 | 高次元積分や正規化が難しく、近似が必要になる | ベイズ推論なら常に解析解が得られるとする |

## 実装で確かめる

コインの表の確率 $\theta$ に一様事前分布を置き、表が出た回数と裏が出た回数で事後分布を更新します。Beta分布はこの尤度と共役なため、正規化を数値積分せずに更新できます。

```python
import numpy as np

heads, tails = 7, 3
grid = np.linspace(0.001, 0.999, 1000)
prior = np.ones_like(grid)                 # Beta(1, 1)
likelihood = grid ** heads * (1 - grid) ** tails
posterior = prior * likelihood
posterior /= np.trapezoid(posterior, grid)

mean_theta = np.trapezoid(grid * posterior, grid)
predictive_head = np.trapezoid(grid * posterior, grid)
print(round(mean_theta, 3), round(predictive_head, 3))
```

ここでは事後分布の平均と、次の1回が表になる予測確率が同じ積分になります。点推定なら最も高い場所だけを返しますが、この計算はグリッド上の全候補を事後分布の重みで平均しています。

<div class="caution">

連続分布の密度は確率そのものではありません。確率は区間で積分して得ます。また尤度は $\theta$ の関数として見たデータの評価であり、$\theta$ について積分して1になる分布だとは限りません。

</div>

## 取り違えやすいもの

| 用語 | ベイズ推論との切り分け |
|---|---|
| [最尤推定（MLE）](/learn/e-shikaku/parameter-estimation/) | 尤度を最大にするパラメータの一点。事前分布を使わない |
| [MAP推定](/learn/e-shikaku/likelihood-and-map/) | 事後分布を最大にする一点。分布を要約した点であり、予測時の積分を省略する |
| [尤度](/learn/e-shikaku/likelihood-and-map/) | データを固定し、パラメータを変えたときの $p(\mathcal{D}\mid\theta)$ |
| 周辺尤度 | 尤度と事前分布をパラメータ全域で積分した $p(\mathcal{D})$ |
| 予測分布 | 事後分布を重みにして、パラメータごとの予測を平均したもの |

## 想起チェック

<details class="recall">
<summary>ベイズの定理で事後分布を作る三つの要素は何か</summary>

事前分布 $p(\theta)$、尤度 $p(\mathcal{D}\mid\theta)$、そして正規化を担う周辺尤度 $p(\mathcal{D})$ です。

</details>

<details class="recall">
<summary>MLEやMAPとベイズ推論の違いを一言で分けると何か</summary>

MLEとMAPはパラメータの一点を返します。ベイズ推論は事後分布というパラメータの分布を保持します。

</details>

<details class="recall">
<summary>ベイズ予測で事後分布を積分する理由は何か</summary>

事後確率が高いパラメータだけでなく、観測後も残る全候補の予測を、その事後分布を重みにして平均するためです。

</details>

<details class="recall">
<summary>近似推論が必要になる直接の理由は何か</summary>

周辺尤度や予測分布に現れる高次元の積分と正規化が、モデルによっては解析的に計算できないためです。

</details>
