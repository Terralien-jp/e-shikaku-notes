---
exam: E資格
concept: 回帰性能指標
slug: regression-metrics
tier: B
area: 機械学習
summary: 回帰予測の誤差を、外れ値・単位・平均予測との比較という異なる観点から読むための指標整理です。
updated: 2026-08-22
sources:
  - title: "3.4. Metrics and scoring: quantifying the quality of predictions"
    url: https://scikit-learn.org/stable/modules/model_evaluation.html
  - title: "mean_squared_error"
    url: https://scikit-learn.org/stable/modules/generated/sklearn.metrics/mean_squared_error.html
---

## ひとことで言うと

回帰性能指標は、予測値と正解値のずれを、業務上どの誤りとして数えるかを決める物差しです。MSE・RMSE・MAEは誤差の大きさを測り、R²は「目的変数の平均を出すだけの予測」と比べてどれだけ改善したかを測ります。指標と損失は同じ式になることがありますが、指標は評価・比較のため、損失は学習で更新方向を作るための量です。

<div class="analogy">

同じ走行記録でも、平均速度、最大速度、燃費では見えるものが違います。回帰指標も、全体の大きな外れを重く見るか、典型的なずれをそのまま見るか、平均予測を基準にするかで結論が変わります。

</div>

## なぜ必要か

「誤差が小さいモデル」を一語で決められないからです。大きな誤差を二乗して強く罰したいならMSE、結果を元の単位で読みたいならRMSE、外れ値一件に評価を支配させたくないならMAEが候補になります。MSEは二乗によって単位も二乗になりますが、RMSEは平方根を取るので目的変数と同じ単位に戻ります。

さらに、値の桁が異なる目的変数を比率で評価したいときはMAPEが候補です。ただし分母が正解値なので、ゼロでは定義できず、ゼロに近い値でも比率が極端になります。平均的なケースだけを見ていると、この事故を見落とします。

<div class="caution">

検証データでMSEが最小でも、現場の許容誤差が「平均して何単位ずれるか」ならMAEやRMSEを併記します。評価指標はモデルの順位だけでなく、失敗の扱い方まで選んでいます。

</div>

## 仕組み

正解値を $y_i$、予測値を $\hat{y}_i$、サンプル数を $n$、誤差を $e_i=y_i-\hat{y}_i$ とします。

$$
\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}e_i^2,\qquad
\mathrm{RMSE}=\sqrt{\mathrm{MSE}},\qquad
\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|e_i|
$$

MSEとRMSEは外れ値に敏感です。誤差が2倍ならMSEへの寄与は4倍になるためです。MAEは誤差に比例するので、外れ値の影響は相対的に穏やかです。これは「どれが正しいか」ではなく、評価したい失敗のコストがどちらに近いかの違いです。

R²は残差平方和を、平均予測からの偏差平方和と比べます。$\bar{y}$ は正解値の平均、$\mathrm{SSE}$ は予測の残差平方和です。

$$
R^2=1-\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}
$$

平均予測と同じならR²は0、完全予測なら1です。平均予測より悪ければ負にもなります。したがってR²は誤差の絶対量そのものではなく、データセット内の基準予測に対する相対評価です。分母がデータのばらつきなので、データセットをまたいだ単純比較にも向きません。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| MSEとMAEの比較 | MSEは外れ値を強く重視、MAEは比例的 | MAEも外れ値を無視する、とする |
| RMSEの単位 | 目的変数と同じ単位 | MSEも同じ単位とする |
| R²の解釈 | 平均予測を基準にした相対的な説明度 | 常に0以上、データ間で絶対比較できるとする |
| MAPEの注意 | 正解値が0または0近傍だと不安定 | 「パーセントだから常に公平」とする |
| scikit-learnのCV | 高い値を良いとするためMSE系は負号名になる | `neg_mean_squared_error`を誤差が負だと解釈する |

## 実装で確かめる

一つの外れ値と小さい値を含むデータで、指標が違う結論を出す例です。

```python
import numpy as np

y = np.array([1., 2., 3., 20.])
pred_a = np.array([1., 2., 3., 10.])
pred_b = np.array([2., 3., 4., 19.])
for name, pred in [("A", pred_a), ("B", pred_b)]:
    e = y - pred
    mse = np.mean(e ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(e))
    r2 = 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
    mape = np.mean(np.abs(e) / np.abs(y))
    print(name, mse, rmse, mae, round(r2, 3), round(mape, 3))
```

出力は `A 25.0 5.0 2.5 0.592 0.125`、`B 1.0 1.0 1.0 0.984 0.471` です。MSE・RMSE・MAE・R²はBを選びますが、MAPEは小さい正解値への相対誤差を重く見てAを選びます。指標を一つだけ見て「性能」と呼ぶと、評価したい失敗と数字の計算がずれます。

## 取り違えやすいもの

| 用語 | 切り分け |
|---|---|
| [MSE](/learn/e-shikaku/mse-and-mae/) | 二乗誤差の平均。外れ値を強く罰し、単位は目的変数の二乗 |
| RMSE | MSEの平方根。外れ値への感度は残しつつ、単位は元に戻る |
| [MAE](/learn/e-shikaku/mse-and-mae/) | 絶対誤差の平均。典型的なずれを元の単位で読む |
| R² | 平均予測を基準にした相対指標。負にもなり、データの分散に依存 |
| MAPE | 正解値で割った相対誤差。ゼロ近傍で不安定 |

## 想起チェック

<details class="recall">
<summary>MSEとMAEで外れ値の影響が違う理由は</summary>

MSEは誤差を二乗するため、誤差が大きいサンプルの寄与が急増します。MAEは絶対値なので誤差に比例します。

</details>

<details class="recall">
<summary>R²が負になるのはどんなときか</summary>

予測の残差平方和が、正解値の平均を常に予測する基準モデルの平方和を上回ったときです。

</details>

<details class="recall">
<summary>MAPEをゼロ近傍で使うと何が起きるか</summary>

分母が小さいため、絶対誤差が小さくても相対誤差が極端に大きくなります。正解値が0なら通常の式は定義できません。

</details>
