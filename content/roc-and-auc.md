---
exam: E資格
concept: ROCとAUC
slug: roc-and-auc
tier: B
area: 機械学習
summary: 分類スコアの閾値を動かしたときのTPRとFPRの関係をROC曲線で読み、順位付けの性能をAUCで要約します。
updated: 2026-08-22
sources:
  - title: "An introduction to ROC analysis"
    url: https://doi.org/10.1016/j.patrec.2005.10.010
  - title: "3.4. Metrics and scoring: quantifying the quality of predictions"
    url: https://scikit-learn.org/stable/modules/model_evaluation.html
---

## ひとことで言うと

ROC（Receiver Operating Characteristic）曲線は、二値分類器の閾値を動かし、各点の真陽性率（TPR）と偽陽性率（FPR）を $({\mathrm{FPR}},{\mathrm{TPR}})$ 平面に描いたものです。AUCは曲線下の面積で、閾値を決める前の順位付け性能を要約します。

<div class="analogy">

入場判定のつまみを少しずつ緩めると、通せる正しい人が増える一方で、通してはいけない人も混ざります。その全ての設定を一本の軌跡にしたのがROC曲線です。

</div>

## なぜ必要か

確率や決定値を返すモデルを、0.5などの1個の閾値だけで比較すると、閾値の選び方とモデルの順位付け能力が混ざります。ROCは閾値を動かしたときのトレードオフを一枚で見せるため、モデル選択と運用閾値の決定を分けて考えられます。

ROC上で良いモデルを選んでも、実際に採用する点が良いとは限りません。誤検知と見逃しの費用、処理できる件数を決めてから閾値を選びます。

| 分けて考える対象 | 見るもの |
|---|---|
| モデルの順位付け | ROC曲線とAUC |
| 運用での判定 | 許容FPRや見逃しコストに対応する閾値 |

## 仕組み

正例の件数を $P$、負例の件数を $N$、閾値を $\theta$ とし、スコア $s$ が $\theta$ 以上なら正例と判定します。TPRは正例を拾う割合、FPRは負例を誤って正例にする割合です。

$$
\mathrm{TPR}(\theta)=\frac{\mathrm{TP}(\theta)}{P},\qquad
\mathrm{FPR}(\theta)=\frac{\mathrm{FP}(\theta)}{N}
$$

$\theta$ を下げると通常はTPRもFPRも上がり、全件を正例にすれば $(1,1)$、全件を負例にすれば $(0,0)$ になります。AUCは、正例から無作為に1件、負例から無作為に1件選んだとき、正例のスコアが負例より高い確率と一致します。同点の扱いを含めると、一般には「高い確率」に同点の半分を加えた順位解釈です。

| ROCの位置 | 読み方 | 判断 |
|---|---|---|
| 左上に近い | TPRが高くFPRが低い | 多くの用途で望ましい |
| 対角線上 | ランダムな順位付けと同程度 | AUCはおおむね0.5 |
| 曲線全体 | 閾値を変えた順位付け | 運用閾値そのものではない |

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| ROCの軸を読む | 横軸FPR、縦軸TPR | 横軸を適合率にする |
| 閾値を下げた変化 | TPRとFPRはともに増えやすい | TPRだけが増えると断定する |
| AUCの意味 | 正例のスコアが負例より高い確率 | 採用閾値での正解率だと読む |
| 不均衡データの評価 | PR曲線も確認し、正例の少なさを反映させる | ROC-AUCだけで十分とする |

## 実装で確かめる

roc_auc_score には0/1の予測結果ではなく、閾値をまだ固定していないスコアを渡します。次の例では、閾値ごとのROC点とAUCをNumPyだけで計算します。

```
import numpy as np

y = np.array([1, 1, 0, 0])
s = np.array([0.9, 0.6, 0.8, 0.2])
points = []
for t in np.r_[np.inf, np.sort(np.unique(s))[::-1], -np.inf]:
    pred = s >= t
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
    points.append((fp / 2, tp / 2))
xs = np.array([p[0] for p in points]); ys = np.array([p[1] for p in points])
auc = np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2)
print(points, auc)
```

出力は [(0.0, 0.0), (0.0, 0.5), (0.5, 0.5), (0.5, 1.0), (1.0, 1.0), (1.0, 1.0)] 0.75 です。スコアの向きと正例ラベルを先に固定してください。

## 取り違えやすいもの

| 指標 | ROC-AUCとの違い |
|---|---|
| PR曲線 | 閾値に対する適合率と再現率を見る。正例が少ないとき、予測した正例の中身をROCより直接に反映しやすい |
| 正解率 | 1つの閾値での結果。クラス不均衡では多数派に引っ張られる |
| AUC | ROC全体の要約。特定の閾値での性能や確率の校正を表さない |

不均衡データでは負例が非常に多いため、少数の誤検知でもFPRは小さく見え、ROCが楽観的になることがあります。正例を拾った結果、予測した正例の大半が誤りではないかを見たい場面では、PR曲線やその要約も併記します。

## 想起チェック

<details class="recall">
<summary>ROC曲線の横軸と縦軸は何か</summary>

横軸がFPR、縦軸がTPRです。閾値を動かしたときの点の集合として読みます。

</details>

<details class="recall">
<summary>AUCを順位付けとしてどう解釈するか</summary>

無作為に選んだ正例のスコアが、無作為に選んだ負例のスコアより高い確率です。同点は半分として扱います。

</details>

<details class="recall">
<summary>AUCが高ければ運用閾値も自動的に良いか</summary>

いいえ。AUCは全閾値の要約なので、費用や許容FPRに応じて運用点を別に選びます。

</details>
