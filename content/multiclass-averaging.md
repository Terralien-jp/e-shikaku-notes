---
exam: E資格
concept: 多クラス平均法
slug: multiclass-averaging
tier: C
area: 機械学習
summary: 多クラス分類のクラス別指標を、クラス単位・事例単位・support単位のどれでまとめるかを使い分ける方法。
updated: 2026-08-22
sources:
  - title: "classification_report"
    url: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
  - title: "precision_recall_fscore_support"
    url: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
---

## ひとことで言うと

多クラス平均法は、クラスごとに計算した指標を1つの値へ集約する規則です。平均の名前は計算の順序を表します。macroはクラスを、microは事例を同等に扱い、weightedは各クラスのsupport（正解ラベルの件数）を重みにします。

<div class="analogy">

クラス別の成績をまとめるとき、クラスごとの点数を同じ重みで平均するのがmacro、全受験者の答案を一つの束として集計するのがmicro、受験者数の多いクラスの点数を重くするのがweightedです。

</div>

## なぜ必要か

クラス別の値を並べるだけでは、モデル全体を比較しにくい一方、1つの平均だけでは少数クラスの失敗を見落とします。そこで、何を1単位として評価するかを先に決めます。少数クラスも同じ重要度ならmacro、全事例を一件ずつ同じ重要度で扱うならmicro、実データのクラス比率を反映したいならweightedです。多数クラスを正しく処理していても、少数クラスをほぼ見逃していれば、macroは大きく下がります。

<div class="caution">

平均の種類を変えるとモデルの順位まで変わり得ます。数値だけを比較せず、少数クラスを見落としたくないのか、全事例の成績を重視するのかを評価目的に結び付けます。

</div>

## 仕組み

クラス $k$ の指標を $m_k$、クラス数を $K$、そのsupportを $n_k$ とします。macroは次の単純平均です。

$$
M_{\mathrm{macro}}=\frac{1}{K}\sum_{k=1}^{K}m_k
$$

weightedはsupportで重み付けします。$N=\sum_{k=1}^{K}n_k$ は全事例数です。

$$
M_{\mathrm{weighted}}=\frac{\sum_{k=1}^{K}n_km_k}{N}
$$

microはクラスごとの指標を平均せず、全クラスの判定結果を合算してから指標を計算します。そのため、多クラス単一ラベル分類では事例数の多いクラスの影響を受け、accuracyと同じ値になる指標もあります。`average` の指定を変えるだけで、同じ予測から異なる問いに答えられます。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| macroとmicroの比較 | macroはクラス同等、microは事例同等 | 「どちらもクラスごとの単純平均」 |
| 不均衡データの平均 | 少数クラスを重視するならmacro | weightedなら少数クラスも同じ影響になる |
| weightedの重み | support、つまり正解ラベル件数 | 予測件数や指標の値を重みにする |

## 実装で確かめる

クラス別の値とsupportから、macroとweightedの違いだけをNumPyで確認します。

```python
import numpy as np

score = np.array([0.90, 0.60, 0.20])
support = np.array([80, 15, 5])
macro = score.mean()
weighted = np.average(score, weights=support)
print(round(macro, 3), round(weighted, 3))
```

出力は `0.567 0.825` です。少数クラスの値が低くても、weightedは多数クラスの値に近づきます。

## 取り違えやすいもの

| 方法 | 集計単位 | 選ぶ場面 |
|---|---|---|
| macro | クラス | クラスごとの公平さを見たい |
| micro | 全事例の合算 | 事例全体の成績を見たい |
| weighted | support付きのクラス | 実データの比率を反映したい |

## 想起チェック

<details class="recall">
<summary>少数クラスの性能を同等に効かせたいとき、どの平均を選ぶか</summary>

macroです。各クラスの値を同じ重みで平均します。

</details>

<details class="recall">
<summary>weightedの重みは何か</summary>

各クラスのsupport、つまり正解ラベル側にある事例数です。

</details>
