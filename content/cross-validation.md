---
exam: E資格
concept: 交差検証
slug: cross-validation
tier: B
area: 機械学習
summary: データを複数の組合せで訓練・評価し、未知データでの性能を一つの分割に依存せず見積もる手続き。
updated: 2026-08-22
sources:
  - title: "Cross-validation: evaluating estimator performance"
    url: https://scikit-learn.org/stable/modules/cross_validation.html
  - title: "cross_validate"
    url: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
---

## ひとことで言うと

交差検証は、利用可能なデータを $k$ 個の fold に分け、各 fold を一度ずつ評価側に回して、未知データでの性能を推定する手続きです。推定しているのは、学習済みモデルを新しいデータへ適用したときの平均的な性能であり、最終モデルを作る更新手法ではありません。

<div class="analogy">

一つの模擬試験だけで実力を決めず、同じ問題集を受験者の組だけ入れ替えて何度も採点する方法です。採点者が変わるのではなく、学習に使う問題と評価に残す問題を交替させます。

</div>

## なぜ必要か

ホールドアウトは訓練用と評価用を一度だけ分けるため、たまたま難しい例・易しい例がどちら側に入ったかで評価値が変わります。k-fold では、各回で $k-1$ fold を訓練、残り1 foldを評価に使い、得られた $k$ 個のスコアを平均します。一つの分割だけに依存しないため、性能推定の分散を下げやすいのが利点です。

その代わり、モデルを $k$ 回学習するので、ホールドアウトの1回学習に比べて計算コストは概ね $k$ 倍です。データが少ないときに評価用へ固定的に取り分けるデータを減らせる一方、計算時間との交換になります。

<div class="caution">

交差検証の平均は、最終モデルを未知のテストセットで一度評価した値そのものではありません。分割方法やデータの独立性が、推定の前提になります。

</div>

## 仕組み

サンプル集合を $D$、fold を $F_1,\ldots,F_k$、学習アルゴリズムを $A$、性能指標を $s$ とします。第 $i$ 回では $D\setminus F_i$ でモデル $\hat f_i=A(D\setminus F_i)$ を学習し、$F_i$ 上のスコアを計算します。

$$
\mathrm{CV}_k=\frac{1}{k}\sum_{i=1}^{k}s(\hat f_i,F_i)
$$

ここで $k$ は分割数、$s(\hat f_i,F_i)$ は第 $i$ fold を評価データにしたスコアです。各サンプルは評価側に一度だけ現れます。したがって、平均値は「このデータから学習した手続きが、同じ生成条件の未見データで示す性能」の推定値として読みます。fold 間のスコアのばらつきも併記すると、分割による不確かさを確認できます。

分類では、少数クラスがある fold から欠けると、ROC AUC などの指標を計算できなかったり、学習自体が失敗したりします。この場合は StratifiedKFold で各 fold のクラス比をおおむね揃えます。ただし層化は fold を均質にするため、観測されたばらつきを小さく見せることがあります。時系列では通常の KFold を使いません。近い時点の観測には自己相関があり、訓練と評価に未来・過去を混ぜると、実運用より有利な相関を評価へ持ち込むからです。時間順に過去で学習し未来で評価する分割を使います。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| k-fold の手順 | $k-1$ foldで学習し、残りを評価する操作を $k$ 回行う | 1回だけ分割して終える |
| ホールドアウトとの差 | 複数スコアの平均で分割依存を抑えやすい | 訓練コストも1回のままと思う |
| 不均衡分類の分割 | 各 fold のクラス比を保つ StratifiedKFold | 少数クラスが評価 fold から消える |
| 時系列の評価 | 過去で学習し未来で評価する | ランダムシャッフルで未来を訓練へ混ぜる |

## 実装で確かめる

NumPy だけで、各 fold を評価側に一度ずつ回す形を確認します。評価値の平均だけでなく、fold 間の標準偏差も残します。

```python
import numpy as np

rng = np.random.default_rng(4)
x = rng.normal(size=30)
y = 2 * x + rng.normal(scale=0.4, size=30)
folds = np.array_split(rng.permutation(len(x)), 5)
scores = []
for i, test in enumerate(folds):
    train = np.concatenate([f for j, f in enumerate(folds) if j != i])
    coef = np.polyfit(x[train], y[train], 1)
    pred = np.polyval(coef, x[test])
    scores.append(np.mean((pred - y[test]) ** 2))
print(np.round(scores, 4))
print(round(float(np.mean(scores)), 4), round(float(np.std(scores)), 4))
```

出力は fold ごとの評価値、続いて平均と標準偏差です。ライブラリの `cross_validate` でも、複数指標のテストスコアに加えて `fit_time` と `score_time` を取得できます。学習済みモデルを最後に全データで作り直す処理と、交差検証中の評価を混同しないでください。

## 取り違えやすいもの

| 手法 | 交差検証との切り分け |
|---|---|
| ホールドアウト | 分割と評価が1回。安価だが分割の偶然に左右されやすい |
| StratifiedKFold | k-fold の分類向け変形。クラス比を fold ごとに保つ |
| TimeSeriesSplit | 時系列向け。時間順を守り、将来の観測を訓練へ入れない |
| テストセット | 最終的な性能確認用。交差検証で何度も選択に使う評価用データとは役割が違う |

## 想起チェック

<details class="recall">
<summary>k-fold で1つの fold は何回評価側になるか</summary>

1回です。$k$ 回の学習・評価を行い、各サンプルが一度ずつ評価側に回ります。

</details>

<details class="recall">
<summary>不均衡な分類で層化を使う理由は何か</summary>

各 fold のクラス比をおおむね保ち、少数クラスが欠けて指標計算や学習が失敗する事態を避けるためです。

</details>

<details class="recall">
<summary>時系列に通常の KFold を使わない理由は何か</summary>

時間的に近い観測を訓練と評価へ混ぜると、自己相関によって実運用より楽観的な評価になり得るためです。過去から未来へ進む分割を使います。

</details>
