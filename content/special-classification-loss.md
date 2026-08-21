---
exam: E資格
concept: 特殊な分類損失
slug: special-classification-loss
tier: C
area: 深層学習
summary: 標準の交差エントロピーでは学習が偏る場面で、例の難しさやラベルの確信度を調整する損失を使い分けます。
updated: 2026-08-22
sources:
  - title: "Focal Loss for Dense Object Detection"
    url: https://arxiv.org/abs/1708.02002
  - title: "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss"
    url: https://arxiv.org/abs/1906.07413
  - title: "When Does Label Smoothing Help?"
    url: https://arxiv.org/abs/1906.02629
---

## ひとことで言うと

特殊な分類損失は、標準の交差エントロピーでは学習が偏る場面で、例の難しさやラベルの確信度を補正する選択肢です。

<div class="analogy">

易しい小問を大量に採点すると難問の結果が埋もれます。易しい例を軽くし、難例や少数クラスへ学習の目を向けます。

</div>

## なぜ必要か

極端な不均衡では、容易な負例が多数あるため損失がそれらに埋もれます。Focal Lossは容易な例の損失を下げます。

少数クラスの汎化にはクラス分布に応じたマージンを入れるLDAM、過信の抑制にはone-hot教師を一様分布との重みに置き換えるLabel Smoothingがあります。

<div class="caution">

損失を変えれば不均衡が自動的に解決するわけではありません。少数クラスを重視するのか、容易な例を軽くするのかを先に切り分けます。

</div>

## 仕組み

正解クラスの予測確率を $p_t$、集中度を決める $\gamma$ とすると、Focal Lossは次です。

$$
\operatorname{FL}(p_t)=-(1-p_t)^\gamma\log(p_t)
$$

$p_t$ は正解クラスの確率、$\gamma$ は非負の値です。$p_t$ が大きい例ほど係数が小さくなります。

Label Smoothingでは、クラス数を $K$、平滑化率を $\varepsilon$、one-hotラベルを $y_k$ として、教師値を次のように変えます。

$$
\tilde{y}_k=(1-\varepsilon)y_k+\frac{\varepsilon}{K}
$$

正解を確率1と固定しないため、過信を抑え、予測の較正を改善する方向に働きます。LDAMはラベル分布を見たクラス依存のマージンを組み込みます。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| Focal Lossの係数 | 易しい例ほど損失の重みが下がる | 難しい例を除外する、と断定する |
| Label Smoothingの効果 | 過信を抑え、較正を改善する | 少数クラスだけを再重み付けする |
| 損失の選択 | 不均衡・難例集中・過信抑制で目的を分ける | どの不均衡にも同じ損失を使う |

## 実装で確かめる

確率が高い例ほどFocal Lossの係数が小さくなることだけをNumPyで確認します。

```python
import numpy as np
p_t = np.array([0.99, 0.6, 0.1])
gamma = 2.0
focal = -(1 - p_t) ** gamma * np.log(p_t)
print(np.round(focal, 6))
```

## 取り違えやすいもの

| 手法 | 主に調整するもの | 使う判断 |
|---|---|---|
| Focal Loss | 例ごとの損失寄与 | 易しい負例が多すぎる |
| LDAM | クラス分布に応じたマージン | 少数クラスの汎化を重視する |
| Label Smoothing | 教師ラベルの確信度 | 予測の過信や較正を扱う |
| クラス再重み付け | クラスごとの損失係数 | 不均衡を重みで補正する |

## 想起チェック

<details class="recall">
<summary>Focal Lossは何の寄与を下げるか</summary>

正しく分類できた容易な例、特に大量の容易な負例の損失への寄与です。難しい例を学習対象から削除する手法ではありません。

</details>
