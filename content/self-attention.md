---
exam: E資格
concept: Self-Attention
slug: self-attention
tier: A
area: 深層学習
summary: 同じ系列の入力をQuery・Key・Valueへ射影し、位置間の類似度でValueを重み付けして各位置の表現を更新する注意機構です。
updated: 2026-08-22
sources:
  - title: "Attention Is All You Need"
    url: https://arxiv.org/abs/1706.03762
  - title: "(Beta) Implementing High-Performance Transformers with Scaled Dot Product Attention (SDPA)"
    url: https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
---

## ひとことで言うと

Self-Attentionは、系列内の各位置が他の位置を参照して、自分の表現を作り直す計算です。同じ入力行列から3種類の線形射影 $Q,K,V$ を作り、QueryとKeyの相性で重みを決め、その重みでValueを加重平均します。入力元が同じ系列なので「self」です。

<div class="analogy">

会議で各人が「自分はいま何を知りたいか」をQuery、「自分はどんな情報を持つか」をKey、「実際に渡せる内容」をValueとして掲示し、QueryとKeyが近い人の発言を強く取り込む仕組みです。探す基準と渡す内容を分けるので、同じ入力から三つの役割を作る意味があります。

</div>

## なぜ必要か

畳み込みのように近傍を固定せず、再帰のように前の時刻の計算を待たずに、系列内の任意の位置を直接参照できます。たとえば長い距離にある語の関係を、何層も経由せず一度の注意計算で結べます。ここで「入力をそのまま比較する」と考えると実装を誤ります。参照したい条件、照合される条件、返す特徴は用途が違うため、三つの射影を別パラメータで持たせます。

ただし、全位置対全位置を比較する代償があります。系列長を $n$、各ヘッドの特徴次元を $d_k$ とすると、スコア行列 $QK^{\mathsf{T}}$ は $n\times n$ で、計算量もメモリも系列長に対して二乗で増えます。長文で重くなる原因は、射影そのものよりこの全ペアの行列を作る部分です。

| 系列長の変化 | 全位置対全位置の組数 | 実装上の影響 |
|---|---:|---|
| $n$ | $n^2$ | 基準 |
| $2n$ | $4n^2$ | スコア計算と注意重みの保持が約4倍 |


## 仕組み

入力を $X\in\mathbb{R}^{n\times d_{\mathrm{model}}}$、Query・Key・Valueの射影行列を $W^Q,W^K,W^V$ とします。まず同じ $X$ から次を計算します。

$$
Q=XW^Q,\qquad K=XW^K,\qquad V=XW^V
$$

$n$ は系列長、$d_{\mathrm{model}}$ は入力特徴次元、$Q,K,V$ はそれぞれ検索条件・照合対象・集約する内容です。$Q,K$ の最終次元を $d_k$、$V$ の最終次元を $d_v$ とすると、計算順は「内積でスコア → スケーリング → マスク → softmax → Valueの加重和」です。

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^{\mathsf{T}}}{\sqrt{d_k}}+M\right)V
$$

$M$ は許可しない対応を表すマスクで、許可位置は0、禁止位置は実質的に $-\infty$ とします。softmaxの前に加えるため、禁止位置の確率が0になります。自己回帰で未来を見せない因果マスク、パディングを無視するマスクはこの場所に入ります。マスクをValueに掛けるだけでは、禁止位置が重み計算に参加してしまうので不十分です。

$\sqrt{d_k}$ で割る理由は単なる正規化ではありません。$q$ と $k$ の各成分が平均0、分散1で独立なら、内積 $q\cdot k=\sum_{i=1}^{d_k}q_i k_i$ の分散は $d_k$ に比例します。次元が大きいほどsoftmaxに入る値の絶対値が大きくなり、最大要素だけがほぼ1になる飽和が起き、勾配が小さくなります。そこで内積を $\sqrt{d_k}$ で割り、スコアの尺度を戻します。分母はsoftmaxの後ではなく前です。

Multi-Head Attentionでは、$Q,K,V$ を一組だけ使うのではなく、ヘッドごとに別の射影を行い、各ヘッドで注意を計算します。ヘッド数を $h$ とし、通常は $d_k=d_v=d_{\mathrm{model}}/h$ として、各ヘッドの出力を連結してから出力射影を適用します。

$$
\mathrm{head}_i=\mathrm{Attention}(XW_i^Q,XW_i^K,XW_i^V),\qquad
\mathrm{MultiHead}=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W^O
$$

ヘッドが分けるのは入力系列そのものではなく、特徴空間と相性の見方です。あるヘッドは近い位置、別のヘッドは遠い依存関係というように、異なる射影空間で同時に参照できます。ヘッド数を増やすだけで情報量が増えるわけではなく、総特徴次元との配分が実装上のポイントです。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| Q/K/Vの役割 | 同じ入力から別々に線形射影し、Q-Kで重み、Vで集約 | Q・K・Vを同じテンソルの別名とする |
| 計算順 | $QK^{\mathsf{T}}$ → $1/\sqrt{d_k}$ → マスク → softmax → $V$ | softmax後にスケールする／$V$同士を比較する |
| スケーリングの理由 | 内積の分散が $d_k$ に比例し、softmax飽和を招くため | 出力ベクトルの長さをそろえるためだけとする |
| Multi-Headの意味 | 射影空間を複数に分け、異なる関係を並列に学習 | 系列を位置ごとに分割する機構とする |
| マスクの位置 | softmax前のスコアに加え、禁止位置を確率0にする | Valueを0にするだけで未来参照を防げるとする |
| 計算量 | $QK^{\mathsf{T}}$ の $n\times n$ により計算量・メモリが $n^2$ で効く | ヘッド数だけが長文時の主因とする |

## 実装で確かめる

NumPyで一つのヘッドを実装します。`scores` の形が系列長の二乗になること、causal maskで未来側の重みが0になることを確認できます。

```python
import numpy as np

rng = np.random.default_rng(0)
n, d_model, d_k, d_v = 4, 6, 3, 2
X = rng.normal(size=(n, d_model))
Wq, Wk, Wv = (rng.normal(size=(d_model, d)) for d in (d_k, d_k, d_v))
Q, K, V = X @ Wq, X @ Wk, X @ Wv
scores = Q @ K.T / np.sqrt(d_k)
mask = np.triu(np.ones((n, n), dtype=bool), k=1)
scores = np.where(mask, -np.inf, scores)
weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
weights /= weights.sum(axis=1, keepdims=True)
out = weights @ V
print(scores.shape, weights.shape, out.shape)
print(np.round(weights[0], 6))
```

出力は `(4, 4) (4, 4) (4, 2)` となり、1番目の位置の重みは `[1. 0. 0. 0.]` です。softmax前に上三角をマスクしたため、先の位置へ重みが流れません。`np.max` を引くのはsoftmaxの値を変えずに指数のオーバーフローを避けるためです。

<div class="caution">

実装では $K$ の転置を忘れるとスコアの軸が崩れ、マスクの向きを間違えると「過去を見られない」モデルになります。バッチやヘッドを追加した場合も、最後の二軸がQuery位置とKey位置になっているかを形状で確認します。

</div>

## 取り違えやすいもの

| 用語 | Self-Attentionとの切り分け |
|---|---|
| Encoder-Decoder Attention | Qは一方の系列、K/Vは別の系列から作る。Self-Attentionは同じ系列から作る |
| Additive Attention | QとKの相性を小さなネットワークで計算する方式。Scaled Dot-Productは内積を使う |
| Multi-Head Attention | Self-Attentionそのものではなく、複数の射影空間で注意を並列化する拡張 |
| Causal Attention | 自己注意に未来禁止マスクを加えた制約付きの形。全Self-Attentionが因果的とは限らない |
| Cross-Attention | 参照元が別系列。Q/K/Vの入力元が同じかどうかで判別する |

## 想起チェック

<details class="recall">
<summary>Q、K、Vはそれぞれ何を決める役割か</summary>

Qは探したい条件、Kは照合される条件、Vは重み付けして集約する内容です。三つとも同じ入力から別の射影で作れます。

</details>

<details class="recall">
<summary>Scaled Dot-Product Attentionの計算順を答える</summary>

まず $QK^{\mathsf{T}}$、次に $\sqrt{d_k}$ で割り、必要ならマスクを加え、softmaxの重みを $V$ に掛けます。

</details>

<details class="recall">
<summary>スケーリング係数で割らないと何が起きるか</summary>

内積の分散が $d_k$ に比例して大きくなり、softmaxが飽和して勾配が小さくなります。

</details>

<details class="recall">
<summary>系列長を2倍にしたとき、Self-Attentionの何が問題になるか</summary>

全位置対全位置のスコア行列が必要なので、スコア計算とその行列を保持するメモリが系列長の二乗で増えます。

</details>
