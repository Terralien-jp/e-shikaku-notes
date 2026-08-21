---
exam: E資格
concept: 多クラス分類損失
slug: multiclass-cross-entropy
tier: B
area: 深層学習
summary: softmax 出力と組み合わせる多クラス分類の損失を、logitsからの安定した計算、勾配、ラベル形式まで実装目線で整理する。
updated: 2026-08-22
sources:
  - title: "CrossEntropyLoss"
    url: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
  - title: "Deep Feedforward Networks"
    url: https://www.deeplearningbook.org/contents/mlp.html
---

## ひとことで言うと

多クラス交差エントロピーは、正解クラスに割り当てた確率が高いほど小さくなる損失です。分類器の最後では、クラスごとの未正規化スコア（logits）を softmax とこの損失へつなぎます。

<div class="analogy">

複数の候補に持ち点を配り、正解の箱に何点残したかを採点する仕組みです。正解箱が空に近いほど、失点は急激に大きくなります。

</div>

## なぜ必要か

多クラス分類では、出力を単なる数値として近づけるより、各クラスの確率分布として正解とのずれを測る方が目的に合います。softmax は logits を確率に変換し、交差エントロピーはその分布のうち正解に対応する成分を評価します。この組み合わせでは、確率化と採点を別々に扱うより勾配が簡潔になります。

| 段階 | 役割 |
|---|---|
| softmax | logitsをクラス確率へ変換 |
| 交差エントロピー | 正解クラスの確率を採点 |

## 仕組み

サンプルの logits を $\\mathbf{z}$、クラス $k$ の予測確率を $p_k$、正解を表す one-hot ベクトルを $\\mathbf{y}$ とします。クラス数を $K$ とすると、損失は次の形です。

$$
L=-\sum_{k=1}^{K} y_k\log p_k=-\log p_t
$$

$t$ は正解クラスの番号です。one-hot の $y_k$ は正解クラスだけ1なので、実際に残るのは正解クラスの対数確率です。logitsから直接計算する場合は、softmax と対数を個別に計算せず、log-sum-exp の形にまとめます。

$$
L=-z_t+\log\left(\sum_{k=1}^{K}e^{z_k}\right)
$$

softmax は全 logits に同じ定数 $c$ を足しても変わりません。そこで $m=\max_k z_k$ として $z_k-m$ を指数に入れると、確率は変えずに指数のオーバーフローを避けられます。

この組み合わせの要点は、logitsに対する勾配が予測確率と正解 one-hot の差になることです。

$$
\frac{\partial L}{\partial z_k}=p_k-y_k
$$

したがって正解クラスでは予測確率を1へ押し上げ、他クラスでは確率を0へ押し下げる向きになります。実装では「softmaxの出力を作ってから交差エントロピー」ではなく、logitsを損失関数へ渡すAPIを選ぶと、この安定化と勾配計算をまとめて扱えます。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| 損失の式 | 正解クラスの確率の負の対数 | 全クラスの確率を同じ重みで足す |
| logitsへの勾配 | $p_k-y_k$ | $y_k-p_k$ と符号を逆にする |
| 数値安定化 | 最大値を引いて指数計算する | 最大値を足す／確率を変えると考える |
| ラベル形式 | 整数 index と one-hot・確率分布でAPIが異なる | one-hotを常に整数ラベルとして渡す |

## 実装で確かめる

NumPyで、maxを引いたsoftmaxと、logitsからの損失・勾配を同じ入力で確認します。

```python
import numpy as np

z = np.array([1.2, -0.3, 0.7])
t = 0
m = z.max()
p = np.exp(z - m) / np.exp(z - m).sum()
loss = -np.log(p[t])
y = np.eye(len(z))[t]
grad = p - y
print(np.round(p.sum(), 6), round(loss, 6), np.round(grad, 6))
```

出力の確率和は `1.0` になり、正解クラスの勾配だけは $p_t-1$、それ以外は各 $p_k$ です。実際のPyTorchでは、整数ラベルなら logits とクラス index を渡し、クラス確率を教師にする場合は確率形式を渡します。softmax済みの値を「logits」として二重に正規化しないことが切り分けの要点です。

## 取り違えやすいもの

| 用語 | 切り分け |
|---|---|
| softmax | logitsをクラス確率へ変換する処理。損失そのものではありません |
| 多クラス交差エントロピー | 正解分布と予測分布を比較する損失。softmaxと組み合わせて使います |
| 平均二乗誤差 | 出力の差の二乗を測る損失。確率分類の勾配の形は同じになりません |
| CrossEntropyLoss | PyTorchのAPI。logits入力から交差エントロピーを計算し、整数 indexまたは確率形式のtargetを受けます |

## 想起チェック

<details class="recall">
<summary>softmaxと交差エントロピーを組み合わせたlogitsの勾配は何か</summary>

予測確率 $p_k$ と正解 one-hot $y_k$ の差、$p_k-y_k$ です。

</details>

<details class="recall">
<summary>logitsから指数を計算する前に最大値を引く理由は何か</summary>

全成分への平行移動ではsoftmaxの値が変わらないため、最大値を引いて指数のオーバーフローを抑えます。

</details>

<details class="recall">
<summary>PyTorchで整数ラベルとone-hotに違いはあるか</summary>

あります。クラス indexを渡す形式と、クラス確率を渡す形式を使い分けます。ラベルの配列形式を確認してからAPIを選びます。

</details>
