---
exam: E資格
concept: パラメータ初期化
slug: parameter-initialization
tier: B
area: 深層学習
summary: 重みを適当に置くのではなく、層をまたいで活性値と勾配のスケールを保つための設計です。ゼロ初期化の対称性、活性化関数ごとの Xavier / He の使い分け、正規化層との関係を整理します。
updated: 2026-08-22
sources:
  - title: "Understanding the difficulty of training deep feedforward neural networks"
    url: https://proceedings.mlr.press/v9/glorot10a.html
  - title: "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
    url: https://arxiv.org/abs/1502.01852
---

## ひとことで言うと

パラメータ初期化は、学習開始時の重みとバイアスを決める操作です。目的は初回の損失を小さくすることではなく、順伝播の活性値と逆伝播の勾配が、深い層を通っても極端に縮小・増大しない状態を作ることです。

<div class="analogy">

長い配管に水を流す前に、各区間の太さを調整して水圧を保つようなものです。入口で水量だけ合わせても、区間ごとに減衰や増幅が重なれば末端は流れません。初期化は層ごとの計算を始める前の「流量設計」です。

</div>

## なぜ必要か

全重みを0にすると、同じ層のニューロンは同じ出力を計算し、同じ勾配を受け取ります。更新後も各ニューロンの重みは同一のままなので、ユニット数を増やしても異なる特徴を分担できません。これが対称性が破れないという意味です。バイアスだけを乱数にしても、重みの役割分担を十分には作れません。

一方、乱数なら何でもよいわけではありません。重みの分散が大きすぎると活性値や勾配が層を進むほど膨らみ、小さすぎると0へ寄ります。GlorotとBengioは、活性化関数の性質と層ごとの活性値・勾配の変化が深いネットワークの最適化を難しくすると分析し、初期化を設計対象として扱いました。

$$
W_{i,:}^{(l)} = W_{j,:}^{(l)} \quad\Longrightarrow\quad \text{ユニット } i,j \text{ は同じ更新を受ける}
$$

## 仕組み

入力を $\mathbf{x}$、第 $l$ 層の重みを $W^{(l)}$、バイアスを $\mathbf{b}^{(l)}$、活性化前の値を $\mathbf{z}^{(l)}$ とすると、計算は次です。

$$
\mathbf{z}^{(l)} = W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, \qquad \mathbf{a}^{(l)} = f(\mathbf{z}^{(l)})
$$

$\mathbf{a}^{(l-1)}$ は前層の出力、$f$ は活性化関数です。初期化では、各層の入力数と出力数、そして $f$ の形を見て重みの分散を決めます。狙いは、層をまたいだ積のスケールをおおむね保つことです。

| 活性化関数の系統 | まず選ぶ初期化 | 判断の軸 |
|---|---|---|
| tanh / sigmoid | Xavier（Glorot）系 | 入出力の両側のスケールを見て分散を決める |
| ReLU / PReLU | He 系 | 負側を0にする整流の影響を分散に反映する |

これは公式名を暗記する表ではなく、非線形性に合わせて分散の設計を変える表です。XavierとHeの式の導出を再現するより、「どの活性化を通る重みか」を先に確認するほうが実装では有効です。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| ゼロ初期化の問題点 | 対称性が残り、同じ層のユニットが同じ更新になる | 「勾配が必ず全て0になる」と限定する |
| 初期化の目的 | 活性値と勾配のスケールを層間で保つ | 初期損失を最小にする |
| 初期化の選択 | ReLU系はHe、tanh/sigmoid系はXavierを起点にする | 活性化関数を見ずに一律選択する |
| 正規化層との併用 | 正規化がスケールを調整するため、初期化だけで挙動を説明しない | 初期化が不要になると断定する |

## 実装で確かめる

ゼロ初期化では、同じ入力を受けるユニットの出力が一致します。乱数初期化では、同じ層でもユニットごとに値が分かれます。分散の違いは、深いネットワークでこの差を増幅または減衰させます。

```python
import numpy as np
rng = np.random.default_rng(0)
x = rng.normal(size=(4, 3))
W_zero = np.zeros((3, 2))
W_rand = rng.normal(0, 1 / np.sqrt(3), size=(3, 2))
print("zero outputs equal:", np.allclose(x @ W_zero[:, 0], x @ W_zero[:, 1]))
print("random outputs equal:", np.allclose(x @ W_rand[:, 0], x @ W_rand[:, 1]))
```

## 取り違えやすいもの

| 用語 | 切り分け |
|---|---|
| [Xavier / Glorot](/learn/e-shikaku/xavier-and-he-init/) | tanhやsigmoidなど、入出力のスケールを両側から見る初期化の系統 |
| [He](/learn/e-shikaku/xavier-and-he-init/) | ReLU系の整流で失われる側を考慮する初期化の系統 |
| バイアス初期化 | 重みの対称性を壊す役割とは別に、オフセットを決める設定 |
| [正規化層](/learn/e-shikaku/normalization-layers/) | 学習中に活性値のスケールを調整する層。初期化の目的と重なる部分があるが同じ操作ではない |

## 想起チェック

<details class="recall">
<summary>全重みを0にすると、なぜユニット数を増やしても学習上の役割が分かれないか</summary>

同じ入力・同じ重みから同じ出力と勾配が生じ、更新後も重みの同一性が保たれるためです。対称性が破れません。

</details>

<details class="recall">
<summary>ReLU系とtanh・sigmoid系で初期化を変える理由は何か</summary>

活性化関数が値を通す仕方が異なり、層を通る活性値と勾配の分散の保ち方も変わるからです。前者はHe、後者はXavierを起点にします。

</details>

<details class="recall">
<summary>正規化層があるモデルで、初期化の設定だけを見てよいか</summary>

よくありません。正規化層も活性値のスケールを調整するため、初期化単独の議論をそのまま当てはめず、層の順序と正規化の有無を一緒に確認します。

</details>
