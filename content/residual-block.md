---
exam: E資格
concept: Residual Block
slug: residual-block
tier: A
area: 深層学習
summary: 入力を恒等ショートカットで足し戻し、主経路には入力からの残差だけを学習させるブロック。勾配の恒等経路、次元合わせの射影、ボトルネックの役割を式と実装で確認する。
updated: 2026-08-22
sources:
  - title: "Deep Residual Learning for Image Recognition"
    url: https://arxiv.org/abs/1512.03385
  - title: "Identity Mappings in Deep Residual Networks"
    url: https://arxiv.org/abs/1603.05027
---

## ひとことで言うと

Residual Block（残差ブロック）は、入力をそのまま通すショートカットと、入力に対する変化を計算する主経路を足し合わせる部品です。入力を $\mathbf{x}$、主経路が表す残差関数を $\mathcal{F}(\mathbf{x}, \theta)$ とすると、出力は $\mathbf{y}=\mathcal{F}(\mathbf{x},\theta)+\mathbf{x}$ です。ここで $\theta$ は主経路の学習パラメータです。ブロック全体を一つの写像として見ると、毎回新しい表現をゼロから作るのではなく、入力からの差分を作って足す構造になっています。

<div class="analogy">

既存の設計図に、毎回すべてを書き直すのではなく、変更箇所だけを差分として重ねるイメージです。変更が不要なら主経路の出力をゼロに近づければよく、元の設計図はショートカットからそのまま残ります。

</div>

## なぜ必要か

通常の積層では、入力 $\mathbf{x}$ から出力 $\mathcal{H}(\mathbf{x})$ までを主経路だけで表現します。これに対して残差ブロックは $\mathcal{H}(\mathbf{x})=\mathcal{F}(\mathbf{x})+\mathbf{x}$ と分解し、学習対象を $\mathcal{H}$ そのものから残差 $\mathcal{F}=\mathcal{H}-\mathbf{x}$ に移します。入力をほぼ保つ場面では、主経路が恒等写像を再現する代わりに、残差を小さくすれば済みます。ショートカットの出力と主経路の出力は、加算点で要素ごとに足し合わせます。

| 観点 | 主経路 | ショートカット |
|---|---|---|
| 計算するもの | 残差 $\mathcal{F}(\mathbf{x},\theta)$ | 恒等写像 $\mathbf{x}$、または射影 |
| 加算前の役割 | 入力に加える変化を作る | 入力の情報を加算点まで運ぶ |
| 次元変更 | 畳み込み側で設計 | 必要なら $1\times1$ 射影で合わせる |

この形の利点は、ブロックを単独で見ても順方向と逆方向に二つの経路があることです。主経路の演算が複雑でも、ショートカットが恒等写像なら入力情報と勾配の通り道が残ります。ここでいう利点は、どんなネットワークでも自動的に精度が上がるという意味ではありません。加算できるテンソル形状、加算位置、活性化の置き方まで含めて、ブロックの定義として確認する必要があります。

## 仕組み

基本形を、主経路 $\mathcal{F}$ とショートカット写像 $\mathcal{S}$ で一般化します。$\mathbf{x}$ は入力特徴マップ、$\mathcal{F}(\mathbf{x},\theta)$ は畳み込みや正規化などを含む主経路、$\mathcal{S}(\mathbf{x})$ はショートカット、$\mathbf{y}$ は加算後の出力です。

$$
\mathbf{y}=\mathcal{F}(\mathbf{x},\theta)+\mathcal{S}(\mathbf{x})
$$

同じチャネル数・空間サイズなら恒等写像 $\mathcal{S}(\mathbf{x})=\mathbf{x}$ を使えます。このとき損失を $L$、加算点から上流へ伝わる勾配を $\partial L/\partial\mathbf{y}$ とすると、加算の微分は次の形になります。

$$
\frac{\partial L}{\partial\mathbf{x}}
=\frac{\partial L}{\partial\mathbf{y}}
\left(\frac{\partial\mathcal{F}}{\partial\mathbf{x}}+\mathbf{I}\right)
$$

$\mathbf{I}$ は恒等写像のヤコビアンです。右辺の $\partial L/\partial\mathbf{y}$ は、主経路の微分を通る項に加えて、係数1の恒等経路にもそのまま現れます。したがって主経路の勾配が小さくても、ショートカット側から同じ勾配が入力へ届きます。ブロックを積み重ねた場合も、各ブロックでこの恒等項が連鎖律に残る、という読み方が試験の要点です。

出力の次元が変わるブロックでは、$\mathbf{x}$ をそのまま足せません。ショートカットに学習可能な射影 $W_s$ を置き、$\mathcal{S}(\mathbf{x})=W_s\mathbf{x}$ として形状を合わせます。畳み込み層では通常、$1\times1$ 畳み込みでチャネル数を変え、必要ならストライドで空間サイズも変えます。主経路とショートカットを別々に設計してから、加算時点のバッチ・チャネル・高さ・幅が一致しているかを確認します。

深い畳み込みブロックではボトルネック構成も使われます。主経路を $1\times1$ 畳み込みでチャネル方向に絞り、$3\times3$ 畳み込みを狭い表現に対して適用し、最後の $1\times1$ 畳み込みで元の幅へ戻します。中央の計算量の大きい $3\times3$ 演算を低チャネル数で実行できるため、計算とパラメータを抑えやすい、という狙いです。最後に戻した主経路と、元の幅を保つ恒等ショートカットを加算します。ボトルネックはショートカットを細くする仕組みではなく、主経路の内部を細くする仕組みです。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| ブロックの出力式 | 主経路の残差とショートカットを加算する | 主経路だけを出力とする |
| 逆伝播の経路 | 恒等ショートカットでは勾配に恒等項が加わる | 勾配も主経路だけを通る |
| 次元が異なる場合 | 射影ショートカットで加算前に形状を合わせる | 異なるチャネル数をそのまま加算する |
| ボトルネックの順序 | $1\times1$ で縮小、$3\times3$、$1\times1$ で復元 | $3\times3$ を高チャネルのまま3回行う |
| 恒等写像の意味 | 学習パラメータを持たず入力をそのまま渡す | 恒等写像を「重みが全て学習済み」と解釈する |

## 実装で確かめる

NumPyで、主経路の係数を小さくした一つのスカラー・ブロックを考えます。$\mathcal{F}(x,\theta)=\theta x$、損失を $L=y^2/2$ とすると、解析的には $\partial L/\partial x=(\theta+1)y$ です。加算の後ろから来た勾配が、主経路の係数 $\theta$ と恒等経路の1に分かれて戻ることを確認します。

```python
import numpy as np

x = 2.0
theta = 0.1
y = theta * x + x
dy = y
dx = dy * theta + dy
analytic = (theta + 1.0) * y
print("y:", y)
print("dx:", dx)
print("max error:", abs(dx - analytic))
```

実行結果は `y: 2.2`、`dx: 2.4200000000000004`、`max error: 0.0` です。`dy * theta` だけを書けば主経路の寄与しか残らず、`dy` が恒等ショートカットから来る寄与です。実際の畳み込み実装では、この加算の前に両経路のテンソル形状を揃えます。

<div class="caution">

主経路の最後にReLUなどの非線形を置く実装では、加算点の前後で勾配の式が変わります。まずどこで加算したかを特定し、恒等経路が加算点まで本当に恒等かを確認してください。

</div>

## 取り違えやすいもの

| 用語 | Residual Blockとの切り分け |
|---|---|
| Dense layer | 前の全層出力を結合する接続。残差ブロックの加算とは結合方法が違う |
| Highway network | ゲートで経路の通過量を制御する。恒等ショートカットを単純加算する残差形とは別 |
| Projection shortcut | 次元を合わせるためのショートカット側の写像。Residual Block全体の名前ではない |
| Bottleneck block | 主経路を縮小・演算・復元する内部構成。残差加算の有無とは別の分類軸 |
| Skip connection | 層を飛び越す接続の一般名。恒等写像に限定されず、Residual Blockはその具体例 |

## 想起チェック

<details class="recall">
<summary>Residual Blockの出力を、主経路とショートカットで書くと</summary>

$\mathbf{y}=\mathcal{F}(\mathbf{x},\theta)+\mathcal{S}(\mathbf{x})$ です。同じ形状なら $\mathcal{S}(\mathbf{x})=\mathbf{x}$ とできます。

</details>

<details class="recall">
<summary>恒等ショートカットが逆伝播で加えるものは何か</summary>

加算点から来た勾配に、恒等写像のヤコビアン $\mathbf{I}$ を掛けた項です。そのため入力勾配には主経路の項だけでなく恒等経路の項も残ります。

</details>

<details class="recall">
<summary>主経路とショートカットの形状が違うとき何を置くか</summary>

学習可能な射影 $W_s$、畳み込みでは典型的に $1\times1$ 畳み込みを置いて、チャネル数や空間サイズを合わせます。

</details>

<details class="recall">
<summary>ボトルネックの3層構成と狙いは何か</summary>

$1\times1$ でチャネルを絞り、$3\times3$ を狭い表現に適用し、$1\times1$ で戻します。計算量の大きい中央演算を低チャネル数で行うためです。

</details>
