---
exam: E資格
concept: 深さ方向と点ごとの畳み込み
slug: depthwise-and-pointwise
tier: B
area: 深層学習
summary: 畳み込みをチャネルごとの空間処理と1×1のチャネル混合に分け、計算量とモデルサイズを大きく削減する構成です。
updated: 2026-08-22
sources:
  - title: "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    url: https://arxiv.org/abs/1704.04861
  - title: "Xception: Deep Learning with Depthwise Separable Convolutions"
    url: https://arxiv.org/abs/1610.02357
---

## ひとことで言うと

分離畳み込みは、標準畳み込みが一度に行う「空間方向の特徴抽出」と「チャネル方向の混合」を2段に分ける方法です。depthwise convolution（深さ方向）は入力チャネルごとに空間フィルタを1枚ずつ適用し、pointwise convolution（点ごと）は1×1畳み込みでチャネルを混ぜます。

<div class="analogy">

画像の各色を別々に調べ、その後に各結果を1画素ずつ混ぜる分業です。標準畳み込みより軽い一方、同じ計算ではありません。

</div>

## なぜ必要か

標準畳み込みでは、$K \times K$ のカーネルが入力チャネル $M$ 全てを見て、出力チャネル $N$ を作ります。入出力チャネルの積が計算量に効くため、モバイルでは遅延とサイズが制約になります。

分離すると、depthwise側はチャネルごとの空間処理、pointwise側は1×1でのチャネル混合を担当します。計算量とパラメータ数を削れる一方、標準畳み込みと同じ結合ではありません。MobileNetは幅や入力解像度の係数で遅延と精度も調整します。

| 観点 | 標準畳み込み | 分離畳み込み |
|---|---|---|
| 空間処理 | チャネル混合と同時 | depthwiseが担当 |
| チャネル混合 | $K×K$カーネル内で同時 | pointwiseが担当 |
| 主な狙い | 結合を保った表現 | 計算量・サイズの削減 |

## 仕組み

入力を $\mathbf{F} \in \mathbb{R}^{H \times W \times M}$、出力を $\mathbf{G} \in \mathbb{R}^{H \times W \times N}$、空間カーネルの幅を $K$ とします。標準畳み込みの計算量は

$$
C_{\mathrm{std}} = K^2 M N H W
$$

です。$M,N$は入力・出力チャネル数、$H,W$は空間サイズです。

depthwiseでは入力チャネル$m$ごとに$K \times K$のフィルタを1枚だけ適用し、計算量は $K^2 M H W$ です。pointwiseが各画素位置で$M$チャネルを$N$チャネルへ混ぜ、計算量は $M N H W$ です。

$$
C_{\mathrm{sep}} = K^2 M H W + M N H W = M H W(K^2+N)
$$

したがって標準畳み込みに対する割合は

$$
\frac{C_{\mathrm{sep}}}{C_{\mathrm{std}}} = \frac{1}{N} + \frac{1}{K^2}
$$

です。$K=3$なら$N$が大きい場合に$1/9$へ近づきますが、実際の割合は$N$でも変わります。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| 2段の役割を問う | depthwiseはチャネルごとの空間処理、pointwiseは1×1のチャネル混合 | depthwiseだけでチャネルを混ぜる |
| 計算量の比を問う | $1/N+1/K^2$ として、$M,H,W$が約分される | 常に$1/K^2$と断定する |
| 軽量モデルとの関係 | MobileNetは分離畳み込みを使い、遅延と精度を係数で調整する | 計算量削減が精度無償で得られる |
| 用語を切り分ける | 深さ方向はチャネル方向、pointwiseは空間1×1 | pointwiseを空間方向の分離と呼ぶ |

## 実装で確かめる

NumPyで標準畳み込みと分離畳み込みの乗算回数を比較します。

```python
K, M, N, H, W = 3, 32, 64, 28, 28
standard = K**2 * M * N * H * W
depthwise = K**2 * M * H * W
pointwise = M * N * H * W
separable = depthwise + pointwise
print(standard, separable, separable / standard)
```

出力は `45158400 5050368 0.1111111111111111` です。$N=64$では$1/64+1/9$となります。レイテンシは実装にも左右されます。

## 取り違えやすいもの

| 用語 | 分離畳み込みとの切り分け |
|---|---|
| [標準畳み込み](/learn/e-shikaku/convolution-operation/) | 空間処理とチャネル混合を1つの$K×K$カーネルで同時に行う |
| depthwise convolution | 入力チャネルごとに独立した空間フィルタを適用する前半 |
| pointwise convolution | 1×1畳み込みでチャネルを別の数へ射影する後半 |
| spatially separable convolution | $K×K$を$K×1$と$1×K$へ分ける別の分解。depthwiseとは軸が違う |
| Xception | 分離畳み込みを積み重ねる設計を提案し、Inceptionとの関係と実験を論じたモデル |

## 想起チェック

<details class="recall">
<summary>depthwiseとpointwiseは、それぞれ何を担当するか</summary>

depthwiseはチャネルごとの空間処理、pointwiseは1×1によるチャネル混合です。

</details>

<details class="recall">
<summary>標準畳み込みに対する計算量の割合は何か</summary>

$C_{\mathrm{sep}}/C_{\mathrm{std}}=1/N+1/K^2$ です。$N$にも依存します。

</details>

<details class="recall">
<summary>分離畳み込みの軽量化で何を交換しているか</summary>

計算量とサイズを抑える代わりに、標準畳み込みと同じ結合ではなくなります。MobileNetは係数で遅延と精度を調整します。

</details>
