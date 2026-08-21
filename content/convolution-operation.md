---
exam: E資格
concept: 畳み込み演算
slug: convolution-operation
tier: A
area: 深層学習
summary: 畳み込みは小さな重みを入力上で共有して局所特徴を取り出す線形演算で、深層学習の実装では通常、カーネルを反転しない相互相関として計算されます。
updated: 2026-08-22
sources:
  - title: "Deep Learning: Convolutional Networks"
    url: https://www.deeplearningbook.org/contents/convnets.html
  - title: "torch.nn.Conv2d"
    url: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
---

## ひとことで言うと

畳み込み演算は、入力の小さな領域に同じカーネルを順に適用し、位置ごとの特徴マップを作る線形演算です。数学の畳み込みはカーネルを反転しますが、ニューラルネットのライブラリが `convolution` と呼ぶ処理は、通常その反転をしない相互相関です。PyTorch の `Conv2d` も相互相関として定義されています。

<div class="analogy">

同じスタンプを紙面上で少しずつずらして押し、押した場所ごとの反応を地図にする作業です。スタンプの模様（重み）は場所ごとに作り直さず、同じものを使います。

</div>

## なぜ必要か

画像を1本のベクトルとして全結合層に入れると、各出力が全画素と結び付きます。畳み込みはこの仮定を崩し、近傍の画素だけを見る局所結合にします。入力の高さ・幅を $H_{in}, W_{in}$、入力チャネル数を $C_{in}$、出力チャネル数を $C_{out}$、カーネルの高さ・幅を $K_h, K_w$ とすると、バイアスを除く畳み込みのパラメータ数は、通常の `groups=1` で

$$
C_{out}C_{in}K_hK_w
$$

です。全結合層で同じ出力テンソルを作るなら、重みは

$$
(H_{in}W_{in}C_{in})(H_{out}W_{out}C_{out})
$$

となります。例えば $32\times32\times3$ から $32\times32\times16$ を作り、$3\times3$ カーネルを使う場合、畳み込みは重み $16\times3\times3\times3=432$ 個、全結合は $3072\times16384=50,331,648$ 個です。畳み込みでは、同じ局所パターンを場所を変えて検出できるため、重み共有によってこの差が生まれます。

## 仕組み

1枚の入力チャネル $X_k$ とカーネル $K_{j,k}$ の相互相関を、出力チャネル $j$ の位置 $(p,q)$ で表すと、次の形です。$b_j$ は出力チャネル $j$ のバイアス、$K_{j,k}(u,v)$ はカーネルの重みです。

$$
Y_j(p,q)=b_j+\sum_{k=0}^{C_{in}-1}\sum_{u=0}^{K_h-1}\sum_{v=0}^{K_w-1}X_k(p+u,q+v)K_{j,k}(u,v)
$$

各出力チャネルは、全入力チャネルの同じ位置の積和を足し合わせて1枚の特徴マップになります。したがってRGB入力なら、3チャネル分を別々に出すのではなく、3チャネルを合算した値が1つの出力チャネルの各位置に対応します。`Conv2d(in_channels, out_channels, kernel_size)` の重み形状は通常 `(out_channels, in_channels, K_h, K_w)` です。

出力の高さ・幅は、`stride` を $S_h,S_w$、片側の `padding` を $P_h,P_w$、`dilation` を $D_h,D_w$ として、

$$
H_{out}=\left\lfloor\frac{H_{in}+2P_h-D_h(K_h-1)-1}{S_h}+1\right\rfloor,
\quad
W_{out}=\left\lfloor\frac{W_{in}+2P_w-D_w(K_w-1)-1}{S_w}+1\right\rfloor
$$

です。`dilation` はカーネル点の間隔を広げる指定で、実効カーネル幅は $D_h(K_h-1)+1$ になります。`stride` は窓を動かす間隔、`padding` は入力の周囲に暗黙に加える幅です。端数は床関数で落ちるため、式を暗算するときも分子を先に計算します。

この構造の利点は3つです。局所結合は不要な遠距離の直接接続を減らし、重み共有は同じ検出器を全位置で使い回します。その結果、入力を平行移動すると特徴も対応する位置へ移るという並進同変性を持ちます。これは「どこに現れたか」を特徴マップに残す性質であり、位置を捨てる不変性とは別です。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| 畳み込みと相互相関の区別 | 実装の `Conv2d` はカーネル反転なしの相互相関 | 数学上の畳み込み式をそのまま実装仕様だとする |
| 3つの利点 | 局所結合・重み共有・並進同変性を対応づける | 並進**不変**性と同一視する |
| 出力サイズ | `stride`、片側 `padding`、`dilation`、床関数を式に入れる | dilation をカーネルサイズそのものに足す |
| チャネルの扱い | 1出力チャネルは全入力チャネルとの積和を合算する | 入力チャネルごとに独立した出力だけを作る |
| パラメータ数 | $C_{out}C_{in}K_hK_w+C_{out}$（biasあり） | 出力画像の面積を重み数にも掛ける |

## 実装で確かめる

次のコードは、3チャネル入力に対する相互相関をNumPyで計算し、`Conv2d` の重み数と出力サイズを同じ引数から求めます。`dilation=2` なので、$3\times3$ カーネルの実効サイズは $5\times5$ です。

```python
import numpy as np

x = np.arange(1 * 3 * 7 * 7, dtype=float).reshape(1, 3, 7, 7)
w = np.ones((2, 3, 3, 3))
b = np.array([0.5, -0.5])
stride, padding, dilation = 2, 1, 2
effective = dilation * (w.shape[2] - 1) + 1
xpad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
hout = (7 + 2 * padding - effective) // stride + 1
y = np.empty((1, 2, hout, hout))
for p in range(hout):
    for q in range(hout):
        window = xpad[:, :, p*stride:p*stride+effective:dilation, q*stride:q*stride+effective:dilation]
        y[:, :, p, q] = (window[:, None] * w[None]).sum(axis=(2, 3, 4)) + b
print(y.shape, w.size + b.size, y[0, :, 0, 0])
```

出力は `(1, 2, 3, 3) 56 [780.5 779.5]` です。形状の $3$ は、式に $H_{in}=7,K_h=3,S_h=2,P_h=1,D_h=2$ を代入した結果です。パラメータ数の $56$ は重み $2\times3\times3\times3=54$ にバイアス2個を加えた値で、出力画素数は含めません。

## 取り違えやすいもの

| 用語 | 畳み込み演算との切り分け |
|---|---|
| 数学上の畳み込み | カーネルを反転してから積和する。ニューラルネットのAPIとは式の向きを確認する |
| 相互相関 | カーネルを反転しない。PyTorch `Conv2d` が実際に採用する演算 |
| 全結合層 | 各出力が入力全体を見る。局所結合も重み共有もない |
| 並進不変性 | 入力をずらしても出力が変わらない性質。畳み込み単体の並進同変性とは異なる |
| `groups` | 入出力チャネル間の接続を分割する引数。`groups=1` は全入力を全出力へ接続し、`groups=in_channels` ではチャネルごとに分離する |

## 想起チェック

<details class="recall">
<summary>PyTorchのConv2dは数学上の畳み込みと同じ向きか</summary>

いいえ。カーネルを反転しない相互相関です。ライブラリの名前ではなく、実装の添字で判断します。

</details>

<details class="recall">
<summary>畳み込みのパラメータ数が出力位置数に依存しない理由は</summary>

同じカーネルを全位置で共有するためです。通常の `groups=1` で bias ありなら、重み数は $C_{out}C_{in}K_hK_w+C_{out}$ です。

</details>

<details class="recall">
<summary>畳み込みが持つ並進に関する性質は不変性か</summary>

同変性です。入力をずらすと、検出された特徴も対応してずれます。位置情報を捨てる不変性とは異なります。

</details>

<details class="recall">
<summary>Conv2dで入力チャネルは出力チャネルごとにどう扱われるか</summary>

各出力チャネルは、全入力チャネルに対応するカーネルとの積和を足し合わせます。`groups` を指定すると接続範囲を分割できます。

</details>
