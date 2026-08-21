---
exam: E資格
concept: 重みとバイアス
slug: weights-and-biases
tier: A
area: 深層学習
summary: 層の写像で、重みは入力の混ぜ方、バイアスは出力の基準位置を担います。パラメータ数の数え方と、初期化・実装上の登録までを確認します。
updated: 2026-08-22
sources:
  - title: "torch.nn.parameter.Parameter"
    url: https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
  - title: "Deep Learning"
    url: https://www.deeplearningbook.org/contents/mlp.html
---

## ひとことで言うと

全結合層は、入力を別の座標へ移す写像 $\mathbf{y}=W\mathbf{x}+\mathbf{b}$ を学習します。重み $W$ は各入力成分をどれだけ混ぜるか、バイアス $\mathbf{b}$ は入力がゼロでも出力に持たせる基準値を決めます。活性化関数を挟む場合も、まずこのアフィン変換が層の可動部分です。

<div class="analogy">

重みは各入力つまみの音量、バイアスは入力が無音でも設定しておく基準音量です。つまみの比率だけ変えても、無音のときの出力位置は動かせません。基準音量を別に持つことで、同じ混ぜ方のまま出力全体を移動できます。

</div>

## なぜ必要か

重みだけの写像 $\mathbf{y}=W\mathbf{x}$ は、必ず原点を原点へ写します。したがって、入力空間の原点を通らない位置に決定境界を置きたい場合や、入力がゼロでも非ゼロの応答を出したい場合、そのままでは表現できません。バイアスを足すと、写像の傾きや方向を重みで、位置をバイアスで独立に調整できます。

$$
\mathbf{x}=\mathbf{0}\ \Longrightarrow\ W\mathbf{x}=\mathbf{0}
$$

これは「バイアスがあると何でも表現できる」という意味ではありません。1層の線形写像にバイアスを足したものはアフィン写像であり、非線形な曲がりは活性化関数や層の積み重ねが担います。試験では、重みを傾き、バイアスを切片と対応づけると、原点を通る制約を見落としません。

## 仕組み

入力次元を $d_{\mathrm{in}}$、出力次元を $d_{\mathrm{out}}$ とし、$\mathbf{x}\in\mathbb{R}^{d_{\mathrm{in}}}$、$W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$、$\mathbf{b}\in\mathbb{R}^{d_{\mathrm{out}}}$ とします。層の出力は

$$
\mathbf{y}=W\mathbf{x}+\mathbf{b}
$$

です。$W$ の第 $j$ 行は出力 $y_j$ のための入力の組合せで、$W_{ji}$ は入力 $x_i$ がその出力へ寄与する係数です。$\mathbf{b}$ の第 $j$ 成分 $b_j$ は、入力に依存しない出力 $y_j$ のずれです。ミニバッチでは $X\in\mathbb{R}^{n\times d_{\mathrm{in}}}$ に対して $XW^\mathsf{T}+\mathbf{b}$ と書け、$\mathbf{b}$ は全サンプルの行へブロードキャストされます。

パラメータ数は「要素数を足す」だけです。重みは $d_{\mathrm{out}}d_{\mathrm{in}}$ 個、バイアスは $d_{\mathrm{out}}$ 個なので、合計は

$$
N=d_{\mathrm{out}}d_{\mathrm{in}}+d_{\mathrm{out}}=d_{\mathrm{out}}(d_{\mathrm{in}}+1)
$$

です。例えば入力64、出力128なら、重みは $128\times64=8192$ 個、バイアスは128個、合計8320個です。畳み込みでも考え方は同じで、カーネルの要素数に出力チャネル数を掛けた重みと、通常は出力チャネルごとのバイアスを数えます。

初期化が必要なのは、学習開始時に各ユニットが同じ役割になるのを避けるためです。隠れ層の重みを全て同じ値にすると、同じ入力を受けるユニットの出力と勾配も同じになり、更新後も同じ振る舞いを続けます。これが対称性です。乱数などでユニットごとに異なる重みを与えると、異なる方向の勾配を受けられます。ここで重要なのは特定の初期化手法の名前ではなく、対称性を残したままではユニットが分化できないという因果関係です。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| $W\mathbf{x}+\mathbf{b}$ の各項の役割 | $W$ は入力の線形結合、$\mathbf{b}$ は入力に依存しないシフト | バイアスを入力の重みと同じものとして扱う |
| 全結合層のパラメータ数 | 入力次元×出力次元＋出力次元 | バイアスを数えない／入力次元を足す |
| バイアスを除いた表現力 | 原点を原点へ写す線形写像に限定される | 活性化関数があればバイアス不要とする |
| 初期化の目的 | 同一値の重みが作る対称性を破り、ユニットを分化させる | 初期化だけで学習済みの性能になると考える |

## 実装で確かめる

PyTorchでは、学習対象にしたいテンソルを `Parameter` として持たせます。`Parameter` は単なるテンソル値ではなく、モジュールのパラメータとして扱われる型です。`nn.Module` の属性に代入すると、オプティマイザへ渡すパラメータ列に登録されます。

```python no-run
import torch
from torch import nn

class Affine(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dout, din))
        self.bias = nn.Parameter(torch.zeros(dout))

    def forward(self, x):
        return x @ self.weight.T + self.bias

layer = Affine(64, 128)
print(sum(p.numel() for p in layer.parameters()))  # 8320
```

ここで `weight` を普通の Tensor として属性に置くと、値は計算に使えても、通常の `Module` のパラメータ登録から外れます。逆に `Parameter` にすれば、勾配を計算するだけでなく、`layer.parameters()` を通じて更新対象として列挙できます。形状と登録状態は別の確認事項です。

<div class="caution">

`bias=False` の層はバイアスの要素数がゼロになるだけで、重みが入力を別の位置へ移せるようにはなりません。パラメータ数を計算するときは、コードの設定でバイアスが無効化されていないかを先に確認します。

</div>

## 取り違えやすいもの

| 用語 | 重み・バイアスとの切り分け |
|---|---|
| 線形写像 | 数学的には原点を保つ $W\mathbf{x}$。バイアスを含む $W\mathbf{x}+\mathbf{b}$ はアフィン写像です |
| [活性化関数](/learn/e-shikaku/activation-functions/) | 層の出力を非線形に変換するもの。バイアスのように全体を加算シフトするパラメータではありません |
| 学習率 | パラメータ更新の幅を決めるハイパーパラメータ。重みやバイアスそのものではありません |
| `Parameter` | 値の役割ではなく、PyTorchでモジュールの学習対象として登録される Tensor の型です |
| [初期化](/learn/e-shikaku/parameter-initialization/) | 学習前の値を決める操作。学習中に勾配で値を変える更新とは別です |

## 想起チェック

<details class="recall">
<summary>層の線形変換で、重みとバイアスは何を担うか</summary>

重み $W$ は入力成分の混ぜ方と係数、バイアス $\mathbf{b}$ は入力に依存しない出力のシフトを担います。

</details>

<details class="recall">
<summary>入力64、出力128の全結合層のパラメータ数は</summary>

重みが $128\times64=8192$ 個、バイアスが128個なので、合計8320個です。

</details>

<details class="recall">
<summary>バイアスを外すと何が表現できなくなるか</summary>

写像は $W\mathbf{x}$ に限定され、原点を原点へ写します。原点を通らない切片や、入力ゼロでの非ゼロ出力を直接表現できません。

</details>

<details class="recall">
<summary>重みを全て同じ値で初期化すると何が起きるか</summary>

同じ入力を受けるユニットの出力と勾配が同じになり、更新後も同じ役割に留まります。初期化ではこの対称性を破る必要があります。

</details>
