---
exam: E資格
concept: BPTT
slug: bptt
tier: B
area: 深層学習
summary: RNNを時間方向に展開し、各時刻の損失から共有パラメータへ逆向きに勾配を集める手続きと、長い系列を区切るtruncated BPTTを整理します。
updated: 2026-08-22
sources:
  - title: "Sequence Modeling: Recurrent and Recursive Nets"
    url: https://www.deeplearningbook.org/contents/rnn.html
  - title: "Advances in Optimizing Recurrent Networks"
    url: https://arxiv.org/abs/1212.0901
---

## ひとことで言うと

BPTT（Backpropagation Through Time）は、RNNを時刻ごとに横へ展開した計算グラフに、通常の誤差逆伝播を適用する手続きです。各時刻で同じ重みを使うため、逆向きにたどりながら得た勾配を時刻方向に足し合わせてから更新します。

<div class="analogy">

同じ担当者が毎日の判断を行う業務記録を、日付ごとの別担当者として並べて検証するようなものです。最後の判断の誤りを前の日へ戻し、各日の担当者が共通の手順にどれだけ関与したかを合算します。

</div>

## なぜ必要か

通常の逆伝播は層の順序を逆にたどりますが、RNNの依存関係には時刻もあります。時刻 $t$ の状態は $t-1$ の状態を使うため、系列の最後で生じた損失の原因を前の時刻まで帰属させるには、時間方向へグラフを展開する必要があります。

展開長を $T$ とすると、順伝播も逆伝播も基本的に $T$ 回分のセル計算を行います。さらに各時刻の状態や活性化を逆伝播用に保持するため、計算量とメモリ使用量は展開長に比例して増えます。長い系列をそのまま一括処理できないとき、精度だけでなくこの二つの資源が区切り方を決めます。

| 展開方法 | 逆伝播の範囲 | 主な負担 |
|---|---|---|
| 完全なBPTT | 系列全体 | 長いほど計算量と保存状態が増える |
| truncated BPTT | 固定長の窓だけ | 境界より前へ勾配が届かない |

## 仕組み

入力を $\mathbf{x}_t$、隠れ状態を $\mathbf{h}_t$、共有される再帰重みを $W_{hh}$、入力重みを $W_{xh}$、バイアスを $\mathbf{b}$、状態更新の活性化関数を $f$ とします。時刻 $t$ の状態と損失 $L_t$ は次のように計算します。

$$
\mathbf{h}_t=f(W_{hh}\mathbf{h}_{t-1}+W_{xh}\mathbf{x}_t+\mathbf{b}),\qquad L=\sum_{t=1}^{T}L_t
$$

$T$ は展開する時刻数、$L_t$ は時刻 $t$ の損失です。逆伝播では $L_t$ の影響を $\mathbf{h}_{t-1}$ へ戻し、同じ $W_{hh}$ が各時刻で使われた分の寄与を合計します。概念的には、状態に対する勾配を $\boldsymbol{\delta}_t=\partial L/\partial\mathbf{h}_t$ とすると、次の時刻から来る項を含めて計算します。

$$
\boldsymbol{\delta}_t=\frac{\partial L_t}{\partial\mathbf{h}_t}+W_{hh}^{\mathsf{T}}\boldsymbol{\delta}_{t+1}\odot f'(\mathbf{a}_t),\qquad \frac{\partial L}{\partial W_{hh}}=\sum_{t=1}^{T}\boldsymbol{\delta}_t\mathbf{h}_{t-1}^{\mathsf{T}}
$$

$\mathbf{a}_t$ は活性化前の値、$f'$ はその導関数、$\odot$ は要素ごとの積です。重要なのは、時間ごとに別の重みを学習するのではなく、共有重みへの勾配を全時刻から集める点です。

truncated BPTTでは、系列を長さ $K$ の窓に分け、各窓の中だけを逆伝播します。窓の先頭で勾配を止めるため、前窓から渡した状態を数値として使えても、その状態を作った計算グラフには戻りません。したがって勾配は境界を越えて伝わらず、$K$ より前の入力が現在の損失へ与える勾配はその更新では計算されません。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| BPTTの対象 | RNNを時間方向に展開し、通常の逆伝播を適用する | 時刻ごとに別モデルを学習すると考える |
| 計算量・メモリ | 展開長に比例して増える。状態保存も必要 | 共有重みだから時刻数に依存しないとする |
| truncated BPTT | 長さ $K$ の範囲で逆伝播し、境界で勾配を切る | 状態を渡せば勾配も前窓へ届くとする |
| 状態の扱い | 窓間で状態を引き継ぐか、初期化するかを目的に合わせて選ぶ | 状態を引き継ぐことと計算グラフをつなぐことを同一視する |

## 実装で確かめる

窓ごとに状態を渡す実装と、窓ごとに状態をゼロへ戻す実装は、順方向の状態の扱いが異なります。自動微分ライブラリでは、引き継ぐ状態を `detach` してから次の窓へ渡すと、値だけを継承してBPTTの境界を作れます。

```python
import numpy as np

def run_chunks(xs, W, U, chunk, carry_state):
    h = np.zeros(W.shape[0])
    outputs = []
    for start in range(0, len(xs), chunk):
        if not carry_state:
            h = np.zeros_like(h)
        for x in xs[start:start + chunk]:
            h = np.tanh(W @ h + U @ x)
            outputs.append(h.copy())
        # NumPyではグラフを持たないが、ここがdetachする境界に相当する
    return np.asarray(outputs)

rng = np.random.default_rng(0)
xs = rng.normal(size=(6, 2))
W, U = rng.normal(size=(3, 3)), rng.normal(size=(3, 2))
assert run_chunks(xs, W, U, 2, True).shape == (6, 3)
```

## 取り違えやすいもの

| 用語 | BPTTとの切り分け |
|---|---|
| [通常の逆伝播](/learn/e-shikaku/backpropagation/) | 層方向の計算グラフを逆にたどる。BPTTはその考え方を時間方向へ適用する |
| truncated BPTT | BPTTの近似的な実行方法。指定窓の外へ勾配を流さない |
| 状態リセット | 次の窓の初期状態をゼロなどにする実装。勾配を切ることとは別の選択 |
| 勾配クリッピング | 勾配の大きさを制限する処理。逆伝播をどこまで行うかは決めない |

## 想起チェック

<details class="recall">
<summary>BPTTで何を時間方向に行うか</summary>

RNNの計算グラフを時刻ごとに展開し、そのグラフへ逆伝播を適用します。

</details>

<details class="recall">
<summary>展開長を増やすと何が増えるか</summary>

セル計算の回数と、逆伝播のために保持する状態の量が増えます。

</details>

<details class="recall">
<summary>状態を引き継いだtruncated BPTTで、勾配も境界を越えるか</summary>

越えません。状態の数値は次の窓へ渡せますが、境界より前の計算グラフを切れば、その範囲へは勾配が伝わりません。

</details>
