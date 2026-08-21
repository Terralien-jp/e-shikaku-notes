---
exam: E資格
concept: GRU
slug: gru
tier: B
area: 深層学習
summary: GRUは更新ゲートとリセットゲートで隠れ状態を直接制御する、セル状態を持たないゲート付きRNNです。
updated: 2026-08-22
sources:
  - title: "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
    url: https://arxiv.org/abs/1406.1078
  - title: "Deep Learning: Sequence Modeling"
    url: https://www.deeplearningbook.org/contents/rnn.html
---

## ひとことで言うと

GRU（Gated Recurrent Unit）は、LSTMに動機づけられたゲート付きRNNです。更新ゲートとリセットゲートの2つで過去の隠れ状態を残す量と、候補状態を作るときに過去を参照する量を調整します。LSTMのような独立したセル状態は持たず、隠れ状態だけを引き継ぎます。

<div class="analogy">

一つのメモ帳だけを使い、更新ゲートで「前のページをどれだけ残すか」、リセットゲートで「新しいページを書くとき前のページをどれだけ読むか」を決める方式です。保管場所を隠れ状態に一本化するので、状態の対応を追いやすくなります。

</div>

## なぜ必要か

通常のRNNでは、各時刻の隠れ状態を非線形変換で作り直すため、長い系列の情報を学習で保ちにくいという問題があります。GRUは、過去をそのまま混ぜて次へ渡す経路をゲートで作り、長期情報を残す判断をデータから学習できるようにしました。

原論文は、提案した隠れユニットをLSTMより計算・実装しやすいものとして説明しています。ゲートが2つで、セル状態も別に持たないため、同じ隠れサイズなら管理する状態とゲート由来の計算が少なく、一般にパラメータ数と計算量を抑えやすいのが実装上の利点です。ただし実測速度はカーネルやバッチ形状にも左右されます。GRUとLSTMのどちらが常に優れるかは、タスクや設定を離れて決着した話ではありません。

| 不便 | GRUで導入するもの | 得られる制御 |
|---|---|---|
| 過去を毎回作り直す | 更新ゲート | 隠れ状態を保持する量 |
| 古い情報を候補に混ぜ続ける | リセットゲート | 過去を参照する量 |

## 仕組み

入力を $\mathbf{x}_t$、前時刻の隠れ状態を $\mathbf{h}_{t-1}$、シグモイド関数を $\sigma$、要素積を $\odot$ とします。まずリセットゲート $\mathbf{r}_t$ と更新ゲート $\mathbf{z}_t$ を計算します。

$$
\mathbf{r}_t=\sigma(\mathbf{W}_r\mathbf{x}_t+\mathbf{U}_r\mathbf{h}_{t-1}),\qquad
\mathbf{z}_t=\sigma(\mathbf{W}_z\mathbf{x}_t+\mathbf{U}_z\mathbf{h}_{t-1})
$$

リセットゲートは候補状態を作るときの過去の参照量です。候補状態 $\tilde{\mathbf{h}}_t$ は、過去の隠れ状態に $\mathbf{r}_t$ を掛けてから変換します。$\phi$ は通常 $\tanh$ を使う非線形関数です。

$$
\tilde{\mathbf{h}}_t=\phi\left(\mathbf{W}\mathbf{x}_t+\mathbf{U}(\mathbf{r}_t\odot\mathbf{h}_{t-1})\right)
$$

最後に更新ゲートで、過去を残す側と候補を採用する側を混ぜます。

$$
\mathbf{h}_t=\mathbf{z}_t\odot\mathbf{h}_{t-1}+(1-\mathbf{z}_t)\odot\tilde{\mathbf{h}}_t
$$

$\mathbf{z}_t$ が1に近ければ過去を多く保持し、0に近ければ候補へ更新します。$\mathbf{r}_t$ が0に近い部分では、候補状態が過去をほぼ無視します。試験では更新式の係数の向き、リセットを掛ける位置、セル状態を別記しない点が確認箇所です。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| ゲートの役割 | 更新は過去を残す量、リセットは候補計算で過去を読む量 | 2つの役割を逆にする |
| 状態の構造 | 隠れ状態を更新し、独立したセル状態を持たない | LSTMと同じ2状態とする |
| 更新式の読み取り | $\mathbf{z}_t$ と $1-\mathbf{z}_t$ が過去と候補を分担する | 両方に同じ係数を掛ける |
| LSTMとの比較 | 構造が簡素で、優劣はタスク依存 | GRUが常に高精度・高速と断定する |

## 実装で確かめる

次のコードは、1時刻ぶんのGRU更新をNumPyでそのまま計算します。`z` が大きいと、出力が前の隠れ状態に近づくことを確認できます。

```python
import numpy as np

def gru_step(x, h, Wz, Uz, Wr, Ur, W, U):
    sigmoid = lambda a: 1 / (1 + np.exp(-a))
    z = sigmoid(Wz @ x + Uz @ h)
    r = sigmoid(Wr @ x + Ur @ h)
    h_tilde = np.tanh(W @ x + U @ (r * h))
    return z * h + (1 - z) * h_tilde, z, r

rng = np.random.default_rng(0)
x, h = rng.normal(size=3), rng.normal(size=4)
M = [rng.normal(size=(4, 3)), rng.normal(size=(4, 4))]
new_h, z, r = gru_step(x, h, M[0], M[1], M[0], M[1], M[0], M[1])
print(new_h.shape, z.min() >= 0 and z.max() <= 1, r.min() >= 0 and r.max() <= 1)
```

## 取り違えやすいもの

| 用語 | GRUとの切り分け |
|---|---|
| LSTM | ゲートとセル状態を持つ。GRUは隠れ状態に状態を統合する |
| 通常のRNN | ゲートなしで候補をほぼ毎回更新する。GRUは保持・リセットを学習する |
| 更新ゲート | 過去の隠れ状態と候補の混合比を決める |
| リセットゲート | 候補状態の計算で過去の隠れ状態をどれだけ使うか決める |

## 想起チェック

<details class="recall">
<summary>GRUがLSTMと異なり別のセル状態を持たない理由は</summary>

GRUは隠れ状態だけを更新し、更新ゲートで保持経路を作ります。状態を隠れ状態に統合した構造です。

</details>

<details class="recall">
<summary>更新ゲートが1に近いと、隠れ状態はどうなるか</summary>

$\mathbf{h}_t=\mathbf{z}_t\odot\mathbf{h}_{t-1}+(1-\mathbf{z}_t)\odot\tilde{\mathbf{h}}_t$ なので、前時刻の隠れ状態を多く保持します。

</details>

<details class="recall">
<summary>リセットゲートが0に近いと候補状態の計算はどう変わるか</summary>

$\mathbf{r}_t\odot\mathbf{h}_{t-1}$ が小さくなり、候補状態は現在入力を中心に、過去をほぼ無視して計算されます。

</details>
