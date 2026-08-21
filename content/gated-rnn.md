---
exam: E資格
concept: ゲート付きRNN
slug: gated-rnn
tier: A
area: 深層学習
summary: LSTMとGRUは、状態を残すか更新するかを学習するゲートで、時間方向の勾配を運びやすくしたRNNです。
updated: 2026-08-22
sources:
  - title: "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
    url: https://arxiv.org/abs/1406.1078
  - title: "Deep Learning: Sequence Modeling"
    url: https://www.deeplearningbook.org/contents/rnn.html
---

## ひとことで言うと

ゲート付きRNNは、過去の状態をどれだけ残し、現在の入力をどれだけ取り込むかを、入力依存のゲートで制御するRNNです。LSTMはセル状態 $c_t$ と隠れ状態 $h_t$ を分け、GRUは隠れ状態だけで同じ役割を担います。

<div class="analogy">

状態を毎時刻ぜんぶ書き換えるメモ帳ではなく、保存、追記、閲覧のつまみを持つメモ帳です。古いメモを残すつまみを開けば、後の時刻まで同じ情報を運べます。

</div>

## なぜ必要か

素のRNNを時間方向に展開すると、遠い時刻の損失への勾配は、各時刻の状態遷移のヤコビ行列を掛けた積になります。各因子の最大特異値が恒常的に1未満なら積は指数的に小さくなり、1を超える方向が続けば爆発します。したがって、入力から離れた時刻の情報を、学習に必要な勾配と一緒に保つのが難しくなります。

$$
\frac{\partial L}{\partial \mathbf{h}_{t-k}}
=\frac{\partial L}{\partial \mathbf{h}_{t}}
\prod_{j=t-k+1}^{t}\frac{\partial \mathbf{h}_{j}}{\partial \mathbf{h}_{j-1}}
$$

ここで $L$ は損失、$k$ はさかのぼる時刻数、$\mathbf{h}_j$ は時刻 $j$ の隠れ状態です。この積が長期依存で小さくなることが、ゲートを導入する直接の動機になります。

ゲートはこの積を、単なる非線形変換の積から、状態をほぼそのまま通す経路との組合せに変えます。LSTMではセル状態に加算的な更新経路を設け、忘れる量と書き込む量を別々に決めます。GRUでは更新ゲートが旧状態と候補状態の混合比を決め、リセットゲートが候補を作るときに旧状態を見る量を決めます。どちらも「長期記憶を必ず保存する」のではなく、保存すべき時間スケールを学習する仕組みです。

## 仕組み

時刻 $t$ の入力を $\mathbf{x}_t$、隠れ状態を $\mathbf{h}_t$、LSTMのセル状態を $\mathbf{c}_t$、$\sigma$ をシグモイド関数、$\odot$ を要素ごとの積とします。LSTMの代表的な更新は次です。$W$ と $U$ は入力・隠れ状態から各ゲートへ写す学習パラメータ、$\mathbf{b}$ はバイアスです。

$$
\begin{aligned}
\mathbf{i}_t &= \sigma(W_i\mathbf{x}_t+U_i\mathbf{h}_{t-1}+\mathbf{b}_i), &
\mathbf{f}_t &= \sigma(W_f\mathbf{x}_t+U_f\mathbf{h}_{t-1}+\mathbf{b}_f),\\
\mathbf{o}_t &= \sigma(W_o\mathbf{x}_t+U_o\mathbf{h}_{t-1}+\mathbf{b}_o), &
\tilde{\mathbf{c}}_t &= \tanh(W_c\mathbf{x}_t+U_c\mathbf{h}_{t-1}+\mathbf{b}_c),\\
\mathbf{c}_t &= \mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t, &
\mathbf{h}_t &= \mathbf{o}_t\odot\tanh(\mathbf{c}_t).
\end{aligned}
$$

$\mathbf{i}_t$ は入力ゲート、$\mathbf{f}_t$ は忘却ゲート、$\mathbf{o}_t$ は出力ゲート、$\tilde{\mathbf{c}}_t$ は新しい候補、$\mathbf{c}_t$ はセル状態です。各ゲートの値は0〜1なので、入力を通す量、古いセル状態を残す量、セル状態を隠れ状態へ出す量を連続的に選べます。

ここでセル状態だけを $\mathbf{c}_{t-1}$ で微分すると、要素ごとに

$$
\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}}=\mathbf{f}_t
$$

です。よって時間を $k$ ステップさかのぼる経路には、おおむね $\prod_{j=t-k+1}^{t}\mathbf{f}_j$ が現れます。忘却ゲートが1に近い区間ではこの経路の勾配がそのまま通り、毎回tanhの導関数を掛ける構造より減衰しにくい、というのがLSTMの要点です。もちろん $\mathbf{f}_t$ が小さい区間では忘れるので、ゲートは勾配を無条件に保存する機構ではありません。

GRUは原論文の記法で、リセットゲート $\mathbf{r}_t$、更新ゲート $\mathbf{z}_t$、候補状態 $\tilde{\mathbf{h}}_t$ を

$$
\begin{aligned}
\mathbf{r}_t &= \sigma(W_r\mathbf{x}_t+U_r\mathbf{h}_{t-1}),\\
\mathbf{z}_t &= \sigma(W_z\mathbf{x}_t+U_z\mathbf{h}_{t-1}),\\
\tilde{\mathbf{h}}_t &= \phi(W\mathbf{x}_t+U(\mathbf{r}_t\odot\mathbf{h}_{t-1})),\\
\mathbf{h}_t &= \mathbf{z}_t\odot\mathbf{h}_{t-1}+(1-\mathbf{z}_t)\odot\tilde{\mathbf{h}}_t
\end{aligned}
$$

と計算します。$\phi$ は候補状態の活性化関数、$W_r,U_r,W_z,U_z,W,U$ は学習パラメータです。リセットゲートが0に近いと候補は旧状態をほぼ見ず、更新ゲートが1に近いと旧状態をそのまま引き継ぎます。LSTMのセル状態に相当する独立経路はありませんが、更新式の第1項が状態を直接運ぶ経路になります。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| LSTMの3ゲートの役割 | 入力・忘却・出力を、書込み・保持・読出しとして区別する | 出力ゲートをセル状態の更新量とする |
| 勾配が通る理由 | $\partial\mathbf{c}_t/\partial\mathbf{c}_{t-1}=\mathbf{f}_t$。忘却ゲートが1に近い経路がある | ゲートがあるから常に勾配消失しない |
| GRUの式 | 更新は旧状態と候補の混合、リセットは候補計算内の旧状態に掛かる | リセットゲートを最終状態の混合係数にする |
| LSTMとGRUの比較 | LSTMはセル状態と隠れ状態を分離し4つのゲート相当、GRUは隠れ状態のみで2ゲート | GRUにも独立したセル状態がある |

## 実装で確かめる

GRUの更新ゲートを1に近づけると、候補状態を計算しても出力は旧状態に近いままです。原論文の更新順をそのままNumPyで確認します。

```python
import numpy as np

def gru_step(x, h, Wr, Ur, Wz, Uz, W, U):
    sigmoid = lambda a: 1 / (1 + np.exp(-a))
    r = sigmoid(Wr @ x + Ur @ h)
    z = sigmoid(Wz @ x + Uz @ h)
    h_tilde = np.tanh(W @ x + U @ (r * h))
    return z * h + (1 - z) * h_tilde, r, z

x = np.array([0.2, -0.4])
h = np.array([0.7, -0.3])
zeros = np.zeros((2, 2))
Wz = np.eye(2) * 10; Uz = np.zeros((2, 2))
out, r, z = gru_step(x, h, zeros, zeros, Wz, Uz, zeros, zeros)
print(np.round(z, 6), np.round(out, 6))
```

このコードでは $\mathbf{z}=\sigma(10\mathbf{x})$ なので、実行結果は `[0.880797 0.017986] [ 0.616558 -0.005396]` です。第1成分は更新ゲートが大きく、候補がゼロでも旧状態 $0.7$ の寄与が大きく残ります。第2成分では更新ゲートが小さいため候補側（ここではゼロ）へ寄ります。この「どちらを採るか」が、ゲートを単なる係数ではなく状態更新の設計として読むポイントです。

<div class="caution">

ゲートの値が1に近いことと、勾配が完全に1であることは同じではありません。LSTMでもセル状態から隠れ状態へは出力ゲートとtanhを通りますし、ゲート自体の依存関係にも勾配があります。素通りと呼べるのは、セル状態間の直接経路に限った説明です。

</div>

## 取り違えやすいもの

| 比較軸 | LSTM | GRU |
|---|---|---|
| 状態 | セル状態 $\mathbf{c}_t$ と隠れ状態 $\mathbf{h}_t$ | 隠れ状態 $\mathbf{h}_t$ のみ |
| ゲート | 入力・忘却・出力 | リセット・更新 |
| 状態更新 | 古いセル状態を忘却し、候補を書き込む | 旧状態と候補を更新ゲートで混ぜる |
| パラメータ数 | 入力・隠れ状態から4本のアフィン変換 | 3本のアフィン変換（リセット、更新、候補） |
| 読み違え | 出力ゲートはセル状態から隠れ状態への出口 | 更新ゲートは候補を採る量ではなく、原論文式では旧状態を残す量 |

隠れサイズを $H$、入力サイズを $D$ とし、各アフィン変換にバイアスを含めるなら、LSTMはおおむね $4H(D+H+1)$、GRUは $3H(D+H+1)$ 個のパラメータです。実装では結合行列にまとめることがありますが、ゲート数に対応してこの差が生じます。GRUの更新ゲートの向きは実装ごとに別表現へ変形されることがあるため、試験では式の定義、実装では旧状態側に掛かる係数を確認します。

## 想起チェック

<details class="recall">
<summary>LSTMで長期の勾配を運ぶ直接経路はどの状態を通るか</summary>

セル状態 $\mathbf{c}_t$ です。セル状態間の微分は忘却ゲート $\mathbf{f}_t$ になり、これが1に近い区間では勾配が減衰しにくくなります。

</details>

<details class="recall">
<summary>LSTMの入力・忘却・出力ゲートを一言ずつ区別する</summary>

入力は候補を書き込む量、忘却は古いセル状態を残す量、出力はセル状態を隠れ状態へ出す量です。

</details>

<details class="recall">
<summary>GRUのリセットゲートと更新ゲートはどこに効くか</summary>

リセットゲートは候補状態を作るときの旧状態に掛かり、更新ゲートは旧状態と候補状態の混合比を決めます。原論文の式では更新ゲートが1に近いほど旧状態を残します。

</details>

<details class="recall">
<summary>LSTMとGRUの構造上の差を二つ挙げる</summary>

LSTMはセル状態を隠れ状態と分け、入力・忘却・出力の3ゲートを持ちます。GRUは隠れ状態だけを使い、リセット・更新の2ゲートで更新します。候補の変換を含めたアフィン変換は、それぞれ4本と3本です。

</details>
