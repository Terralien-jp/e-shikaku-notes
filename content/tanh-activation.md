---
exam: E資格
concept: tanh活性化
slug: tanh-activation
tier: C
area: 深層学習
summary: tanhは出力を-1から1へ収め、0を中心にする活性化関数です。勾配はシグモイドより通りやすい一方、飽和と勾配消失は残ります。
updated: 2026-08-22
sources:
  - title: "Tanh"
    url: https://docs.pytorch.org/docs/2.13/generated/torch.nn.Tanh.html
  - title: "Practical recommendations for gradient-based training of deep architectures"
    url: https://arxiv.org/abs/1206.5533
---

## ひとことで言うと

tanhは入力を $-1$ から $1$ の範囲へ写し、出力の中心を0にする活性化関数です。隠れ状態の符号を保ちたい層で使いやすく、LSTMやGRUでは候補値の生成に登場します。

<div class="analogy">

入力を、0を中心にした滑らかな出力メーターへ変換します。大きな正負の入力はそれぞれ上限・下限に張り付き、小さな入力だけがよく動きます。

</div>

## なぜ必要か

シグモイドの出力は正の側へ偏りますが、tanhは0中心です。さらに原点での導関数の最大値は1で、シグモイドの最大値0.25より大きいため、同じ入力尺度なら勾配が通りやすいという利点があります。ただし「勾配消失がない」という意味ではありません。入力の絶対値が大きい領域では飽和し、導関数は0に近づきます。

シグモイドを $\sigma$ と書くと、tanhは入力と出力を線形にずらした関係でも表せます。

$$
\tanh(x)=2\sigma(2x)-1
$$

## 仕組み

入力を $x$、出力を $y$ とすると、定義は次のとおりです。

$$
y = \tanh(x) = \frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}
$$

導関数は出力自身で表せます。

$$
\frac{\mathrm{d}y}{\mathrm{d}x}=1-y^2
$$

$x=0$ では導関数が1、$|x|$ が大きいと $y$ が $\pm1$ に近づいて導関数が0になります。したがって、0中心という性質と勾配の通りやすさは、飽和しないことを保証しません。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| 出力範囲と中心 | 範囲は $(-1,1)$、中心は0 | $(0,1)$ とする |
| 勾配の比較 | 原点で最大1、シグモイドは最大0.25 | 「常に勾配が1」 |
| 導関数 | $1-\tanh^2(x)$、または $1-y^2$ | $y(1-y)$ とする |
| RNN系での用途 | LSTM/GRUの候補値を作る | ゲートの出力と混同する |

## 実装で確かめる

NumPyで値と導関数を同時に確認します。

```python
import numpy as np

x = np.array([-3.0, 0.0, 3.0])
y = np.tanh(x)
dy = 1.0 - y**2
print("y:", np.round(y, 6))
print("dy:", np.round(dy, 6))
```

実行結果は `y: [-0.995055  0.        0.995055]`、`dy: [0.009866 1.       0.009866]` です。両端で値がほぼ張り付き、中心だけ導関数が大きいことが分かります。

## 取り違えやすいもの

| 対象 | tanhとの切り分け |
|---|---|
| [シグモイド](/learn/e-shikaku/sigmoid-activation/) | 出力は $(0,1)$。tanhは0中心で、原点の最大導関数が大きい |
| [ReLU](/learn/e-shikaku/relu-family/) | 正の側は飽和しないが、負の側で勾配が0になる。tanhは両側で飽和する |
| [LSTM/GRUのゲート](/learn/e-shikaku/gated-rnn/) | ゲートは主にシグモイド、候補値の生成はtanh。役割を混ぜない |

## 想起チェック

<details class="recall">
<summary>tanhの出力範囲と、原点での導関数は</summary>

出力は $(-1,1)$、原点での導関数は1です。

</details>

<details class="recall">
<summary>tanhは飽和しないか</summary>

飽和します。$|x|$ が大きいと出力は $\pm1$ に近づき、導関数は0に近づきます。

</details>
