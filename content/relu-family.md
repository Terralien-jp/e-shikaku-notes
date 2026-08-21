---
exam: E資格
concept: ReLU系活性化
slug: relu-family
tier: B
area: 深層学習
summary: ReLUの負側で勾配が止まる理由を起点に、Leaky ReLU・PReLU・ELU・GELUの負側の扱いと選択基準を整理します。
updated: 2026-08-22
sources:
  - title: "ReLU"
    url: https://docs.pytorch.org/docs/2.13/generated/torch.nn.ReLU.html
  - title: "LeakyReLU"
    url: https://docs.pytorch.org/docs/2.13/generated/torch.nn.LeakyReLU.html
  - title: "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
    url: https://arxiv.org/abs/1502.01852
---

## ひとことで言うと

ReLU系は、正側の勾配を保ったまま非線形性を入れる活性化です。焦点は負側で、出力をゼロにするか、小さな傾きや滑らかな値を残すかで、勾配と出力分布の性質が変わります。

<div class="analogy">

入力を坂道に流す弁だと考えると、ReLUは負側の流れを完全に止め、正側だけをそのまま通します。派生関数は、止めた側に細い通路を残すか、曲線の斜面を置くかを選んだものです。

</div>

## なぜ必要か

ReLUの正側では導関数が1なので、深い層を逆向きにたどっても、正側にいるユニットは活性化だけで勾配を縮めません。一方、負側では出力も導関数も0です。重み更新で入力が正側へ戻らない限り、そのユニットは学習に参加しない「死んだユニット」になります。大きな負のバイアス、学習率が大きい更新、入力分布の移動などで負側に固定されるのが実装上の発火条件です。

負側を完全に切る設計は計算が単純ですが、各ユニットの出力は非負になります。重みが正負を打ち消して次層へ渡す必要があり、平均が0からずれると、次層の入力の中心やバイアス更新にも影響します。したがって「勾配消失を避ける」だけでなく、負側をどれだけ情報経路として残すかで選びます。

| 状態 | 出力 | 逆伝播で起きること |
|---|---|---|
| ReLUの正側 | 入力を通す | 活性化の勾配は1 |
| ReLUの負側 | 0にする | 活性化の勾配は0 |
| Leaky/PReLUの負側 | 負の値を残す | 小さい傾きで更新経路を残す |

## 仕組み

活性化前の値を $x$、出力を $f(x)$、負側の傾きを決める係数を $\alpha$ とします。ReLUは次の形です。

$$
f(x)=\max(0,x),\qquad f'(x)=\begin{cases}1 & (x>0)\\0 & (x<0)\end{cases}
$$

したがって負側に入ったユニットの局所勾配は0です。Leaky ReLUは負側にも固定の傾き $\alpha>0$ を残します。

$$
f(x)=\begin{cases}x & (x\geq 0)\\\alpha x & (x<0)\end{cases},\qquad
f'(x)=\begin{cases}1 & (x>0)\\\alpha & (x<0)\end{cases}
$$

PReLUはこの $\alpha$ 自体を学習パラメータにしたものです。原論文の要点は、負側の傾きを固定せず、整流ユニットを一般化することです。ELUは負側を指数関数で滑らかに飽和させ、GELUは入力の大きさに応じて確率的に通す形の滑らかなゲートです。どちらも「負側を一律にゼロにする」ReLUとは異なりますが、負側の値が残ることと、負側の勾配が常に大きいことは別です。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| ReLUの勾配 | 正側は1、負側は0として読む | 負側も入力値に比例して勾配が残るとする |
| 死んだユニット | 負側に固定され、更新に必要な勾配が0になる状態 | 出力が一時的に0なら必ず死んだとする |
| 派生の比較 | Leakyは固定傾き、PReLUは傾きを学習、ELU/GELUは滑らかに負側を扱う | すべてを「ReLUの負側を少し通す」と同一視する |
| 出力分布 | ReLUは出力が非負で、平均0とは限らない | 正負が必ず対称になると考える |

## 実装で確かめる

負側の出力と数値微分を同じ入力で比較します。数値微分は0付近の不連続をまたぐと値が平均化されるため、ここでは0から離れた点を使います。

```python
import numpy as np

x = np.array([-2.0, -0.5, 0.5, 2.0])
alpha = 0.1
relu = np.maximum(0, x)
leaky = np.maximum(0, x) + alpha * np.minimum(0, x)
eps = 1e-6
leaky_num = (np.maximum(0, x + eps) + alpha * np.minimum(0, x + eps)
             - np.maximum(0, x - eps) - alpha * np.minimum(0, x - eps)) / (2 * eps)
print("ReLU:", relu)
print("LeakyReLU:", leaky)
print("Leaky slope:", leaky_num)
```

出力は `ReLU: [0.  0.  0.5 2. ]`、`LeakyReLU: [-0.2  -0.05  0.5   2.  ]`、`Leaky slope: [0.1 0.1 1.  1. ]` です。負側の値を残すと、ユニットが負側にいても勾配の経路が消えません。

## 取り違えやすいもの

| 手法 | 負側の扱い | 選択時の読み方 |
|---|---|---|
| ReLU | 0に切る | 単純さと正側の勾配を優先。ただし死んだユニットに注意 |
| Leaky ReLU | 固定係数で線形に通す | 負側にも確実に勾配を残す |
| PReLU | 負側の傾きを学習する | 表現力を増やす代わりに学習パラメータが増える |
| ELU | 負側を指数関数で滑らかに飽和 | 負側の値を残し、折れ曲がりを弱める |
| GELU | 入力の大きさに応じて滑らかに通す | 0/1の硬いゲートではなく、連続的な重み付けとして読む |

ReLUの出力平均が正にずれること自体は、直ちに失敗を意味しません。問題は、そのずれが次層の入力分布を動かし、学習率や初期化、正規化の前提と組み合わさることです。負側を残す派生はこのずれを小さくし得ますが、係数や曲線の選択で挙動は変わるため、「平均を必ず0にする関数」とは扱いません。

## 想起チェック

<details class="recall">
<summary>ReLUで死んだユニットが起きにくい理由は何か</summary>

負側では導関数が0なので、負側に固定されると、そのユニット自身の勾配による更新が起きません。入力やバイアスが正側へ戻る経路が残っている場合だけ再び動きます。

</details>

<details class="recall">
<summary>Leaky ReLUとPReLUの差は何か</summary>

どちらも負側に傾きを残しますが、Leaky ReLUの係数は固定、PReLUの係数は学習します。

</details>

<details class="recall">
<summary>ReLUの出力平均が0からずれるとき、何を見るか</summary>

出力が非負になるため平均は一般に0へは揃いません。次層の入力分布、バイアス、初期化や正規化との組合せを確認します。

</details>
