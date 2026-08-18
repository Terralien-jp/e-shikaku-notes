---
exam: E資格
concept: 誤差逆伝播法
slug: backpropagation
tier: A
area: 深層学習
summary: 合成関数の微分を出力側から再利用しながら計算し、全パラメータの勾配を1回の走査で求める手法。
updated: 2026-08-18
sources:
  - title: "Autograd mechanics — PyTorch documentation"
    url: https://pytorch.org/docs/stable/notes/autograd.html
---

<!-- ★これは形の見本です。content/ には置かず、サイトにも出しません。
     分量は規約（Tier A = 1,800〜2,800字）に足りていません。構成だけを見てください。 -->

## ひとことで言うと

**連鎖律を出力側から適用して、途中結果を使い回しながら全パラメータの勾配を求める手続き**です。

<div class="analogy">
<strong>要するに、伝票を遡って各部署の責任額を確定させる作業です。</strong>最終的な損失という1つの金額があり、それが各層のパラメータのせいでどれだけ動くかを、出力に近い側から順に配分していきます。前から順に計算し直すと同じ部分計算を何度もやることになるので、後ろから配るほうが安い。
</div>

## 仕組み

損失 $L$ に対する層 $l$ の重み $W^{(l)}$ の勾配は、連鎖律で次のように分解されます。

$$
\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}
$$

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| 計算量の比較 | 出力側からの走査で部分結果を再利用する | 前向きに全パラメータぶん微分する案 |

## 想起チェック

<details class="recall">
<summary>なぜ出力側から計算するのか</summary>
途中の偏微分を使い回せるため。入力側からだとパラメータごとに走査が要る。
</details>
