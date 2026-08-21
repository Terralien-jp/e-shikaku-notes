---
exam: E資格
concept: KLとJSダイバージェンス
slug: kl-and-js-divergence
tier: B
area: 応用数学
summary: KLダイバージェンスの向きが生む違いと、対称化したJSダイバージェンスの性質を式と実装で整理します。
updated: 2026-08-22
sources:
  - title: "Divergence measures based on the Shannon entropy"
    url: https://doi.org/10.1109/18.61115
  - title: "Deep Learning: Probability and Information Theory"
    url: https://www.deeplearningbook.org/contents/prob.html
---

## ひとことで言うと

KLダイバージェンスは、基準分布 $p$ を近似分布 $q$ で表したときの情報量のずれです。JSダイバージェンスは、$p$ と $q$ の中間分布を介してKLを対称化した量です。どちらも「確率分布同士の差」を測りますが、KLは向きを持ち、JSは対称です。

<div class="analogy">

KLは「正解分布が出す場所を、近似分布がどれだけ説明し損ねたか」を片方向に採点します。JSは両者の中間案を作ってから、正解側と近似側を同じ重みで採点します。

</div>

## なぜ必要か

平均二乗誤差は、確率分布の形や確率の比を直接扱いません。たとえば、正解分布が確率を置く領域で $q(x)$ がゼロなら、そこを説明できないことを大きく罰したい場合があります。このとき確率の対数比を使うKLが自然です。

一方、KLは $p$ と $q$ を入れ替えると値が変わります。分布間の比較を一方に依存しない形で行いたいときは、混合分布を使うJSを選びます。したがって「どちらが正しい距離か」ではなく、何を基準に誤差を測るかで使い分けます。

| 目的 | 向いている量 |
|---|---|
| 基準分布の確率質量を近似側に覆わせる | $D_{\mathrm{KL}}(p\|q)$ |
| 比較を対称にし、上限のある値にする | $D_{\mathrm{JS}}(p,q)$ |

## 仕組み

$p(x)$ を基準分布、$q(x)$ を近似分布、$x$ を確率変数とすると、KLダイバージェンスは

$$
D_{\mathrm{KL}}(p\|q)=\mathbb{E}_{x\sim p}\left[\log\frac{p(x)}{q(x)}\right]
$$

です。期待値を $p$ から取るため、$p(x)$ が大きい場所で $q(x)$ が小さいと強く効きます。$p$ と $q$ の順序を逆にした

$$
D_{\mathrm{KL}}(q\|p)=\mathbb{E}_{x\sim q}\left[\log\frac{q(x)}{p(x)}\right]
$$

は、$q$ が確率を置いた場所で $p$ を評価します。これが同じ分布対でも値や最適化の傾向が変わる理由です。KLは常に非負ですが、一般には対称でなく、三角不等式も満たさないため距離ではありません。$p(x)>0$ なのに $q(x)=0$ なら、$D_{\mathrm{KL}}(p\|q)$ は発散します。

JSでは混合分布 $m(x)=(p(x)+q(x))/2$ を作り、

$$
D_{\mathrm{JS}}(p,q)=\frac{1}{2}D_{\mathrm{KL}}(p\|m)+\frac{1}{2}D_{\mathrm{KL}}(q\|m)
$$

とします。$m$ は両方の分布が確率を置く場所を含むので、片方の分布のゼロ確率による直接的な発散を避けられます。JSは $p$ と $q$ を交換しても同じです。対数の底を2にすれば値は $0$ 以上 $1$ 以下になり、自然対数なら上限は $\log 2$ です。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| KLの性質 | 非負だが、非対称で距離ではない | 非負だから距離だとする |
| KLの向き | 期待値をどちらの分布から取るかで変わる | $D_{\mathrm{KL}}(p\|q)=D_{\mathrm{KL}}(q\|p)$ とする |
| JSの定義 | $m=(p+q)/2$ を介した2つのKLの平均 | KLを単に絶対値化した量とする |
| 上限 | 対数の底が2なら $1$、自然対数なら $\log 2$ | 底を無視して上限を固定する |

## 実装で確かめる

離散分布で、KLの向きとJSを計算します。ゼロ確率の項は数学上の発散を `log(0)` に任せず、条件分岐で表します。

```python
import math

p = [0.5, 0.5, 0.0]
q = [0.5, 0.0, 0.5]

def kl(a, b):
    if any(x > 0 and y == 0 for x, y in zip(a, b)):
        return math.inf
    return sum(x * math.log(x / y) for x, y in zip(a, b) if x > 0)

m = [(x + y) / 2 for x, y in zip(p, q)]
js = (kl(p, m) + kl(q, m)) / 2
print(kl(p, q), kl(q, p), js)
```

出力は `inf inf 0.34657359027997264` です。両方向のKLはゼロ確率で発散しますが、混合分布 $m$ とのKLは有限になるため、JSは有限です。

## 取り違えやすいもの

| 量 | 切り分け |
|---|---|
| $D_{\mathrm{KL}}(p\|q)$ | $p$ を基準に $q$ の説明不足を測る。順序を入れ替えられない |
| $D_{\mathrm{KL}}(q\|p)$ | $q$ を基準に評価する別の量。同じ「KL」でも値は別 |
| $D_{\mathrm{JS}}(p,q)$ | 混合分布を介する対称な比較。上限も対数の底で決まる |
| 交差エントロピー | $H(p,q)=-\mathbb{E}_{p}[\log q(x)]$。$H(p,q)=H(p)+D_{\mathrm{KL}}(p\|q)$ で、KLそのものではない |

特に実装では、損失の入力が確率なのか対数確率なのかを確認します。数式の $q$ をそのまま渡すAPIとは限らず、対数確率を受け取る実装では、式の形を変換してから計算します。

## 想起チェック

<details class="recall">
<summary>KLダイバージェンスは距離か</summary>

距離ではありません。非負ですが、一般に対称性と三角不等式を満たしません。

</details>

<details class="recall">
<summary>KLで順序が重要なのはなぜか</summary>

期待値を取る分布が変わるからです。$D_{\mathrm{KL}}(p\|q)$ は $p$ の確率質量を基準に評価します。

</details>

<details class="recall">
<summary>JSで使う混合分布は何か</summary>

$m=(p+q)/2$ です。$p$ と $q$ それぞれとのKLを平均することで、対称な量にします。

</details>
