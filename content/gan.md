---
exam: E資格
concept: GAN
slug: gan
tier: A
area: 深層学習
summary: 生成器と識別器を競わせ、識別器が最適なら生成器がデータ分布との差を小さくするminimaxゲームとして学習する生成モデルです。
updated: 2026-08-22
sources:
  - title: "Generative Adversarial Networks"
    url: https://arxiv.org/abs/1406.2661
---

## ひとことで言うと

GAN（Generative Adversarial Network）は、ノイズからデータを作る生成器 $G$ と、本物か生成物かを判定する識別器 $D$ を同時に学習する枠組みです。$D$ は本物を1、生成物を0に近づけ、$G$ は $D$ をだます方向に更新されます。したがって、単独の損失を最小化するモデルではなく、二者のminimaxゲームとして読むのが出発点です。

<div class="analogy">

偽造品を作る側と鑑定士を、同じ訓練ラウンドで競わせる構図です。鑑定士が見抜けるうちは偽造側に情報が返り、偽造品が本物の分布に近づくと、鑑定士の判定は表裏を区別できない1/2へ近づきます。

</div>

## なぜ必要か

生成器だけに「よいサンプル」を直接採点するのは難しいため、判定を学ぶ $D$ を別に置き、その出力を生成器への学習信号にします。原論文の枠組みでは、生成器は入力ノイズ $\mathbf{z}$ をデータ空間のサンプル $G(\mathbf{z})$ に写し、識別器はそのサンプルがデータ分布から来た確率を出力します。学習中は $D$ を複数回更新してから $G$ を1回更新する交互更新を使います。

| 役割 | 入力 | 更新の向き |
|---|---|---|
| 生成器 $G$ | ノイズ $\mathbf{z}$ | 識別器が誤る確率を高める |
| 識別器 $D$ | 本物 $\mathbf{x}$ または $G(\mathbf{z})$ | 本物を1、生成物を0に分ける |

## 仕組み

データ分布を $p_{\text{data}}$、生成器が作る分布を $p_g$、ノイズの分布を $p_{\mathbf{z}}$ とします。原論文の価値関数は次です。$\mathbf{x}$ はデータ例、$\mathbf{z}$ はノイズ、$\theta_g$ と $\theta_d$ はそれぞれ生成器・識別器のパラメータです。

$$
\min_G\max_D V(D,G)=\mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}[\log D(\mathbf{x})]+\mathbb{E}_{\mathbf{z}\sim p_{\mathbf{z}}}[\log(1-D(G(\mathbf{z})))]
$$

$D$ はこの値を最大化し、$G$ は最小化します。$G$ を固定したとき、各データ点でこの式を最大にする識別器は

$$
D_G^*(\mathbf{x})=\frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x})+p_g(\mathbf{x})}
$$

です。つまり、$D$ の最適化はニューラルネットの出力をただ押し上げる話ではなく、二つの密度の比を判定する問題に読み替えられます。

この最適識別器を価値関数へ代入し、$C(G)=\max_D V(D,G)$ と置くと、原論文の導出は

$$
C(G)=-\log 4+2\,\mathrm{JSD}(p_{\text{data}}\Vert p_g)
$$

に到達します。Jensen–Shannon ダイバージェンスは0以上で、二つの分布が等しいときだけ0です。したがって理論上の生成器の目的は、識別器が最適なら $\mathrm{JSD}(p_{\text{data}}\Vert p_g)$ を最小化することです。大域的な最小値は $p_g=p_{\text{data}}$ で、そのとき $D_G^*(\mathbf{x})=1/2$、$C(G)=-\log 4$ になります。ただしこれは十分な容量と、各段階で識別器を最適化できるという理論条件です。実装では内側の最適化を完了させず、ミニバッチで交互に更新します。

実装上の不安定さは、初期の $G$ が悪いときに目立ちます。$D$ が生成サンプルを高い確信度で偽物と判定すると、minimax形の $\log(1-D(G(\mathbf{z})))$ が飽和し、$G$ へ戻る勾配が弱くなります。原論文はこの局面で、$G$ に同じ固定点を持つ非飽和な目的 $\max_G\log D(G(\mathbf{z}))$ を使う置き換えを示し、初期の勾配を強くできると説明しています。ここで変えているのは実用的な生成器の目的であり、minimaxゲームの理論上の均衡そのものではありません。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| GANの目的関数 | $D$ は最大化、$G$ は最小化する二つの期待値の和 | 両者が同じ方向に最大化する、とする |
| 最適識別器 | $p_{\text{data}}/(p_{\text{data}}+p_g)$ | $p_g$ を分子に置く、または和を忘れる |
| $G$ の理論上の目的 | 最適な $D$ の下でJSDを最小化する | 「識別器の損失を直接最小化」とだけ答える |
| 均衡の条件 | $p_g=p_{\text{data}}$、$D=1/2$ | $D=1$ が成功状態だとする |
| 勾配飽和への対処 | $\log(1-D(G(\mathbf{z})))$ から $\log D(G(\mathbf{z}))$ へ置換 | 後年の派生手法を原論文の提案として混ぜる |

## 実装で確かめる

最適識別器の式を、二つの離散分布でそのまま計算します。確率の並びを固定すれば、$p_g=p_{\text{data}}$ のとき各点の判定が $1/2$ になることを確認できます。

```python
import numpy as np

p_data = np.array([0.2, 0.5, 0.3])
p_g = np.array([0.1, 0.6, 0.3])
d_star = p_data / (p_data + p_g)
print("D*:", np.round(d_star, 6))

p_same = p_data.copy()
d_same = p_data / (p_data + p_same)
print("same:", d_same)
```

この計算で、識別器を学習しなくても密度が既知なら最適値が直接求まります。実際のGANでは $p_g$ を明示的な密度として扱わず、$G$ と $D$ の微分可能なネットワークを交互に更新します。

## 取り違えやすいもの

| 用語 | GANとの切り分け |
|---|---|
| 生成器 $G$ | 分布からサンプルを作る側。識別器の出力を通る勾配で更新される |
| 識別器 $D$ | 本物と生成物を判定する側。価値関数を最大化する |
| minimax目的 | 二者の理論上のゲーム全体。実装では生成器に非飽和目的を使う場合がある |
| [JSD](/learn/e-shikaku/kl-and-js-divergence/) | 最適識別器を代入したときに現れる、生成分布とデータ分布の距離 |
| VAE・拡散モデル | いずれも生成モデルですが、ここで扱うGANの競争的な目的関数とは切り分けて考えます |

## 想起チェック

<details class="recall">
<summary>GANのminimaxで最大化するのはどちらか</summary>

識別器 $D$ が $V(D,G)$ を最大化し、生成器 $G$ が最小化します。

</details>

<details class="recall">
<summary>識別器が最適なときの最適判定式は何か</summary>

$D_G^*(\mathbf{x})=p_{\text{data}}(\mathbf{x})/(p_{\text{data}}(\mathbf{x})+p_g(\mathbf{x}))$ です。

</details>

<details class="recall">
<summary>最適識別器を代入した生成器の目的は何を最小化するか</summary>

$C(G)=-\log 4+2\,\mathrm{JSD}(p_{\text{data}}\Vert p_g)$ なので、Jensen–Shannon ダイバージェンスを最小化します。

</details>

<details class="recall">
<summary>勾配飽和時に原論文が示した生成器の目的の置き換えは何か</summary>

$\log(1-D(G(\mathbf{z})))$ を最小化する代わりに、$\log D(G(\mathbf{z}))$ を最大化します。原論文の説明では固定点は同じで、学習初期の勾配が強くなります。

</details>
