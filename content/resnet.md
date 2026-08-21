---
exam: E資格
concept: ResNet
slug: resnet
tier: A
area: 深層学習
summary: 深さを増すと訓練誤差まで悪化する劣化問題に対し、入力を恒等写像で近道させ、残差を学習することで最適化を容易にしたネットワークです。
updated: 2026-08-22
sources:
  - title: "Deep Residual Learning for Image Recognition"
    url: https://arxiv.org/abs/1512.03385
  - title: "Deep Learning: Convolutional Networks"
    url: https://www.deeplearningbook.org/contents/convnets.html
---

## ひとことで言うと

ResNet（Residual Network）は、層を単純に積み重ねる代わりに、入力をショートカットで足し戻し、残差写像を学習するネットワークです。深くしたときの問題を、正則化やデータ追加ではなく、写像の parameterization（何を学習対象にするか）の変更で扱います。

<div class="analogy">

目的地までの道を毎回ゼロから設計するのではなく、現在地からの「差分」だけを案内する道案内です。目的地が現在地に近ければ、差分を小さくすればよく、元の場所へ戻る経路はショートカットが確保します。

</div>

## なぜ必要か

深いネットワークは表現力が高いので、浅いネットワークを含む解を表現できるはずです。浅いモデルを深いモデルへ埋め込むなら、追加層を恒等写像にすれば、理屈の上では訓練誤差は悪化しません。しかし原論文の CIFAR-10 と ImageNet の比較では、単純な plain network は深くするとテスト誤差だけでなく**訓練誤差も**上がりました。これは汎化 gap の拡大で説明する過学習ではなく、最適化が良い解へ到達できない degradation problem（劣化問題）です。

ResNet は、追加層に恒等写像そのものを作らせるのではなく、入力からの差分を作らせます。恒等写像が適切な場合、非線形層を重ねて $x$ を再現するより、残差をほぼゼロへ押すほうが容易だ、というのが発想です。原論文は、深い残差ネットが最適化しやすく、深さを増して精度を得られることを実験で示しました。

| 見る点 | plain network | residual network |
|---|---|---|
| 追加層の役割 | 写像全体を作り直す | 入力からの残差を作る |
| 恒等写像が望ましい場合 | 非線形層の組合せで近似する | 残差を0へ近づける |
| 深さを増したとき | 訓練誤差が悪化し得る | 34層が18層より良くなる比較を示した |

## 仕組み

入力を $\mathbf{x}$、積み重ねた層が学習する残差写像を $\mathcal{F}(\mathbf{x},{W_i})$、出力を $\mathbf{y}$ とすると、基本形は次です。$W_i$ は残差側の学習パラメータです。

$$
\mathbf{y}=\mathcal{F}(\mathbf{x},\{W_i\})+\mathbf{x}
$$

同じ式を通常の写像 $\mathcal{H}(\mathbf{x})$ で書けば、$\mathcal{F}=\mathcal{H}-\mathbf{x}$ です。逆伝播で上流へ戻る勾配を $\partial L/partialmathbf{y}$（$L$ は損失）とすると、ショートカットを含む入力勾配は概念的に

$$
\frac{\partial L}{\partial\mathbf{x}}=\frac{\partial L}{\partial\mathbf{y}}\left(\frac{\partial\mathcal{F}}{\partial\mathbf{x}}+\mathbf{I}\right)
$$

です。$\mathbf{I}$ は恒等写像のヤコビアンです。残差側の微分が小さくても、ショートカット由来の $\mathbf{I}$ によって勾配の通り道が残ります。ただし「必ず勾配消失しない」という意味ではなく、深い層をまたぐ直接経路を追加する、という意味です。

加算には $\mathbf{x}$ と $\mathcal{F}(\mathbf{x})$ の形状が一致していなければなりません。一致する区間は identity shortcut で、原論文では追加パラメータも計算量もありません。チャネル数や空間サイズを変える箇所では、(A) identity にゼロを足して次元を増やすか、(B) $1\times1$ 畳み込みによる射影 $W_s$ で合わせます。射影を使う場合は次の形です。

$$
\mathbf{y}=\mathcal{F}(\mathbf{x},\{W_i\})+W_s\mathbf{x}
$$

空間サイズを変える箇所では stride 2 も使います。試験で「恒等写像だから常に同じ形」と短絡すると、次元変更箇所を落とします。

深い ImageNet 向けでは bottleneck（ボトルネック）を用います。$1\times1$ 畳み込みでチャネルを減らし、$3\times3$ で処理し、もう一度 $1\times1$ で戻す3層構成です。高次元のまま $3\times3$ を重ねる計算量を抑えつつ、残差側を深くできます。原論文はこの構成で ResNet-50、101、152 を作り、152層でも VGG-16/19 より少ない FLOPs と報告しています。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| 深くすると訓練誤差も増える理由 | 過学習ではなく、追加層を恒等写像にする解を最適化器が見つけにくい劣化問題 | 「テスト誤差だけが増える過学習」 |
| 残差学習の式 | $\mathcal{F}(x)+x$ のように入力を足し、差分を学習する | $\mathcal{F}(x)$ だけを出力する |
| 勾配が届く経路 | 恒等ショートカットの勾配が残差側を経由せず伝わる | 「ショートカットが勾配を遮断する」 |
| 次元が変わる箇所 | zero-padding または射影 $W_s$、空間変更なら stride も確認 | 形状不一致のまま要素加算する |
| bottleneck の順序 | $1\times1$ で減らす → $3\times3$ → $1\times1$ で戻す | $3\times3$ を高チャネルのまま3回行う |

## 実装で確かめる

残差側の線形写像と恒等ショートカットを NumPy で置き、入力勾配に「残差経路＋恒等経路」が現れることを確認します。

```python
import numpy as np
rng = np.random.default_rng(0)
x = rng.normal(size=(2, 3))
W = rng.normal(size=(3, 3))
y = x @ W + x
loss = (y ** 2).mean()
dy = 2 * y / y.size
dx = dy @ W.T + dy                 # 残差経路 + 恒等ショートカット
eps = 1e-6
num = np.zeros_like(x)
for i in range(x.size):
    xp, xm = x.copy(), x.copy()
    xp.flat[i] += eps; xm.flat[i] -= eps
    num.flat[i] = (((xp @ W + xp) ** 2).mean() - ((xm @ W + xm) ** 2).mean()) / (2 * eps)
print(loss, np.abs(dx - num).max())
```

この出力の第2値は数値微分との差です。`W` の残差側がどのような値でも、`+ dy` が恒等経路から加わる点が確認できます。実装では加算前にチャネル数と空間サイズを検査し、変わる箇所だけ射影または padding を選びます。

<div class="caution">

ResNet を「深くすれば自動的に精度が上がる仕組み」と捉えるのは誤りです。原論文が示したのは、同じ深さ・幅・パラメータ数で plain より最適化しやすく、深さの増加による訓練誤差の劣化を抑えたことです。データ、初期化、正規化などを変えた効果と混同しません。

</div>

## 取り違えやすいもの

| 用語 | ResNet との切り分け |
|---|---|
| plain network | 層を単純に積む比較対象。深くすると訓練誤差まで悪化する劣化問題が現れた |
| Highway network | ゲートで経路を制御し、ゲートにパラメータがある。ResNet の identity shortcut は常に開いていてパラメータを持たない |
| [bottleneck](/learn/e-shikaku/residual-block/) | ResNet の深い構成で計算量を抑えるブロック設計。残差学習そのものとは別の設計判断 |
| [projection shortcut](/learn/e-shikaku/residual-block/) | 次元を合わせる $W_s$。恒等ショートカットと違い、学習パラメータを追加する |
| [過学習](/learn/e-shikaku/overfit-underfit/) | 訓練誤差は下がるのにテスト誤差が上がる現象。劣化問題は訓練誤差自体が増える |

## 想起チェック

<details class="recall">
<summary>ResNet が解こうとした「劣化問題」は過学習とどう違うか</summary>

深くした plain network で訓練誤差まで増える最適化上の問題です。テスト誤差だけが増える過学習とは異なります。

</details>

<details class="recall">
<summary>残差学習で実際に学習する写像は何か</summary>

$\mathcal{F}(\mathbf{x})=\mathcal{H}(\mathbf{x})-\mathbf{x}$ です。出力は残差に入力を加えた $\mathcal{F}(\mathbf{x})+\mathbf{x}$ になります。

</details>

<details class="recall">
<summary>ショートカットが勾配伝播に与える効果は何か</summary>

恒等経路の微分である $\mathbf{I}$ が残るため、残差側の微分だけに依存しない直接経路ができます。勾配消失を数学的に常にゼロにする主張ではありません。

</details>

<details class="recall">
<summary>入力と残差の次元が違うときはどうするか</summary>

identity にゼロを追加する方法か、$1\times1$ 畳み込みの射影 $W_s$ で合わせます。空間サイズを変える箇所では stride 2 も確認します。

</details>
