---
exam: E資格
concept: L1とL2ペナルティ
slug: l1-and-l2-penalty
tier: B
area: 深層学習
summary: L1とL2の勾配・更新則の違いを、ゼロに止まる挙動と weight decay の実装まで結び付けて整理します。
updated: 2026-08-22
sources:
  - title: "Deep Learning"
    url: https://www.deeplearningbook.org/contents/regularization.html
  - title: "SGD"
    url: https://docs.pytorch.org/docs/2.13/generated/torch.optim.SGD.html
---

## ひとことで言うと

L1ペナルティは重みの絶対値を、L2ペナルティは重みの二乗を損失に加えます。同じ「重みを小さくする」処理でも、L1は小さい重みをゼロへ押し込み、L2は全体を滑らかに縮める点が実装上の分岐です。

<div class="analogy">

L1は細い枝を根元から切る剪定、L2はすべての枝を同じ比率で短くする剪定です。どちらも木を小さくしますが、残る枝の本数は同じになりません。

</div>

## なぜ必要か

損失を $J(\theta)$、パラメータを $\theta$、係数を $\lambda>0$ とすると、目的関数はデータ損失にペナルティを足した形になります。L1とL2では勾配の形が違うため、同じ学習率・同じ $\lambda$ でも更新の軌跡が変わります。

実装では、L2ペナルティを optimizer の `weight_decay` として指定することが多く、「損失に足す場合」と「optimizerが勾配に項を足す場合」を同じ更新則で読めることが必要です。

| 見る対象 | L1 | L2 |
|---|---|---|
| 更新を決める量 | 符号（ゼロでは劣勾配） | 重みそのもの |
| 典型的な結果 | ゼロが生じる | 非ゼロのまま縮む |

## 仕組み

L1とL2を次のように定義します。$\theta_i$ はパラメータの第 $i$ 成分、$J$ はデータ損失、$\lambda$ は罰則の強さです。

$$
J_{L1}(\theta)=J(\theta)+\lambda\sum_i|\theta_i|,\qquad
J_{L2}(\theta)=J(\theta)+\frac{\lambda}{2}\sum_i\theta_i^2
$$

L2では微分がそのまま $\lambda\theta_i$ になるため、通常のSGD（学習率を $\eta$、データ損失の勾配を $g_i$ とします）は次になります。

$$
\theta_i\leftarrow\theta_i-\eta(g_i+\lambda\theta_i)
=(1-\eta\lambda)\theta_i-\eta g_i
$$

データ勾配がゼロなら、毎回 $1-\eta\lambda$ 倍です。これが weight decay と呼ばれる理由です。ただし momentum や AdamW では更新の解釈が変わるため、試験ではまず「勾配に $\lambda\theta$ を加えるSGD」の条件を確認します。

L1の微分はゼロで連続ではありません。$\theta_i\ne0$ では符号関数 $\operatorname{sign}(\theta_i)$、ゼロでは劣勾配 $s_i\in[-1,1]$ を使います。

$$
\partial|\theta_i|=\begin{cases}\{+1\}&\theta_i>0\\ \{-1,1\}&\theta_i=0\\ \{-1\}&\theta_i<0\end{cases}
$$

そのためゼロ付近では、データ損失の勾配がこの範囲に収まると更新をゼロで止められます。L2は非ゼロの重みをゼロにしにくく、L1は疎なパラメータを作りやすい差があります。バイアス項は入力に依存しないオフセットで重みとは役割が違うため、通常は罰則から外します。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| L1とL2の勾配を比較 | L1は符号、L2は重みに比例 | L1の勾配を常に重みそのものとする |
| L2の更新則を読む | $g+\lambda\theta$ を学習率で更新し、勾配ゼロなら一定割合で縮む | 毎回一定量を引くとする |
| ゼロの扱い | L1はゼロで劣勾配を選び、ゼロに止まり得る | ゼロでも通常の微分値が一意にあるとする |
| 罰則の対象を選ぶ | 通例は重みだけで、バイアスを除外 | 全パラメータへ必ず同じ罰則をかける |

## 実装で確かめる

データ損失の勾配をゼロに固定し、L2の割合縮小とL1の更新をNumPyで確認します。ゼロで選ぶ劣勾配も明示します。

```python
import numpy as np

theta = np.array([2.0, -1.0, 0.0])
eta, lam = 0.1, 0.2
l2 = theta - eta * (lam * theta)
assert np.allclose(l2, (1 - eta * lam) * theta)

subgrad = np.sign(theta)
subgrad[2] = 0.0  # ゼロで選ぶ劣勾配
l1 = theta - eta * lam * subgrad
assert np.allclose(l1, [1.98, -0.98, 0.0])
print(l2, l1)
```

## 取り違えやすいもの

| 用語 | 切り分け |
|---|---|
| L1ペナルティ | 符号を使う。ゼロ解・疎性が生じやすい |
| L2ペナルティ | 重みに比例する勾配。SGDでは weight decay と同じ形で読める |
| weight decay | 実装上の呼び名。単純SGDではL2型の勾配項として現れる |
| バイアスの更新 | 入力に依存しないオフセット。通常はL1/L2の対象外 |

## 想起チェック

<details class="recall">
<summary>L2のデータ勾配がゼロのとき、1回の更新で重みはどうなるか</summary>

$\theta$ は $1-\eta\lambda$ 倍になります。一定量を引くのではなく、現在値に比例して縮みます。

</details>

<details class="recall">
<summary>L1で重みがゼロに止まり得る理由は何か</summary>

ゼロでの劣勾配が $[-1,1]$ の範囲を持つためです。データ損失側の勾配を相殺する値を選べます。

</details>

<details class="recall">
<summary>バイアスを通常は正則化しない理由は何か</summary>

バイアスは入力に依存しない全体のオフセットで、重みとは役割が違うためです。

</details>
