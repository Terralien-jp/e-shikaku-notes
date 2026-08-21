---
exam: E資格
concept: パノプティック分割
slug: panoptic-segmentation
tier: B
area: 深層学習
summary: stuff と thing を同じ画像上で扱い、全画素にクラスとインスタンス ID を割り当てる統合分割です。
updated: 2026-08-22
sources:
  - title: "Panoptic Segmentation"
    url: https://arxiv.org/abs/1801.00868
---

## ひとことで言うと

パノプティック分割は、画像の全画素に意味クラスとインスタンス ID を割り当てる出力形式です。数えられる対象を thing、道路や空のように個体を数えない領域を stuff として、両者を一つのシーン分割に収めます。

<div class="analogy">

画像を「地図」にする作業です。道路には道路という地名だけを付け、人や車にはクラス名に加えて個体番号も振ります。地図の同じ場所を道路と車が同時に占有できないのと同じく、最終出力の画素は一つの領域にだけ属します。

</div>

## なぜ必要か

意味クラスだけでは、同じクラスの人や車を区別できません。一方、物体ごとのマスクだけを組み合わせると、道路や空のような背景領域が抜けます。パノプティック分割はこの二つを別々の評価対象にせず、stuff と thing を含む全体像として扱うために提案されました。

ただし、単純に二つのモデルの出力を重ねればよいわけではありません。物体マスク同士や物体と stuff が重なると、1画素に複数の答えが生じます。したがって、認識だけでなく、重なりを解消して一貫した非重複のシーンにする処理までが課題です。

| 入力の見方 | 欠ける情報 | 統合後に必要なもの |
|---|---|---|
| クラス中心 | 同じクラスの個体差 | thing ごとの ID |
| 個体中心 | stuff の全画素 | stuff を含むクラス割当 |
| 二つを別々に保持 | 画素の競合解決 | 一画素一セグメント |

## 仕組み

画素 $p$ の出力を $(c_p, i_p)$ とします。$c_p$ は意味クラス、$i_p$ はインスタンス ID です。同じ thing クラスでも $i_p$ が異なれば別個体であり、stuff では ID を区別しません。出力は次の制約を満たします。

$$
\text{各画素 }p\text{ は一つの }(c_p,i_p)\text{ にだけ割り当てられる}
$$

原論文のベースラインは、まずインスタンス側から非重複の thing セグメントを作り、それを semantic 側の結果と統合します。thing と stuff が同じ画素で競合したときは thing を優先し、その画素に thing のクラスと ID を残します。これは統合のためのヒューリスティックであり、二つの予測が自然に整合することを意味しません。

評価には、クラスごとの予測セグメント $p$ と正解セグメント $g$ を IoU が $0.5$ より大きいときに対応付ける PQ を使います。$TP$ は対応した組、$FP$ は余分な予測、$FN$ は見逃した正解です。

$$
\mathrm{PQ}=\underbrace{\frac{\sum_{(p,g)\in TP}\mathrm{IoU}(p,g)}{|TP|}}_{\mathrm{SQ}}\times\underbrace{\frac{|TP|}{|TP|+\frac{1}{2}|FP|+\frac{1}{2}|FN|}}_{\mathrm{RQ}}
$$

$\mathrm{SQ}$ は対応した領域の形の質、$\mathrm{RQ}$ は検出・分類の質です。PQ は stuff と thing の各クラスに同じ形で適用されますが、semantic 用 IoU と instance 用 AP を単純に足した指標ではありません。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| 出力形式 | 全画素にクラスと ID。stuff の ID は無視 | クラスだけ、または物体マスクだけ |
| thing と stuff | thing は個体を分け、stuff は領域として扱う | stuff にも物体番号を必須にする |
| 統合後の制約 | セグメントは非重複で、競合画素を整合処理する | 二つの予測をそのまま重ねる |
| PQ の意味 | PQ = SQ × RQ。IoU と認識の両面を見る | semantic IoU と instance AP の合成 |

## 実装で確かめる

小さなラベル画像で、thing の予測が stuff と重なる画素を thing 優先で置き換えます。ID はクラスと別の軸なので、同じ車クラスでも個体を区別できます。

```python
import numpy as np

stuff = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # 道路
thing_class = np.array([[0, 2, 0], [0, 2, 0], [0, 0, 0]])  # 車クラス
thing_id = np.array([[0, 7, 0], [0, 7, 0], [0, 0, 0]])
class_id = stuff.copy()
instance_id = np.zeros_like(stuff)
mask = thing_class != 0
class_id[mask], instance_id[mask] = thing_class[mask], thing_id[mask]
print(class_id.tolist())
print(instance_id.tolist())
```

thing のマスクが存在する画素では、stuff のクラス 1 が車クラス 2 に置き換わり、ID 7 が残ります。これが「全画素を埋める」と「重なりを許さない」を同じ出力に落とす最小の例です。

## 取り違えやすいもの

| 用語 | パノプティック分割との切り分け |
|---|---|
| 意味分割 | クラスを画素ごとに出すが、同一クラスの個体 ID は持たない |
| インスタンス分割 | thing の個体を分けるが、stuff を含む全画素の統合形式ではない |
| IoU | セグメントの重なりを測る量。PQ の SQ に使われるが PQ 全体ではない |
| AP | 信頼度を使う instance 評価。stuff を含む PQ とは役割が異なる |

## 想起チェック

<details class="recall">
<summary>パノプティック分割の1画素の出力は何か</summary>

意味クラスとインスタンス ID の組です。stuff では ID を区別しません。

</details>

<details class="recall">
<summary>統合時に thing と stuff が同じ画素を予測したらどうするか</summary>

原論文のベースラインでは thing を優先し、そのクラスとインスタンス ID を割り当てます。

</details>

<details class="recall">
<summary>PQ は何と何の積か</summary>

対応セグメントの形の質である SQ と、対応数から見る認識の質である RQ の積です。

</details>
