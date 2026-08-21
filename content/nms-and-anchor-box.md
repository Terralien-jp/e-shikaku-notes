---
exam: E資格
concept: NMSとアンカーボックス
slug: nms-and-anchor-box
tier: B
area: 深層学習
summary: 重複する検出枠をIoUとスコアで整理するNMSと、予測の出発点として用意するアンカーボックスの設計を、実装判断につなげて整理します。
updated: 2026-08-22
sources:
  - title: "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
    url: https://arxiv.org/abs/1506.01497
  - title: "SSD: Single Shot MultiBox Detector"
    url: https://arxiv.org/abs/1512.02325
  - title: "nms"
    url: https://docs.pytorch.org/vision/stable/generated/torchvision.ops.nms.html
---

## ひとことで言うと

NMS（Non-Maximum Suppression）は、同じ物体を囲む複数の検出枠から、スコアの高い枠を残して重複枠を落とす後処理です。アンカーボックスは、モデルがいきなり任意の矩形を作るのではなく、位置ごとに用意した基準枠から大きさや位置のずれを予測するための候補枠です。

<div class="analogy">

同じ物体について届いた複数の報告を、まず信頼度順に並べ、代表者を一人だけ残すのがNMSです。アンカーボックスは、報告書を書くためにあらかじめ用意した用紙の型で、横長・縦長・大きめなど複数の型を持たせます。

</div>

## なぜ必要か

候補枠を細かく置くと、一つの物体に対して少しずつ位置の違う検出が多数出ます。これをそのまま表示すれば、同じ犬や車に何個もの矩形が重なり、検出数としても信頼できません。NMSは、候補を消す基準をスコアと重なりに分けて、この出力を一つに整理します。

アンカーボックスは、物体の位置だけでなく形の初期値を与えます。横長の物体に縦長の基準枠しかなければ、そこからの補正が大きくなり、学習・推論の両方で不利です。したがって、枠のアスペクト比とスケールは固定の飾りではなく、データに合わせて決めるハイパーパラメータです。多すぎれば予測数と計算量が増え、少なすぎれば対象形状を覆えません。

| 設計を変える | 起きること |
|---|---|
| アスペクト比を増やす | 横長・縦長など異なる形への初期適合が増えるが、候補数も増える |
| スケールを広げる | 大小の対象を覆いやすくなるが、解像度との対応を確認する必要がある |

## 仕組み

二つの矩形の交差領域を $A \cap B$、和集合を $A \cup B$ とすると、IoU（Intersection over Union）は次で定義します。

$$
\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$

IoUは重なりの割合で、0なら接触していない状態、1なら同じ矩形です。NMSでは、まずスコア $s_i$（枠 $i$ が物体らしい度合い）の降順に枠を並べ、最高スコアの枠を採用します。その枠とのIoUが閾値 $\tau$ より大きい他の枠を捨て、残った枠で同じ操作を繰り返します。

$$
\text{discard }j\quad\text{if}\quad \operatorname{IoU}(b_i,b_j)>\tau
$$

ここで $b_i,b_j$ は矩形、$\tau$ はIoU閾値です。$\tau$ が高すぎると、かなり重なっていても別枠として残るため重複検出が増えます。低すぎると、別の物体や細長い物体の近接した枠まで消し、見逃しに近い結果になります。公式APIでも、より高スコアの枠とのIoUが閾値を超える枠を破棄する仕様です。

アンカーボックス $a=(c_x,c_y,w,h)$ は中心 $(c_x,c_y)$、幅 $w$、高さ $h$ を持つ基準枠です。モデルはこの枠そのものではなく、中心のずれや幅・高さの変化を予測して、最終枠へ変換します。設計時は、対象の代表的な縦横比と大きさを枠の集合に含め、特徴マップの解像度が異なる位置にも候補を割り当てます。これは「候補枠を増やせば常によい」という話ではなく、データの物体形状と計算量の釣り合いを取る作業です。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| NMSの処理順 | スコア最大を残し、IoUが閾値を超える枠を抑制して反復 | IoUが低い枠を捨てる、または全枠を一度に平均する |
| IoU閾値の変更 | 高いほど重複を残し、低いほど抑制が強い | 高いほど抑制が強いと逆に覚える |
| アンカーボックスの意味 | 予測の基準となる事前の候補枠 | 正解矩形そのもの、または推論後の最終枠 |
| 枠の設計値 | アスペクト比とスケールは対象分布に関わるハイパーパラメータ | どのデータでも同じ比率・大きさが最適 |

## 実装で確かめる

NumPyだけで、最高スコアを残し、IoUが閾値を超える枠を順に除く最小実装を確認します。座標は $(x_1,y_1,x_2,y_2)$、スコアは枠ごとの信頼度です。

```python
import numpy as np

def nms(boxes, scores, threshold):
    order = scores.argsort()[::-1]
    kept = []
    while order.size:
        i = order[0]
        kept.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iou = inter / (area[i] + area[rest] - inter)
        order = rest[iou <= threshold]
    return np.array(kept)

boxes = np.array([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]])
scores = np.array([0.9, 0.8, 0.7])
assert np.array_equal(nms(boxes, scores, 0.5), np.array([0, 2]))
```

一つ目の枠と二つ目は大きく重なるため、スコアの低い二つ目だけが抑制されます。三つ目は重ならないので残ります。実装では、NMSに渡す枠とスコアが同じ順序で対応しているか、座標の端点を幅・高さの計算にどう含めるかを確認してください。ここがずれると、閾値を調整しても期待した抑制になりません。

## 取り違えやすいもの

| 用語 | NMS・アンカーボックスとの切り分け |
|---|---|
| [IoU](/learn/e-shikaku/iou-and-map/) | 重なりを測る指標で、NMSそのものではない |
| confidence score | 残す枠を選ぶ順位の値。重なりの大きさではない |
| アンカーボックス | 予測前の基準枠。NMSが扱う出力枠と同一とは限らない |
| default box | アンカーボックスと同じく予測の基準枠を指す呼び方 |
| region proposal | 候補領域を生成する段階。NMSは候補を絞る段階 |

## 想起チェック

<details class="recall">
<summary>NMSで最高スコアの枠を残したあと何をするか</summary>

その枠とのIoUが閾値を超える枠を捨て、残りから再び最高スコアの枠を選びます。

</details>

<details class="recall">
<summary>IoU閾値を高くしたときの出力傾向</summary>

抑制条件を満たしにくくなるため、重複枠が残りやすくなります。

</details>

<details class="recall">
<summary>アンカーボックスのアスペクト比とスケールを調整する理由</summary>

対象物の形や大きさに近い基準枠を用意し、必要な補正量と候補数のバランスを取るためです。

</details>
