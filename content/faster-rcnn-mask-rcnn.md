---
exam: E資格
concept: Faster R-CNNとMask R-CNN
slug: faster-rcnn-mask-rcnn
tier: A
area: 深層学習
summary: RPNで候補領域の生成を学習に組み込み、共有特徴を使って二段階検出を高速化したFaster R-CNNと、マスク分岐・RoIAlignで画素単位の位置合わせを加えたMask R-CNNを整理します。
updated: 2026-08-22
sources:
  - title: "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
    url: https://arxiv.org/abs/1506.01497
  - title: "Mask R-CNN"
    url: https://arxiv.org/abs/1703.06870
---

## ひとことで言うと

Faster R-CNNは、画像全体の畳み込み特徴を共有しながら、RPN（Region Proposal Network）で物体らしい候補領域を生成し、その候補を分類・矩形回帰する二段階検出器です。Mask R-CNNはこの構成を保ったまま、各RoIの物体マスクを予測する分岐を並列に追加し、RoIAlignで位置合わせのずれを抑えます。一段階検出のYOLOやSSDが画像から直接検出するのに対し、二段階検出は「候補を絞る段階」と「候補を精密化する段階」を分けます。

<div class="analogy">

広い倉庫を一度に検品する代わりに、まず見回り係が「この棚を詳しく見る」と候補を挙げ、専門係が棚ごとに品名と位置を確定する流れです。Mask R-CNNでは、専門係に「品物の輪郭を塗り分ける担当」も加わります。

</div>

## なぜ必要か

従来のR-CNN系では、候補領域の生成に外部の手法を使うため、検出ネットワークが速くなっても候補生成がボトルネックになります。Faster R-CNNはこの候補生成をRPNとしてネットワーク内に取り込み、候補の位置と物体らしさを学習します。RPNは全画像の畳み込み特徴を検出器と共有するため、同じ画像を二つのネットワークで別々に畳み込む無駄を避けられます。

RPNが返すのは最終検出結果ではありません。候補領域をRoIとして次段へ渡し、RoIごとにクラスと矩形のずれを推定します。IoUによる候補の評価やNMSによる重複除去はこの流れの前提であり、RPNが担う新しさは「候補生成そのものを学習する」点です。

| 方式 | 候補領域の生成 | 畳み込み特徴 |
|---|---|---|
| R-CNN系の従来構成 | 外部の候補生成 | 検出側で利用 |
| Faster R-CNN | RPNが学習して生成 | RPNと検出器で共有 |

## 仕組み

画像を共有畳み込みネットワークに通して特徴マップを作ります。その各位置に複数のアンカーを置き、RPNはアンカーごとにobjectness（物体を含む確からしさ）と、正解矩形へ移すための回帰量を出します。アンカーは異なるスケールやアスペクト比の基準矩形なので、固定解像度の特徴マップから大きさの違う物体を扱う足場になります。

RPNの学習は、分類と矩形回帰を合わせた損失として表せます。$i$をアンカーの番号、$p_i$を物体らしさ、$p_i^*$を正解ラベル、$\mathbf{t}_i$を回帰量、$\mathbf{t}_i^*$を正解回帰量とすると、典型的な形は次です。

$$
L=\frac{1}{N_{cls}}\sum_i L_{cls}(p_i,p_i^*)+\lambda\frac{1}{N_{reg}}\sum_i p_i^*L_{reg}(\mathbf{t}_i,\mathbf{t}_i^*)
$$

$N_{cls}$と$N_{reg}$はそれぞれ分類・回帰の正規化係数、$\lambda$は両者の重みです。$p_i^*$を掛けるため、矩形回帰は物体アンカーに対して効き、背景アンカーに無理な矩形を学習させません。候補はobjectnessで順位付けし、NMSで重複を減らして検出段階へ渡します。

検出段階では、共有特徴から各RoIの特徴を取り出し、分類ヘッドと矩形回帰ヘッドで最終結果を出します。速度上の要点は、RPNと検出器が同じ画像特徴を再利用することです。RPNだけを別ネットワークとして足したのではなく、候補を提案する計算と検出の計算を共有特徴の上に統合した、と捉えると構造を取り違えません。

Mask R-CNNはここにマスクヘッドを並列に加えます。RoIごとにクラス別の二値マスクを画素対画素で予測し、分類ヘッドが選んだクラスのマスクを使います。クラス分類とマスク予測を一つの画素分類に混ぜないのがポイントです。

RoIPoolはRoIを固定サイズへ切り出す際、境界やビンの座標を量子化します。矩形検出では許容できる近似でも、マスクは画素単位の輪郭を扱うため、入力と出力の対応が崩れます。RoIAlignは量子化を行わず、連続座標上のサンプリング点を双線形補間して特徴を取り出します。つまり、マスク分岐を足すだけでは不十分で、RoIごとの空間位置を保つ取り出し方まで変える必要があります。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| Faster R-CNNの二段階 | RPNで候補を作り、次段で分類・矩形回帰する | RPNが最終クラス判定まで行う |
| RPNの役割 | アンカーごとのobjectnessと矩形回帰を学習する | 外部の固定的な候補生成器と説明する |
| 高速化の理由 | RPNと検出器が全画像の畳み込み特徴を共有する | 候補数を増やせば速くなるとする |
| Mask R-CNNの追加点 | 分類・矩形回帰と並列なマスク分岐 | マスクを矩形回帰の出力と同一視する |
| RoIAlignの目的 | RoIPoolの量子化による位置ずれを避ける | 単にRoIを大きく切り出す処理とする |

## 実装で確かめる

実際の検出器は多数の畳み込み層を含みますが、共有特徴からRPNと検出ヘッドへ分岐する関係はNumPyで確認できます。ここでは特徴ベクトルを共有し、objectness・クラス・矩形回帰を別の線形ヘッドで計算します。

```python
import numpy as np

rng = np.random.default_rng(0)
feature = rng.normal(size=(4, 8))       # 4個のRoIの共有特徴
heads = {name: rng.normal(size=(8, out))
         for name, out in [("objectness", 1), ("class", 3), ("box", 4)]}
outputs = {name: feature @ weight for name, weight in heads.items()}
assert outputs["objectness"].shape == (4, 1)
assert outputs["class"].shape == (4, 3)
assert outputs["box"].shape == (4, 4)
print({name: value.shape for name, value in outputs.items()})
```

この分岐で、共有特徴を作る計算は一度、ヘッド固有の計算は目的ごとに行われます。RPNの出力が候補を作り、検出側がその候補を精密化するというデータの向きも、単一のクラス分類器として実装しない点も確認できます。

## 取り違えやすいもの

| 用語 | Faster R-CNN・Mask R-CNNとの切り分け |
|---|---|
| R-CNN | RoIごとに画像から特徴を計算する原型。共有特徴上でRoIを処理する構成とは計算の重複が違う |
| Fast R-CNN | 候補領域は外部から与えられ、共有特徴からRoIを処理する。Faster R-CNNは候補生成もRPNで学習する |
| Faster R-CNN | RPNと分類・矩形回帰の二段階。マスク分岐は標準構成に含めない |
| Mask R-CNN | Faster R-CNNにマスク分岐を加え、RoIAlignで空間位置を保つ |
| [semantic segmentation](/learn/e-shikaku/semantic-segmentation/) | クラスごとの画素分類で、同じクラスの個体を分けない。Mask R-CNNはRoI単位のインスタンスを分ける |

## 想起チェック

<details class="recall">
<summary>RPNは何を予測し、なぜアンカーを使うか</summary>

アンカーごとにobjectnessと矩形回帰量を予測します。異なる大きさ・縦横比の基準を用意することで、特徴マップ上の各位置から多様な矩形を提案できます。

</details>

<details class="recall">
<summary>Faster R-CNNが候補生成を速くできる理由は何か</summary>

RPNと検出器が画像全体の畳み込み特徴を共有し、候補生成のために別の画像畳み込みを繰り返さないからです。

</details>

<details class="recall">
<summary>Mask R-CNNでRoIAlignが必要になる理由は何か</summary>

RoIPoolの座標量子化がRoI内の画素位置をずらすためです。輪郭を画素対画素で予測するマスクでは、そのずれが問題になり、RoIAlignは量子化せず補間して位置を保ちます。

</details>

<details class="recall">
<summary>Mask R-CNNで追加された分岐は既存の何と並列か</summary>

分類分岐と矩形回帰分岐に対して、物体マスクを予測する分岐が並列に追加されます。

</details>
