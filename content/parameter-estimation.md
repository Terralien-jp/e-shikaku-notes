---
exam: E資格
concept: 確率モデルの推定
slug: parameter-estimation
tier: A
area: 応用数学
summary: 観測データの尤度が最大になるようパラメータを選ぶ最尤推定を軸に、負の対数尤度＝損失・MAP推定・KLダイバージェンス最小化との等価性を整理する。
updated: 2026-08-22
sources:
  - title: "Deep Learning: Probability and Information Theory"
    url: https://www.deeplearningbook.org/contents/prob.html
  - title: "Deep Learning: Numerical Computation"
    url: https://www.deeplearningbook.org/contents/numerical.html
  - title: "Deep Learning: Machine Learning Basics"
    url: https://www.deeplearningbook.org/contents/ml.html
---

## ひとことで言うと

確率モデルの推定とは、**パラメータ $\theta$ を持つ確率分布 $p_{\text{model}}(\mathbf{x}; \theta)$ を、手元の観測データが最もそれらしく出るように決める**作業です。この「最もそれらしく」を尤度（データが観測される確率）で測り、最大化するのが最尤推定（MLE）です。

<div class="analogy">

サイコロを100回振って6が異常に多く出たとします。「歪みのないサイコロ」というモデルと「6の出目が重いサイコロ」というモデルのどちらがこの結果を出しやすいかを比べると、後者の方が圧倒的にそれらしい。最尤推定はこの比較を連続なパラメータ空間全体でやり、最も観測結果を出しやすい1点を選ぶ手続きです。

</div>

## なぜ必要か

ニューラルネットの学習は「損失関数を最小化する」と説明されますが、その損失関数の多くは最尤推定の負の対数尤度そのものです。回帰の二乗誤差も分類の交差エントロピーも、**背後にどの確率分布を仮定したかが決まれば、損失の形は導出物であって設計の余地がありません**。損失関数を天下り的に覚えるのではなく、想定した分布から自動的に決まる、という順序を押さえておくと、聞かれ方が変わっても対応できます。

| 立場 | パラメータ $\theta$ の扱い | 出力 |
|---|---|---|
| 最尤推定（MLE） | 未知だが1つの固定値 | 点推定 $\hat{\theta}$ |
| MAP推定 | 未知だが事前分布 $p(\theta)$ を持つ | 点推定 $\hat{\theta}$（事前分布で補正） |
| ベイズ推定 | 確率変数として扱う | 事後分布 $p(\theta \mid \mathcal{D})$ そのもの |

MLEとMAPは「1点を選ぶ」点推定である点で共通し、ベイズ推定は分布を丸ごと持ち越す点で異なります。この章では点推定側、特にMLEを中心に扱います。

## 仕組み

観測データ $\mathcal{D} = \{\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(n)}\}$ が独立同分布（i.i.d.）で $p_{\text{model}}(\mathbf{x}; \theta)$ から得られたと仮定すると、尤度は同時確率の積になります。

$$
\theta_{\text{ML}} = \arg\max_{\theta} \prod_{i=1}^{n} p_{\text{model}}(\mathbf{x}^{(i)}; \theta)
$$

積は $n$ が増えると桁落ちしやすく、微分も扱いにくいため、単調増加関数である $\log$ を通して和に変換します。$\log$ を取っても $\arg\max$ の位置は変わりません。

$$
\theta_{\text{ML}} = \arg\max_{\theta} \sum_{i=1}^{n} \log p_{\text{model}}(\mathbf{x}^{(i)}; \theta)
= \arg\min_{\theta} \left( -\sum_{i=1}^{n} \log p_{\text{model}}(\mathbf{x}^{(i)}; \theta) \right)
$$

右辺のカッコの中身が**負の対数尤度（NLL）**で、これがそのまま損失関数になります。ここまでは定義の言い換えです。

**KLダイバージェンス最小化との等価性**は、経験分布 $\hat{p}_{\text{data}}$（観測データそのものが定める分布）を導入すると見えます。$\hat{p}_{\text{data}}$ とモデル分布 $p_{\text{model}}$ のKLダイバージェンスは次の形です（[Deep Learning: Probability and Information Theory](https://www.deeplearningbook.org/contents/prob.html) 式3.50）。

$$
D_{\mathrm{KL}}(\hat{p}_{\text{data}} \,\|\, p_{\text{model}}) = \mathbb{E}_{\mathbf{x} \sim \hat{p}_{\text{data}}} \left[ \log \hat{p}_{\text{data}}(\mathbf{x}) - \log p_{\text{model}}(\mathbf{x}; \theta) \right]
$$

左の項 $\log \hat{p}_{\text{data}}(\mathbf{x})$ は $\theta$ を含まないため、$\theta$ に関する最小化では定数として無視できます。残るのは右の項の最小化だけで、これは交差エントロピー $H(\hat{p}_{\text{data}}, p_{\text{model}}) = -\mathbb{E}_{\mathbf{x} \sim \hat{p}_{\text{data}}}[\log p_{\text{model}}(\mathbf{x}; \theta)]$ の最小化と同じです。期待値をデータ点の平均で置き換えれば、これは負の対数尤度の平均そのもの。**「尤度を最大化する」「交差エントロピーを最小化する」「経験分布とのKLダイバージェンスを最小化する」は同じ最適化を3通りの言葉で言っているだけ**です。

**MAP推定**はここに事前分布 $p(\theta)$ を掛けます。

$$
\theta_{\text{MAP}} = \arg\max_{\theta} \left[ \log p(\mathcal{D} \mid \theta) + \log p(\theta) \right]
$$

$p(\theta)$ を一様分布とみなすと第2項は定数になり、MLEに一致します。**MLEはMAPの事前分布を「何も言わない」に固定した特殊ケース**です。L2正則化は $\theta$ にガウス事前分布を置いたMAP推定と一致することが知られており、この関係は「正則化＝事前分布」という形で試験に出やすい箇所です。

最適化の解き方は2通りに分かれます。$\nabla_\theta \sum \log p_{\text{model}} = 0$ を解析的に解ける場合（正規分布の平均・分散など）は反復なしで一発で求まります。解けない場合は、[Deep Learning: Numerical Computation](https://www.deeplearningbook.org/contents/numerical.html)（4.3節）にある最急降下法の考え方どおり、勾配の逆方向に $\theta \leftarrow \theta - \epsilon \nabla_\theta f(\theta)$ と反復します。ニューラルネットの $p_{\text{model}}$ はほぼ確実に後者です。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| 交差エントロピー損失とMLEの関係 | 交差エントロピー最小化はモデル分布とのKLダイバージェンス最小化＝負の対数尤度最小化と等価 | 「交差エントロピーとMLEは別の原理」という記述 |
| MAPとMLEの違い | MAPは事前分布 $p(\theta)$ を掛ける。事前分布が一様ならMLEに一致 | 「MAPはベイズ推定そのもの」（MAPは点推定、ベイズ推定は事後分布） |
| 正則化と事前分布の対応 | L2正則化はガウス事前分布を置いたMAP推定に対応 | L1とガウス事前分布を結びつける誤り（L1はラプラス分布） |
| i.i.d.仮定の役割 | 尤度を各サンプルの確率の**積**として書くために必要 | 「独立でなくても積で書ける」という誤り |
| 対数を取る理由 | 積を和に変換し数値的に安定させるため。$\arg\max$ の位置は変わらない | 「対数を取ると最適解が変わる」という誤り |

## 実装で確かめる

正規分布のパラメータには解析解（標本平均・標本分散）があります。これと、対数尤度を直接勾配降下法で最大化した結果が一致することを確認します。

```python
import numpy as np
rng = np.random.default_rng(0)
x = rng.normal(loc=3.0, scale=2.0, size=200)

# 解析解（正規分布のMLE。分散は n で割る）
mu_hat = x.mean()
var_hat = ((x - mu_hat) ** 2).mean()

# 勾配法：平均対数尤度を直接最大化する
mu, log_s = 0.0, 0.0                       # log_s は log(sigma^2)
for _ in range(3000):
    s = np.exp(log_s)
    grad_mu = -np.mean(x - mu) / s
    grad_logs = 0.5 - 0.5 * np.mean((x - mu) ** 2) / s
    mu -= 0.1 * grad_mu
    log_s -= 0.1 * grad_logs

print("解析解:", mu_hat, var_hat)
print("勾配法:", mu, np.exp(log_s))
```

実行すると次の通りです。

```
解析解: 3.030526279319881 3.695459269120491
勾配法: 3.030526279319873 3.695459269120498
```

両者は小数第9位まで一致します。$\sigma^2$ を `log_s` 経由で最適化しているのは、生の $\sigma^2$ をそのまま更新すると勾配ステップで負の値に落ちてしまい、確率分布として無効になるためです。指数変換で正値制約を外すのは実装上の定番の手当てです。

## 取り違えやすいもの

| 用語 | 確率モデルの推定との関係 |
|---|---|
| 最小二乗法 | 出力に等分散ガウスノイズを仮定した回帰では、二乗誤差の最小化はMLEと数学的に一致する。ガウス以外を仮定すると一致しない |
| ベイズ推定 | 事後分布 $p(\theta \mid \mathcal{D})$ をまるごと持つ。MAPは事後分布の最頻値1点だけを取る近似にあたる |
| モーメント法 | 標本の平均・分散などモーメントを分布の理論値に一致させて解く。尤度を使わない別の推定原理で、正規分布では結果がMLEと一致するが一般には一致しない |
| L2正則化 | ガウス事前分布を置いたMAP推定と一致する。正則化項は事前分布の対数に相当する |
| KLダイバージェンス | $D_{\mathrm{KL}}(\hat{p}_{\text{data}} \| p_{\text{model}})$ の最小化はMLEと同値。ただし対称ではないため「距離」ではない |

## 想起チェック

<details class="recall">
<summary>尤度の積に対数を取って和に変換してよい理由は何か</summary>

$\log$ は単調増加関数なので、$\log$ を通しても $\arg\max$ の位置は変わりません。積を和に変えることで数値的な桁落ちを避け、微分も扱いやすくなります。

</details>

<details class="recall">
<summary>「交差エントロピー損失の最小化」と「最尤推定」はどういう関係か</summary>

同じ最適化を別の言葉で表したものです。経験分布とモデル分布のKLダイバージェンス最小化のうち、経験分布側の項は $\theta$ に依存しないため落ちて、残るのが交差エントロピーの最小化＝負の対数尤度の最小化になります。

</details>

<details class="recall">
<summary>MAP推定がMLEに一致するのはどんな条件か</summary>

事前分布 $p(\theta)$ を一様分布とみなせる場合です。このとき $\log p(\theta)$ が定数になり、$\arg\max$ に影響しなくなります。

</details>

<details class="recall">
<summary>正規分布のMLEを解析的に解ける場合と、ニューラルネットで解析解が使えない場合の違いは何か</summary>

$\nabla_\theta \sum \log p_{\text{model}} = 0$ が代数的に解ける分布では一発で解が求まります。ニューラルネットのように $p_{\text{model}}$ が複雑な非線形関数の場合はこの式が解析的に解けないため、勾配降下法などの反復法に頼ります。

</details>
