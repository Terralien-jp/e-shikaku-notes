---
exam: E資格
concept: A3C
slug: a3c
tier: B
area: 深層学習
summary: 複数の環境を非同期に走らせ、経験再生に頼らず相関を弱めながら方策と価値関数を同時に学習する手法です。
updated: 2026-08-22
sources:
  - title: "Asynchronous Methods for Deep Reinforcement Learning"
    url: https://arxiv.org/abs/1602.01783
---

## ひとことで言うと

A3C（Asynchronous Advantage Actor-Critic）は、複数の actor-learner がそれぞれ環境を進め、短い軌跡から得た勾配を共有パラメータへ非同期に反映する強化学習法です。Actor は方策、Critic は状態価値を学び、両者の差である advantage を方策勾配の重み付けに使います。

<div class="analogy">

同じ課題を複数の作業員が別々の机で試し、各自の途中結果を中央の設計図へ順次書き戻すイメージです。机ごとに試行の順序がずれるため、同じ経験を保存して混ぜ直さなくても、更新に入るデータの相関が弱くなります。

</div>

## なぜ必要か

DQN 系は経験再生に保存した遷移をランダムに取り出しますが、メモリと追加計算が必要で、古い方策のデータを扱う off-policy 学習が前提です。A3C は複数環境の並列実行でこれを置き換えます。worker ごとに環境が異なるため、勾配の相関が下がり、学習を安定化します。

| 観点 | A3C | DQN 系の典型 |
|---|---|---|
| データの扱い | 複数 worker の新しい軌跡 | replay memory から再標本化 |
| 主な学習対象 | 方策と状態価値 | 行動価値 $Q$ |
| 並列化の役割 | 相関を弱め、更新を進める | 主にデータや計算を拡張 |

原論文は、複数の標準的な強化学習法を非同期化して検証し、その中で非同期 actor-critic が最も良い結果を示したと報告しています。また、Atari の実験を単一のマルチコアCPUでGPUなしに学習したことも主張しています。

## 仕組み

方策を $\pi(a_t\mid s_t;\theta)$、そのパラメータを $\theta$、状態価値を $V(s_t;\theta_v)$、価値側のパラメータを $\theta_v$ とします。worker は $t$ から最大 $n$ step 進み、終端なら0、未終端なら最後の状態価値で bootstrap した n-step return を作ります。

$$
R_t = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1}r_{t+n-1} + \gamma^n V(s_{t+n};\theta_v)
$$

ここで $r_t$ は時刻 $t$ の報酬、$\gamma$ は割引率です。Critic の誤差は $R_t-V(s_t;\theta_v)$ で、Actor に渡す advantage の推定値にもなります。

$$
A_t \approx R_t - V(s_t;\theta_v)
$$

$A_t$ が正なら、その行動は現在の価値予測より良いので選択確率を増やし、負なら減らします。方策側の勾配は次の形です。

$$
\nabla_\theta \log\pi(a_t\mid s_t;\theta)\,A_t
$$

worker はローカルに軌跡と勾配を蓄積し、共有パラメータへ非同期に更新します。別 worker が共有値を書き換えるため、勾配は厳密に同じパラメータから計算されません。この非同期更新が、経験再生なしで多様なサンプルを流し込む設計です。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| A3C の名前の意味 | Asynchronous、Advantage、Actor-Critic の対応を押さえる | A を Agent や Action と読む |
| advantage の役割 | return と状態価値の差で方策勾配を重み付けする | 行動価値そのもの、または報酬そのものとする |
| 経験再生との関係 | 複数環境の並列性でサンプル相関を弱める | replay memory を必須部品とする |
| worker の更新 | ローカル軌跡から勾配を作り共有パラメータへ非同期反映する | 全workerを同期バリアで待たせる |

## 実装で確かめる

次の例は、軌跡から n-step return と advantage を計算します。A3Cでは、この値から方策・価値の勾配を作り、worker で共有モデルを更新します。

```python
import numpy as np

rewards = np.array([0.0, 1.0, 2.0])
values = np.array([0.4, 0.5, 0.6, 0.0])  # 最後は終端のbootstrap値
gamma = 0.9
n = 3
R = values[-1]
returns = np.zeros(n)
for t in range(n - 1, -1, -1):
    R = rewards[t] + gamma * R
    returns[t] = R
advantages = returns - values[:n]
assert np.allclose(returns, [2.52, 2.8, 2.0])
assert np.allclose(advantages, [2.12, 2.3, 1.4])
```

## 取り違えやすいもの

| 用語 | A3Cとの切り分け |
|---|---|
| Actor-Critic | A3Cの土台。A3Cはそこへ非同期 worker と advantage を組み合わせたもの |
| A2C | Actor-Critic の更新を同期してまとめる実装上の対比。非同期性がA3Cとの焦点 |
| [DQN](/learn/e-shikaku/dqn/) | 行動価値を学び、経験再生を使う代表例。A3Cは方策を直接更新する |
| n-step 法 | A3Cがreturnを作る時間方向の方法。A3Cという並列更新機構そのものではない |

## 想起チェック

<details class="recall">
<summary>A3Cで経験再生の代わりになる設計は何か</summary>

複数の worker が別々の環境を非同期に進め、相関の異なる軌跡から更新します。再生メモリからのランダム抽出ではありません。

</details>

<details class="recall">
<summary>advantageは何と何の差か</summary>

return の推定値 $R_t$ と、Critic が出す状態価値 $V(s_t;\theta_v)$ の差です。方策勾配の大きさと向きを行動の良し悪しに合わせます。

</details>

<details class="recall">
<summary>A3Cの非同期性は何を共有するか</summary>

worker が計算した勾配を共有パラメータへ反映します。worker同士が同じ時刻の軌跡を同期してから更新する方式ではありません。

</details>
