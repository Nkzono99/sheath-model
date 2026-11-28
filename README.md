# sheath-model

電子を放出する平面境界（天体表面や粒子シミュレータの上端面）の上に形成される、**一次元・定常プラズマシース**を解くための Fortran ライブラリおよび Python パッケージです。

境界の電位 $\phi_H$、電場 $E_H$、粒子流束、正味電流、および高さ方向の電位・電場・密度プロファイルを計算します。

## できること・モデルの選び方

`sheath-model` には、上流の扱いと用途に応じて **3 つの機能群** があります。

| 用途 | モデルと境界条件 | 得られるもの | 対応言語 | 詳細 |
| --- | --- | --- | --- | --- |
| **準中性の上流へつなぐ標準シースを解く** | **Zhao モデル**：上流（$z \to \infty$）で $\phi = 0, E = 0$ の準中性域を仮定し、背景電子の規格化 $N_e$ を未知数として解く | Type A / B / C の平衡解・プロファイル・流束・解マップ。<br>・零電流（$J_z = 0$）<br>・境界電場指定（$E_H$ 指定）<br>・任意電位での静的評価 | **Fortran**：全機能（bin 放出スペクトル含む）<br>**Python**：$J_z = 0$（Maxwell 源・VDF 評価含む） | [物理モデル](docs/model.md) |
| **遠方の電子分布を固定して調べる** | **固定入射モデル**：遠方 $0\text{ V}$ の電子分布（複数 Maxwell・$\kappa$ 分布）の密度をそのまま固定する | 谷底での分岐判定（Type A / B の境界）、上端 $0\text{ V}$ の有限箱解、電子置換を伴う準中性外側領域付きプロファイル | **Python** | [固定入射モデル](docs/model.md#fixed-reservoir) |
| **PIC 等の粒子シミュレータと結合する** | **Matching Plane 結合**：計算領域の上端面（$z = 0$）を外側の 1 次元シースに接続する | 面の電位・電場・粒子注入／反射条件、流出電子スペクトルからの放出源構築、零電流反復・電場電流の候補選択・checkpoint | **Fortran** | [シミュレータ結合](docs/coupling.md) |

## 計算例（Type A / B / C）

電子放出の強さや入射条件により、Zhao モデルの電位プロファイルは次の 3 種類（**Type A / B / C**）に分かれます。

- **Type A**（非単調）：境界と上流の間に電位の極小（谷）$\phi_{\min} < \min(\phi_H, 0)$ を持つ
- **Type B**（単調減少）：正の境界電位 $\phi_H > 0$ から上流 $0\text{ V}$ へ単調に下がる
- **Type C**（単調増加）：負の境界電位 $\phi_H < 0$ から上流 $0\text{ V}$ へ単調に上がる

[![Type A は内部に電位の極小を持ち、Type B と C は境界から上流へ単調に変化する。上段が電位、下段が電場。](docs/figures/sheath_profiles.png)](docs/figures/sheath_profiles.pdf)

> 計算条件・CSV データ・解の Type マップは [図と計算例](docs/examples.md) を参照してください。

## クイックスタート

### Python

Python 3.10 以上が必要です（NumPy・SciPy が自動で入ります）。

```bash
python -m pip install -e .
python examples/fixed_entry.py
```

最小限のコードで上の図の Type A（太陽高度 $60^\circ$、無ドリフト）を解く例です。

```python
from sheath_model import ZhaoParams, SheathSolver

params = ZhaoParams(sun_elevation_deg=60.0, electron_drift_mode="zero")
profile = SheathSolver().solve_profile(params, branch="A")
eq = profile.equilibrium
print(f"phi_H = {eq.surface_potential_v:.3f} V, phi_min = {eq.minimum_potential_v:.3f} V")
# -> phi_H = 3.840 V, phi_min = -0.332 V
```

### Fortran

Fortran 2008（submodule 対応）コンパイラと [fpm](https://fpm.fortran-lang.org/) が必要です。外部依存はありません。

```bash
fpm run --example fixed_entry
```

自分の Fortran プロジェクトから使う場合は、`fpm.toml` に依存を追加して `use sheath_model` で呼び出します。

```toml
[dependencies]
sheath-model = { git = "https://github.com/Nkzono99/sheath-model.git", tag = "v0.2.1" }
```

> [!TIP]
> 太陽高度入力（`zhao_equilibrium_input` / `ZhaoParams`）の既定は内向き電子ドリフトあり（`normal`）です。厳密な半無限境界では正ドリフトの Type A / C は物理条件で棄却されるため、無ドリフトで解く場合は `electron_drift_mode = 'zero'` を指定してください。詳しくは [導入と最初の計算](docs/getting-started.md#最初に知っておくべき注意点) を参照してください。

## ドキュメント一覧

目的に応じて以下のドキュメントを参照してください。

| 知りたいこと | ドキュメント |
| --- | --- |
| 環境構築、最初の計算、プロジェクトへの組み込み方 | [導入と最初の計算](docs/getting-started.md) |
| 単位、座標・符号の規約、エラー処理、解の保証範囲 | [共通の規約](docs/conventions.md) |
| 物理仮定、基礎方程式、Type A/B/C、固定入射モデル、適用範囲 | [物理モデル](docs/model.md) |
| Fortran の全型・手続き・オプションの詳細仕様 | [Fortran API リファレンス](docs/fortran-api.md) |
| Python の全クラス・関数・固定入射モデル・CLI の詳細仕様 | [Python API リファレンス](docs/python-api.md) |
| 粒子シミュレータ（matching plane）との結合手順と責務 | [粒子シミュレータとの結合](docs/coupling.md) |
| 規格化、非線形探索、解マップ・continuation、検証内容 | [数値解法と検証](docs/numerics.md) |
| 模式図、Type マップの読み方、図の再生成手順、実行例一覧 | [図と計算例](docs/examples.md) |

## ライセンス

Python 実装と新規の公開窓口は [MIT](LICENSE)、BEACH 由来の数値実装は [Apache-2.0](LICENSES/Apache-2.0.txt) です。
出典と改変の概要は [NOTICE](NOTICE) を参照してください。
