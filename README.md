# sheath-model

Zhao の一次元光電子シース（Type A/B/C）を解く、独立した Fortran ライブラリです。
`fpm.toml` から外部依存なしで利用でき、公開窓口は `use sheath_model` です。

| 計算 | 関数 | 入力 → 出力 |
| --- | --- | --- |
| J=0 定常解 | `solve_equilibrium` | 太陽高度・プラズマ・光電子源 → 電位・密度・流束・電流 |
| E_H 指定解 | `solve_prescribed_field` | 法線電場・プラズマ・光電子源 → 電位・密度・流束・電流 |
| 定常解の局所密度 | `evaluate_density` | J=0 解・電位・領域 → 各粒子集団の密度 |
| 定常プロファイル | `solve_profile` | J=0 モデルの入力・積分設定 → 高さ・電位・電場・密度 |

J=0 モデルは零電流条件を課します。E_H 指定モデルは電場を境界条件として与え、電流は結果として返します。
いずれも、名前付きの入力型と結果型を使う関数呼び出しです。初期化手順や内部状態はありません。
入力・出力の単位と物理的な意味は [Fortran API](docs/fortran-api.md) を参照してください。

ソース版では、背景電子を上流 VDF から軌道保存に沿って写像し、代数根の物理プロファイルも検査します。
v0.1.0 から解の存在域が変わります。特に正の内向き電子ドリフトを持つ A/C は、
完全反射・半無限上流の仮定と両立しないため不採用です。以下の A 枝の例は無ドリフト電子を明示します。
導出、固定する量と解く量、複数解の扱いは [運動論モデル](docs/kinetic-model.md) に記載しています。

## シース解と Type の範囲

以下は **電子の法線ドリフトをゼロ** とした計算例です。
共通条件は `n_i=8.7 cm⁻³`、`T_e=12 eV`、`T_pe=2.2 eV`、陽子イオン、
`v_sw=468 km/s`、`v_i=v_sw sin(alpha)` です。`alpha` は太陽高度、上流の電位基準は 0 V です。

### Type A/B/C の 1D プロファイル

`J=0`、光電子参照密度 `n_pe,ref=64 cm⁻³` として解いた電位と電場です。
光電子源密度は `n_pe,0=n_pe,ref sin(alpha)`。横軸は表面から上流へ向かう高さで、最初の 60 m を表示しています。
各曲線は遠方の電位打ち切り `|phi|≈2.2×10⁻⁴ V` までを計算し、そこから先に人工的なゼロ電位の線は追加していません。

![Type A/B/C の J=0 シース解。上段は電位、下段は電場。A は内部極小を持ち、B/C は単調。](docs/figures/sheath_profiles.png)

| Type | 電位の形 | 太陽高度 | 表面電位 | 最小電位 |
| --- | --- | --- | --- | --- |
| A | 内部に極小を持つ非単調解 | 60° | 3.840 V | −0.332 V |
| B | 正電位から上流の 0 へ単調減少 | 20° | 1.616 V | 0 V（上流） |
| C | 負電位から上流の 0 へ単調増加 | 10° | −4.238 V | −4.238 V（表面） |

A は負の表面電位も許します。上表はそのうち正の表面電位を持つ例です。
[PDF](docs/figures/sheath_profiles.pdf) と [数値データ・解の残差](docs/figures/data/profile_metadata.csv) も利用できます。

### 解が得られた範囲と Type

縦軸は光電子源の強さ `r=n_pe,ref/n_i`（対数軸）、各パネルは **65 × 33 点** の計算です。

- 左：`J=0`。太陽高度と `r` を変え、A/B/C をそれぞれ指定して物理解を探索します。
- 右：`E_H` 指定。太陽高度を 20° に固定し、電場と `r` を変え、`solve_prescribed_field_candidates` で候補を取得します。電流は出力です。

![J=0 と E_H 指定のシース解の Type マップ。複数の Type が見つかった点は組合せの色で表示。](docs/figures/sheath_type_maps.png)

色は**この探索で得られた物理解の Type**です。A+B などは同じ条件で複数の Type が得られたことを表し、安定性による選択はしていません。
左の A/B/C マーカーは上の 1D 例の位置です。
灰色は物理条件による候補の棄却、薄灰色は数値探索が未解決で採用解が得られなかった点です。
色付きの点でも、他の枝の探索が未解決の場合があります。有限個の初期値による結果であり、灰色領域も含めて解の不存在や全根の発見を保証しません。

両パネルとも背景電子 Maxwell 分布の規格化は未知量です。電子ドリフトを正にすると A/C の半無限上流条件が成立しなくなるため、この図の範囲をそのまま適用できません。
[PDF](docs/figures/sheath_type_maps.pdf)、[J=0 の格子データ](docs/figures/data/equilibrium_map.csv)、
[E_H 指定の格子データ](docs/figures/data/field_map.csv)、[集計](docs/figures/data/summary.json) を保存しています。
CSV の Type はビット値 `A=1, B=2, C=4` の和で表し、未解決・棄却の情報も別列に記録します。

図は公開 Fortran API で再計算できます。NumPy と Matplotlib を用意し、リポジトリ直下で実行してください。
以下の GNU Fortran / OpenMP の例は 4 コアを使用します。KUDPC では計算ノード割当内で実行します。

```bash
mkdir -p docs/figures/data
OMP_NUM_THREADS=4 OMP_PROC_BIND=false fpm run --example readme_data --compiler gfortran --profile release --flag "-fopenmp" -- docs/figures/data 65 33
python examples/plot_readme_figures.py
```

計算部分は [readme_data.f90](example/readme_data.f90)、描画部分は [plot_readme_figures.py](examples/plot_readme_figures.py) です。
計算コマンドの末尾に `equilibrium` または `field` を付けると、そのカラーマップだけを再計算できます。

## Fortran / fpm

Fortran 2008 対応コンパイラと fpm が必要です。

```bash
fpm build
fpm test
fpm run --example compare_closures
fpm run --example equilibrium_profile
fpm install --prefix ./install
```

KUDPC 等の共有ログインノードでは、ビルド・テスト・例の実行をサイトの計算ノード割当内で行ってください。
別プロジェクトの `fpm.toml` に依存を追加すると、この作業ツリーを利用できます。

```toml
[dependencies]
sheath-model = { path = "../sheath-model" }
```

Git 依存でも利用できます。再現性が必要な場合は利用するコミットの `rev` を指定してください。
公開済みの `v0.1.0` タグには、以下の軌道モデル修正は含まれません。

```toml
[dependencies]
sheath-model = { git = "https://github.com/Nkzono99/sheath-model.git" }
```

E_H 指定モデルの使用例です。

```fortran
use sheath_model
implicit none
type(zhao_field_input) :: input
type(zhao_field_result) :: result
integer(i32) :: status
character(len=256) :: message

input%branch = 'A'
input%electron_drift_mps = 0.0_dp
input%root_selection = 'max_field_energy'
input%electric_field_v_m = 1.62_dp
input%photoelectron_source_density_m3 = 5.5425625842204072e7_dp
call solve_prescribed_field(input, result, status, message)
if (status /= sheath_ok) stop 1
print *, result%boundary_potential_v, result%net_current_a_m2
```

J=0 モデルも `call solve_equilibrium(input, result, status, message)` の形で呼び出します。
`max_field_energy` は正の電場エネルギーが最大の候補を選ぶヒューリスティックで、安定性の判定ではありません。
既定の `require_unique` は複数解を曖昧性として返し、`solve_prescribed_field_candidates` で候補を取得できます。
両方を比較する完全な例は [compare_closures.f90](example/compare_closures.f90) にあります。
ライブラリ自体は `stop` / `error stop` やファイル出力を行いません。

## Python

既存 Python ソルバーは Python 3.10+、NumPy、SciPy で独立して利用できます。

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m sheath_model --branch A --alpha 60 --electron-drift-mode zero --ion-drift-mode normal --zmax-hat 120
```

```python
from sheath_model import ZhaoParams, ZhaoSheathSolver

solver = ZhaoSheathSolver(ZhaoParams(alpha_deg=60.0, electron_drift_mode="zero", zmax_hat=120.0))
profile = solver.solve_profile("A")
print(profile["phi0_V"], profile["phi_m_V"], profile["n_swe_inf_m3"])
```

描画には `python -m pip install -e '.[plot]'` の後、`python examples/plot_profiles.py` を使います。
Python の局所流束・VDF 診断も同じ軌道分布に基づきます。Fortran と Python のバインディングはありません。
Python と Fortran はともに半無限領域の一次積分からプロファイルを構成します。

## 構成とライセンス

- `src/sheath_model.f90`: 公開する型・関数の一覧。
- `src/sheath_model_equilibrium.f90`: J=0 定常解、密度、プロファイル。
- `src/sheath_model_field.f90`: E_H 指定解。
- `src/internal/`: Zhao の式、積分、非線形解法、分岐選択。
- `test/`, `example/`: Fortran の物理テストと利用例。
- `sheath_model/`, `tests/`, `examples/`: Python ソルバー、テスト、描画例。
- [Algorithm notes](docs/algorithm.md): Python モデルの式と仮定。

既存 Python と新規の公開窓口は MIT。BEACH のシース数値実装から抽出した部分は Apache-2.0 です。
出典と変更点は [NOTICE](NOTICE)、ライセンス本文は [LICENSE](LICENSE) と [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt) を参照してください。
