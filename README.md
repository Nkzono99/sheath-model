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

リリース版は Git タグを指定して利用できます。

```toml
[dependencies]
sheath-model = { git = "https://github.com/Nkzono99/sheath-model.git", tag = "v0.1.0" }
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
input%root_selection = 'minimum_energy'
input%electric_field_v_m = 1.62_dp
input%photoelectron_source_density_m3 = 5.5425625842204072e7_dp
call solve_prescribed_field(input, result, status, message)
if (status /= sheath_ok) stop 1
print *, result%boundary_potential_v, result%net_current_a_m2
```

J=0 モデルも `call solve_equilibrium(input, result, status, message)` の形で呼び出します。
両方を比較する完全な例は [compare_closures.f90](example/compare_closures.f90) にあります。
ライブラリ自体は `stop` / `error stop` やファイル出力を行いません。

## Python

既存 Python ソルバーは Python 3.10+、NumPy、SciPy で独立して利用できます。

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m sheath_model --branch A --alpha 60 --zmax-hat 120
```

```python
from sheath_model import ZhaoParams, ZhaoSheathSolver

solver = ZhaoSheathSolver(ZhaoParams(alpha_deg=60.0, zmax_hat=120.0))
profile = solver.solve_profile("A")
print(profile["phi0_V"], profile["phi_m_V"], profile["n_swe_inf_m3"])
```

描画には `python -m pip install -e '.[plot]'` の後、`python examples/plot_profiles.py` を使います。
Python の局所流束・VDF 診断は引き続き Python API で利用します。Fortran と Python のバインディングはありません。
Python の B/C プロファイルは有限区間 BVP、Fortran は半無限領域の一次積分です。

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
