# 導入と最初の計算

このページでは、Fortran ライブラリおよび Python パッケージの導入手順、最初の計算の動かし方、そして使い始めにつまずきやすい注意点を説明します。

- 単位・座標・符号・エラー処理の基本は [共通の規約](conventions.md) を参照してください。
- 全 API の仕様は [Fortran API](fortran-api.md) / [Python API](python-api.md) にまとめています。

---

## Fortran で始める

### 必要環境

- Fortran 2008（`submodule` を含む）に対応したコンパイラ（GNU Fortran など）
- [fpm](https://fpm.fortran-lang.org/)（Fortran Package Manager）
- 外部ライブラリへの依存はありません。

### 最初の計算を実行する

リポジトリ直下で、付属の実行例 `fixed_entry` を動かします。

```bash
fpm run --example fixed_entry
```

温かいイオン（$T_i = 12\text{ eV}$、圧力係数 $3$）と固定した入口速度の条件で、零電流（$J_z = 0$）の Type A を解き、次のような出力を得ます。

```text
Surface potential [V]:     3.84015185
Minimum potential [V]:    -0.33170306
Surface field [V/m]:   1.623755E+00
Inward ion flux [m^-2 s^-1]:   3.526109E+12
```

ソースコードは [example/fixed_entry.f90](../example/fixed_entry.f90) にあります。基本的な呼び出しの流れは次の 3 ステップです。

1. **物理入力を作る**（例：`zhao_equilibrium_input` または `fixed_entry_equilibrium_input`）
2. **`sheath_solver` の求解メソッドを呼ぶ**（例：`solver%solve_equilibrium`）
3. **`status == SHEATH_OK` を確認してから結果を読む**

最小限のコード例（太陽高度 $60^\circ$・無ドリフトの Type A）は次のとおりです。

```fortran
program quickstart
  use sheath_model
  implicit none
  type(sheath_solver) :: solver
  type(zhao_equilibrium_input) :: input
  type(sheath_equilibrium_result) :: res
  integer(i32) :: status
  character(len=256) :: message

  input%sun_elevation_deg = 60.0_dp
  input%electron_drift_mode = 'zero'
  input%branch = 'A'

  call solver%solve_equilibrium(input, res, status, message)
  if (status /= SHEATH_OK) then
    print '(a)', trim(message)
    stop 1
  end if

  print '(a,f8.3,a)', 'Surface potential: ', res%surface_potential_v, ' V'
  print '(a,f8.3,a)', 'Minimum potential: ', res%minimum_potential_v, ' V'
end program quickstart
```

高さ方向の電位・電場・電荷密度プロファイルを CSV 出力する例は [example/equilibrium_profile.f90](../example/equilibrium_profile.f90)（`fpm run --example equilibrium_profile`）にあります。

### 自分の Fortran プロジェクトに追加する

利用側のコードでは `use sheath_model` を使います。

#### fpm プロジェクトの場合

`fpm.toml` でリリースタグ（`tag`）またはコミット（`rev`）を指定します。

```toml
[dependencies]
sheath-model = { git = "https://github.com/Nkzono99/sheath-model.git", tag = "v0.2.1" }
```

> [!NOTE]
> タグ以降に追加された機能（例：[電場と電流を合わせる候補選択](coupling.md#combined-matching)）を使う場合は、その機能を含むコミットを `rev = "..."` で指定するか、新しいリリースタグを使用してください。

ローカルに配置したリポジトリを参照する場合は `path` を使います。

```toml
[dependencies]
sheath-model = { path = "../sheath-model" }
```

#### Make / CMake からリンクする場合

利用側と同じコンパイラ・同じバージョンで、静的ライブラリと `.mod` ファイルをインストールしてからリンクします（以下は GNU Fortran の例です）。

```bash
fpm install --profile release --compiler gfortran --prefix /path/to/sheath-model-install
gfortran -I/path/to/sheath-model-install/include -c host.f90
gfortran host.o /path/to/sheath-model-install/lib/libsheath-model.a -o host
```

---

## Python で始める

### 必要環境とインストール

- Python 3.10 以上
- 依存パッケージ（NumPy・SciPy）はインストール時に自動で入ります。
- Python パッケージは Fortran 実装とは独立した純 Python（NumPy / SciPy）実装です。

リポジトリ直下でインストールし、Fortran 版と同じ条件を解く例を実行します（[examples/fixed_entry.py](../examples/fixed_entry.py)）。

```bash
python -m pip install -e .
python examples/fixed_entry.py
```

出力は Fortran の `fixed_entry` と一致します（表面電位 `3.84015185 V`、極小電位 `-0.33170306 V`）。

図の描画スクリプト（`examples/plot_*.py`）を使う場合は、`plot` オプション付きで Matplotlib も導入します。

```bash
python -m pip install -e '.[plot]'
```

### 最小限のコード例

太陽高度入力（`ZhaoParams`）でプロファイルと表面での粒子流束を計算する例です。

```python
from sheath_model import ZhaoParams, SheathSolver

params = ZhaoParams(sun_elevation_deg=60.0, electron_drift_mode="zero")
solver = SheathSolver()
profile = solver.solve_profile(params, branch="A")

eq = profile.equilibrium
fluxes = profile.fluxes(0.0)
print(f"phi_H   = {eq.surface_potential_v:.4f} V")
print(f"phi_min = {eq.minimum_potential_v:.4f} V (at z = {profile.turning_height_m:.2f} m)")
print(f"E_H     = {profile.electric_field_v_m[0]:.4f} V/m")
print(f"J_net   = {fluxes['J_net_Apm2']:.3e} A/m^2")
```

- 入口速度や放出源を直接指定したい場合は [`FixedEntryParams`](python-api.md#fixedentryparams) を使います。
- 遠方の電子分布を固定したい場合は [固定入射モデルの計算例](python-api.md#fixed-reservoir) を参照してください。
- コマンドラインから手軽に試すこともできます：

  ```bash
  python -m sheath_model --branch A --alpha 60 --electron-drift-mode zero --ion-drift-mode normal
  ```

---

<a id="最初に知っておくべき注意点"></a>

## 最初に知っておくべき注意点

> [!IMPORTANT]
> **背景電子のドリフトと Type A / C の棄却**
>
> `zhao_equilibrium_input`（Fortran）および `ZhaoParams`（Python）の既定値は `electron_drift_mode = 'normal'`（正の内向きドリフト $u > 0$）です。
>
> しかし、厳密な半無限境界（$\phi = 0, E = 0$）のもとでは、正ドリフトを持つ背景電子と完全反射を組み合わせた **Type A と Type C は上流近傍で $E^2 < 0$ となるため、既定設定では物理条件により棄却されます**（`branch = 'auto'` では Type B が返り、`branch = 'A'` や `'C'` を明示するとエラーになります）。
>
> 用途に応じて次のどちらかを選択してください。
> - **無ドリフトの背景電子として解く**（標準的な比較や代表プロファイル）：`electron_drift_mode = 'zero'`（`plasma_input` / `FixedEntryParams` では既定でドリフト 0）を指定します。
> - **正ドリフトを保ったまま A / C を解く**：上流の負 $E^2$ 帯の許容条件（`upstream_band_tolerance` または `upstream_field_floor_v_m`）を設定します。物理的な意味は [ドリフトのある A/C と上流の帯](model.md#ドリフトのある-ac-と上流の帯) を参照してください。

---

## テストを実行する

導入後や変更後に、Fortran・Python のテストスイートを実行して動作を確認できます。

```bash
fpm test
python -m unittest discover -s tests -v
```

テストで検証している項目とコンパイラ別の確認記録は [数値解法と検証](numerics.md#検証) にあります。

## 次に読むドキュメント

- 単位・座標・符号・エラー処理を確認する → [共通の規約](conventions.md)
- 物理仮定と方程式を理解する → [物理モデル](model.md)
- 実行可能なサンプルコードを探す → [実行例の一覧](examples.md#実行例の一覧)
