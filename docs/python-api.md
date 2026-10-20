# Python API リファレンス

Python パッケージ `sheath_model` は、Maxwell 放出源の J=0 モデルを NumPy/SciPy で実装した独立版です。
Fortran のバインディングではありません。E_H 指定、静的評価、bin スペクトル、シミュレータ結合は [Fortran API](fortran-api.md) にあります。
分布関数・単位・判定条件は Fortran と揃えており、プロファイルに加えて局所の速度分布と流束を評価できます。
単位と符号は [共通の規約](conventions.md) に従います。

```python
from sheath_model import FixedEntryParams, SheathSolver, SearchOptions, ProfileOptions

inputs = FixedEntryParams(
    ion_density_m3=8.7e6, ion_entry_speed_mps=405299.88897111727,
    ion_temperature_ev=12., ion_pressure_factor=3.,
    electron_temperature_ev=12., electron_drift_mps=0.,
    emission_density_m3=55.42562584220407e6, emission_temperature_ev=2.2,
)
solver = SheathSolver(search=SearchOptions(method="newton"), profile=ProfileOptions(zmax_hat=120.))
root = solver.solve_equilibrium(inputs, branch="A")
profile = solver.build_profile(root)
print(root.surface_potential_v, root.minimum_potential_v)
print(profile.fluxes(0.)["J_net_Apm2"])
```

入力、solver、結果、プロファイルは変更できない dataclass です。値を変えるときは `dataclasses.replace` を使います。
solver は前回の解を保持せず、別の物理入力にも再利用できます。

## 物理入力

### `FixedEntryParams`

| 属性 | 既定値 | 意味 |
| --- | --- | --- |
| `ion_density_m3` | 8.7e6 | 上流イオン密度 |
| `ion_entry_speed_mps` | 405299.88897111727 | 内向き法線入口速度 |
| `ion_temperature_ev`, `ion_pressure_factor` | 0, 1 | イオン温度と圧力項の係数 |
| `electron_temperature_ev` | 12 | 背景電子温度 |
| `electron_drift_mps` | 0 | 背景電子の内向き法線ドリフト |
| `emission_density_m3`, `emission_temperature_ev` | 0, 2.2 | 放出電子の Maxwell 分布の規格化（外向き半分の密度はこの半分）と温度 |
| `ion_mass_kg`, `electron_mass_kg` | 陽子・電子質量 | |

### `ZhaoParams`

太陽高度と全風速から法線条件と光電子源を作ります（[変換規則](model.md#太陽風と照射角からの入力)）。
属性と既定値は Fortran の `zhao_equilibrium_input` と同じです（`sun_elevation_deg=60`、`photoelectron_reference_density_m3=64e6`、
`solar_wind_speed_mps=468e3`、`electron_drift_mode="normal"`、`ion_drift_mode="normal"` など）。
導出した `electron_drift_mps`、`ion_entry_speed_mps`、`emission_density_m3`、`emission_temperature_ev` を読み取り専用の属性で参照できます。

## `SheathSolver`

`SheathSolver(search=SearchOptions(), profile=ProfileOptions(), continuation=ContinuationOptions(), equilibrium_atlas=None)`

| メソッド | 戻り値 | 動作 |
| --- | --- | --- |
| `solve_equilibrium(inputs, *, branch="auto", initial_guess=None)` | `EquilibriumResult` | 指定 Type、または auto の順で最初の物理解。`initial_guess` は以前の結果で、同じ Type なら初期値に加える |
| `solve_equilibrium_candidates(inputs, *, branch="auto", deflation=True, max_roots=16)` | `CandidateSet` | 見つかった物理解の集合（`candidates`、`diagnostics`）。`max_roots` は Type ごと |
| `build_profile(root)` | `SheathProfile` | 探索せず、root が持つ入力で検査してから再構成 |
| `solve_profile(inputs, *, branch="auto", initial_guess=None)` | `SheathProfile` | 求解と再構成 |
| `build_equilibrium_atlas(parameters, *, branches=("A","B","C"), options=None, deflation=False)` | `EquilibriumAtlas` | 新しい解マップを作る。solver に付けた表は変えない |

auto の順は、`ZhaoParams` の太陽高度 20° 未満で C→A→B、それ以外 A→B→C です。
探索に失敗すると `SearchFailure` を送出し、`exception.diagnostics` に A/B/C 順の診断が残ります。

### `SearchOptions`

Fortran の `sheath_search_options` と同じ意味です（`scalar_max_depth` は E_H 指定用なのでありません）。
`method` は `"auto"`、`"newton"`、`"lm"`、`"bracket"`（B/C を明示したときだけ）、
既定値は `residual_tolerance=1e-10`、`max_iterations=100`、`max_backtracks=24`、`max_starts=32`、`use_default_guesses=True`、
`bracket_points=96`、`potential_extent=200`、`upstream_band_tolerance=0`、`upstream_field_floor_v_m=0` です。

### `ProfileOptions`

位置と電位は無次元で指定します。長さの尺度は `inputs.length_scale_m`、電位の尺度は放出電子温度です。

| 属性 | 既定値 | 意味 |
| --- | --- | --- |
| `zmax_hat` | 80 | 返す高さの上限（長さ尺度の倍数） |
| `n_profile_grid` | 600 | B/C の電位点数。32 以上で、実際には 600 点以上を使う |
| `n_type_a_grid` | 8000 | A の上流側の電位点数。32 以上で、実際には 2000 点以上を使う |
| `profile_phi_tol_hat` | 1e-3 | 上流側の打ち切り電位 \|φ\|。B/C は \|φ_H\| の半分を上限とする |
| `type_a_phi_m_eps_hat` | 1e-5 | A の格子を極小からずらして始める幅。\|φ_m\| の 5% を上限とする |
| `upstream_band_margin` | 0.05 | 受理した上流の帯（E²<0）がある根で、帯の手前に残す余白。帯の端から極小（B/C は境界電位）までの深さに対する割合。0 より大きく 1 未満 |

### `ContinuationOptions`

Fortran と同じ意味と既定値です（`method="parameter"` または `"arclength"`、`initial_step=0.25`、`min_step=1e-4`、`max_step=0.5`、
`max_steps=128`、`max_root_distance=0.75`）。

## 結果

### `EquilibriumResult`

| 属性・メソッド | 意味 |
| --- | --- |
| `inputs`, `branch` | 解いた入力と Type |
| `surface_potential_v`, `minimum_potential_v` | φ_H、φ_min [V] |
| `ambient_electron_density_m3` | 背景電子の規格化 N_e（上流の全電子密度ではない） |
| `residual_norm`, `diagnostics` | 規格化残差、`SearchDiagnostics` |
| `upstream_negative_band_v`, `upstream_field_deficit_v_m` | 受理した上流の帯 [V] と場の不足 [V/m] |
| `density(potential_v, *, side="upper")` | 局所密度の `DensityResult`。A は `side="lower"` または `"upper"` |

`DensityResult` の属性は `ion_m3`、`electron_free_m3`、`electron_reflected_m3`、`emitted_free_m3`、`emitted_captured_m3` と、
性質 `charge_c_m3` です。

### `SheathProfile`

| 属性・メソッド | 意味 |
| --- | --- |
| `equilibrium` | 元の `EquilibriumResult` |
| `z_m`, `z_hat`, `potential_v`, `electric_field_v_m` | 境界から上流への読み取り専用配列。間隔は非一様 |
| `density` | 配列を持つ `DensityResult` |
| `turning_height_m` | A の極小の高さ [m]。B/C は −1 |
| `sample(z, *, unit="m")` | 位置 z の局所状態（電位、電場、各集団の密度、到達の下限速度など）の dict |
| `fluxes(z, *, unit="m")` | 位置 z の各集団の数流束・電流の dict。表示格子ではなく分布の流束積分で評価する |
| `vdf(z, *, species="all", unit="m", n_v=4001, n_sigma=6., ion_sigma_frac=.03)` | 位置 z の一次元速度分布の dict |

位置は既定で m です。無次元位置は `unit="hat"` を明示します。計算した区間の外は `ValueError` です。

`fluxes` の `*_signed_m2s`・`*_Apm2` は +z 向きを正とする値で、例えば境界へ入るイオンの数流束は `-fluxes["Gamma_swi_signed_m2s"]`、
正味電流は `fluxes["J_net_Apm2"]` です。集団の名前は `swe`（背景電子）、`swi`（イオン）、`phe`（放出電子）です。

`vdf` の `species` は `"swe"`、`"phe"`、`"swi"`、`"all"`、`n_v` は 201 以上です。
各集団の `vz_mps` と `g_total`、成分別の分布（背景電子は `g_free_incoming`・`g_reflected_incoming`・`g_reflected_outgoing`、
放出電子は `g_free_outgoing`・`g_captured_outgoing`・`g_captured_returning`）を返します。
イオンは冷たいビームを幅 `ion_sigma_frac` の Gauss 分布で表示用に正則化したもので、温かいイオンでは定義されず `ValueError` です。
温かいイオンでは `species="swe"` または `"phe"` を指定してください。

## 解マップ

```python
from dataclasses import replace
from sheath_model import EquilibriumAtlas, ContinuationOptions

sweep = [replace(inputs, emission_density_m3=value) for value in (54e6, 57e6)]
atlas = solver.build_equilibrium_atlas(sweep, branches=("A",))
atlas.save("equilibrium-atlas.txt")
worker = replace(solver, equilibrium_atlas=EquilibriumAtlas.load("equilibrium-atlas.txt"),
                 continuation=ContinuationOptions(method="arclength"))
query = worker.solve_equilibrium(inputs, branch="A")
found = worker.solve_equilibrium_candidates(inputs, branch="A")
print(len(found.candidates), found.diagnostics.atlas_hits)
```

| 名前 | 意味 |
| --- | --- |
| `EquilibriumAtlas(points=(), *, options=None)` | 呼び出し側所有の表。`points` は `AtlasPoint` のタプル |
| `atlas.add(result, *, component=None, search=None)` | 元の方程式と判定条件で検査して登録する。求解による自動登録はない |
| `atlas.save(path)`, `EquilibriumAtlas.load(path, *, options=None)` | Fortran の J=0 表と共通のテキスト形式 |
| `atlas.attempts` | `build_equilibrium_atlas` の結果。入力・Type ごとに `accepted`/`excluded`/`unresolved` と診断を記録する |
| `AtlasOptions` | `neighbors=8`、`max_distance=1.`、`interpolate=True`、`component_distance=.75` |

表は初期値と continuation に使い、問い合わせでは毎回元の方程式を解き直します。空白の領域は解の不存在を意味しません。
Fortran が書いた bin スペクトルの記録も読めますが、Maxwell 源の問い合わせには使いません。

## 補助関数とコマンドライン

`ion_density_ratio(potential_v, entry_energy_ev, pressure_energy_ev=0.)` と `ion_critical_potential(entry_energy_ev, pressure_energy_ev=0.)` は
Fortran と同じ量を返し、範囲外では `ValueError` を送出します。

`python -m sheath_model` は `ZhaoParams` の J=0 解を表示します。

```bash
python -m sheath_model --branch A --alpha 60 --electron-drift-mode zero --ion-drift-mode normal --zmax-hat 120
```

コマンドラインのドリフトの既定値は `full` で、`ZhaoParams` の既定値（`normal`）と異なります。
ドリフトのある A/C は棄却されるので（[物理モデル](model.md#ドリフトのある-ac-と上流の帯)）、通常は `--electron-drift-mode zero` を指定してください。
