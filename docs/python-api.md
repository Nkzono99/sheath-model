# Python API リファレンス

Python パッケージ `sheath_model` は NumPy/SciPy で実装した二つの閉じ方を提供します。
`SheathSolver` は Zhao の J=0 モデルを解き、プロファイルと局所の速度分布を返します。
[遠方の電子分布を固定するモデル](#fixed-reservoir) は、谷の診断と有限箱・外側領域付きのプロファイルを提供します。
Zhao モデルの E_H 指定・静的評価・bin 放出源とシミュレータ結合は [Fortran API](fortran-api.md) にあります。
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

<a id="fixed-reservoir"></a>

## 遠方の電子分布を固定するモデル

`FixedReservoir` では、0 V のリザーバーにおける法線電子分布の全密度を入力します。
有限箱では 0 V の上端からの入射分を使い、外側領域付きのモデルでは遠方の電子・イオンの全密度を等しくします。
入射速度は壁向きを正とし、温度は eV、放出は外向きの数流束で与えます。
[方程式と枝の意味](model.md#fixed-reservoir) を参照してください。

### 入力

| 型 | 属性と単位 |
| --- | --- |
| `MaxwellianElectrons(density_m3, temperature_ev, drift_mps=0.0)` | 全速度域の密度、温度、内向きの法線ドリフト。複数の成分を足し合わせられる |
| `KappaElectrons(density_m3, core_temperature_ev, kappa)` | 全速度域の密度とコアの温度。中心は静止、有限の数流束には `kappa > 1` |
| `FixedReservoir(electrons, ion_density_m3, ion_speed_mps, ion_temperature_ev=0.0, ion_pressure_factor=1.0, ion_mass_kg=..., electron_mass_kg=...)` | 電子成分の列、イオンの密度と正の法線速度。質量の既定は陽子と電子。温イオンの入口は超音速 |
| `MaxwellianEmission(flux_m2_s, temperature_ev)` | 半 Maxwell 放出の総数流束と温度 |
| `BinnedEmission(energy_edges_ev, bin_flux_m2_s)` | 増加する非負のエネルギー境界と各 bin の積分数流束。bin 内で微分数流束が一定 |
| `EmissionSpectrum` | `flux_above(barrier_ev)` は障壁以上の数流束。 `differential_flux(energy_ev)` はスカラーまたは配列の法線エネルギーに対する微分数流束 [m⁻² s⁻¹ eV⁻¹] |

電子密度・電子温度と粒子質量は正、放出源の流束は非負です。両プロファイルソルバーは正の `MaxwellianEmission` を使います。
規格化された速度 $u=v/\sqrt{2e/m_e}$ での電子分布は `reservoir.phase(u)` で評価します。数密度はこの分布を $u$ で積分した値です。
深さ `depth` [V] に届く電子の流束と、その電位での通過電子密度は `reservoir.inward_flux(depth)` と `reservoir.passing_density(depth)` で評価できます。

### 診断とプロファイル

| 呼び出し | 意味 |
| --- | --- |
| `classify_branch(reservoir, emission, *, ratio_tolerance=1e-7, scan_points=96)` | 中性の仮の谷を探索して `BranchCriterion` を返す。比の許容幅は正、基本の走査点数は 8 以上 |
| `turning_field_difference(reservoir, emission, depth_v, barrier_ev, offset_v)` | 谷から `offset_v` [V] だけ高い両側の点で、壁側から上流側を引いた電場二乗差 [V²/m²]。オフセットは 0 から `min(depth_v, barrier_ev)` まで |
| `solve_finite_column(reservoir, emission, height_m, *, branch=None, points=2049, order=80)` | 上端 0 V、上端電場を自由にした高さ `height_m > 0` の零電流解。枝は `"A"`、`"B"`、`"C"` または単調解を先に試す `None`。電位側の点数は 129 以上、求積次数は 24 以上 |
| `solve_presheath(reservoir, emission, branch, *, seed=None, nodes=65, order=96, far_tau=16.0)` | `"A"`、`"B_out"`、`"B_nw"`、`"C"` の外側領域付きの零電流解。節点数は 9 以上の奇数、求積次数は 24 以上、外端までの光学距離は 1〜50 |
| `replacement_targets(reservoir, energy_ev, outer_depth_v=0.0)` | 置換先の（外向き、内向き）分布。単位は `reservoir.phase` と共通。負エネルギーにはゼロ速度の値を使い、正の外側谷の深さを与えるとその谷の手前の反射軌道に一次混合を使う |

`BranchCriterion` は `outcome`（`A`、`B`、`marginal`、`no_current`、`no_minimum`、`unresolved`）と、解決した中性谷の
`neutral_depth_v`、`emission_barrier_ev`、`ratio`、`effective_emission_temperature_ev` を返します。
有効温度は障壁以上の数流束を障壁での微分数流束で割った値です。
`no_current` は放出がゼロの場合か、障壁なしの入射電子流束がイオン流束以下の場合です。
`no_minimum` は走査点で電子の過剰がすべて負だった場合、`unresolved` は探索で候補を一意に絞れなかった場合です。

`FiniteColumnSolution` は `branch`、配列 `z_m`・`potential_v`・`field_v_m`、`escaping_flux_m2_s`、`minimum_height_m`、`monotone_max_height_m` を返します。
位置は壁の 0 m から上端 H へ増加します。有限の上限が見つからない単調解では上限が無限大になることがあります。

`PresheathSolution` は `branch`、`wall_potential_v`、`band_potential_v`、`minimum_potential_v`（谷のない B_nw では `None`）、
外側の配列 `outer_tau`・`outer_potential_v`、無衝突シースの配列 `sheath_z_m`・`sheath_potential_v`・`sheath_field_v_m`、
`current_residual_a_m2`、評価点の `outer_neutrality_residual_m3` と継続用の `outer_well_seed` を返します。
シースの配列は壁から準中性域へ向かい、漸近端の手前で止まります。外側の配列は準中性域を `tau=0` とします。

B_out では近隣解の `outer_well_seed` を次の `seed` に渡せます。初期値を直接与える場合は
`OuterWellSeed(minimum_depth_v, band_above_minimum_v, minimum_tau, escaping_flux_m2_s, turning_path=())` を使います。
任意の `turning_path` は谷を 0、準中性域を 1 とする増加座標と光学距離の組です。
極浅い谷には近隣解からの継続が有効です。両プロファイルソルバーは入力の不正を `ValueError`、数値的な探索の失敗を `ArithmeticError` で返します。

### 最初の計算

次の条件では、仮の中性谷からは Type B 側と判定されます。56.5 m の箱では上端側に谷が現れ、有限箱の出力は `A` になります。
外側領域付きのモデルで同じ入力を使うと、谷はシースの外に来る `B_out` が得られます。

```python
from scipy.constants import elementary_charge
from sheath_model import (
    FixedReservoir, MaxwellianElectrons, MaxwellianEmission,
    classify_branch, solve_finite_column, solve_presheath,
)

wind = FixedReservoir(
    (MaxwellianElectrons(20e6, 4.0, 400e3),),
    ion_density_m3=20e6, ion_speed_mps=400e3, ion_temperature_ev=10.0,
)
emission = MaxwellianEmission(36e-6 / elementary_charge, 2.2)
print(classify_branch(wind, emission).outcome)
column = solve_finite_column(wind, emission, 56.5)
print(column.branch, column.minimum_height_m)
outer = solve_presheath(wind, emission, "B_out")
print(outer.branch, outer.band_potential_v, outer.minimum_potential_v)
```

## 補助関数とコマンドライン

`ion_density_ratio(potential_v, entry_energy_ev, pressure_energy_ev=0.)` と `ion_critical_potential(entry_energy_ev, pressure_energy_ev=0.)` は
Fortran と同じ量を返し、範囲外では `ValueError` を送出します。

`python -m sheath_model` は `ZhaoParams` の J=0 解を表示します。

```bash
python -m sheath_model --branch A --alpha 60 --electron-drift-mode zero --ion-drift-mode normal --zmax-hat 120
```

コマンドラインのドリフトの既定値は `full` で、`ZhaoParams` の既定値（`normal`）と異なります。
ドリフトのある A/C は棄却されるので（[物理モデル](model.md#ドリフトのある-ac-と上流の帯)）、通常は `--electron-drift-mode zero` を指定してください。
