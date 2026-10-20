# Fortran API リファレンス

公開窓口は `use sheath_model` の一つだけです。`src/internal/` のモジュールと、ここに記載のない型束縛手続きは内部用で、予告なく変わります。
単位・座標・符号・ステータスは [共通の規約](conventions.md)、方程式と判定条件は [物理モデル](model.md) を参照してください。
粒子シミュレータとの結合 API は [シミュレータとの結合](coupling.md) にまとめています。

## 公開名の一覧

| 役割 | 名前 |
| --- | --- |
| 物理入力 | [`plasma_input`](#plasma_input)、[`fixed_entry_equilibrium_input`](#j0-の入力)、[`zhao_equilibrium_input`](#j0-の入力)、[`prescribed_field_input`](#e_h-指定の入力)、抽象型 `sheath_equilibrium_input` |
| 放出源 | [`emission_source`](#放出源)、`maxwellian_emission`、`binned_emission` |
| 数値設定と求解 | [`sheath_solver`](#sheath_solver)、`sheath_search_options`、`sheath_continuation_options`、`sheath_profile_options` |
| 結果 | [`sheath_equilibrium_result`](#j0-の結果)、[`prescribed_field_result`](#e_h-指定の結果)、[`sheath_profile_result`](#プロファイル)、[`sheath_density_result`](#局所密度)、[`sheath_search_diagnostics`](#探索の診断) |
| 静的評価 | [`evaluate_sheath_state`](#静的評価)、`sheath_state_result` |
| 局所密度 | [`evaluate_density`](#局所密度) |
| 解マップ | [`sheath_equilibrium_atlas`](#解マップと-continuation)、`sheath_field_atlas`、`sheath_atlas_options`、`sheath_atlas_point` |
| シミュレータ結合 | `sheath_matching_conditions`、`evaluate_matching_conditions`、`sheath_matching_iteration`、`plane_electron_access_energy_ev`、`plane_ion_access_energy_ev`、`plane_electron_escape_threshold_ev`、`plane_ion_escape_threshold_ev`、`plane_electron_inward_flux_m2_s`、`log_energy_edges`、`log_energy_bin_count`、`log_energy_bin`、`LOG_ENERGY_DEFAULT_*`、`energy_bin`、`observed_outflow_source`（[結合](coupling.md)） |
| 放出源の評価 | `emitted_fluxes`、`source%with_outward_flux`（[放出源](#放出源)） |
| 補助 | [`ion_density_ratio`、`ion_critical_potential`](#補助関数と定数)、`dp`、`i32`、`SHEATH_*` |

## 物理入力

### `plasma_input`

上流の状態と放出源を固定する型です。E_H 指定と静的評価にはこの型（または派生型）を、J=0 には `fixed_entry_equilibrium_input%plasma` として渡します。

| 成分 | 既定値 | 制約・意味 |
| --- | --- | --- |
| `ion_density_m3` | 8.7e6 | 正。上流イオン密度 n_i∞ |
| `ion_entry_speed_mps` | 405299.88897111727 | 正。上流での内向き法線速度 u_i∞ |
| `ion_temperature_ev` | 0 | 非負。0 で冷たいイオン |
| `ion_pressure_factor` | 1 | 正。イオン圧力項の係数 |
| `electron_temperature_ev` | 12 | 正。背景電子温度 T_e |
| `electron_drift_mps` | 0 | 背景電子の内向き法線ドリフト。正にすると A/C は既定では棄却される（[ドリフトのある A/C](model.md#ドリフトのある-ac-と上流の帯)） |
| `ion_mass_kg`, `electron_mass_kg` | 陽子・電子質量 | 正 |
| `emission` | 密度 0・温度 2.2 eV の Maxwell 源 | [放出源](#放出源) |

すべて有限値が必要です。入口はイオン熱音速より速く、`m_i u_i∞²/e > ion_pressure_factor*T_i` を満たす必要があります。
条件を満たさない入力は、求解・評価の呼び出し時に `SHEATH_INVALID_ARGUMENT` になります。
イオン入口速度の既定値は、太陽高度 60°・風速 468 km/s の法線成分です。

### 放出源

`emission_source` は境界から外向きに出る電子の分布で、成分は private です。次のコンストラクタで作ります。

```fortran
! 半 Maxwell 源: 規格化密度 N_em [m^-3] と温度 [eV]
plasma%emission = maxwellian_emission(density_m3=2.0e7_dp, temperature_ev=2.2_dp)

! bin 源: 法線運動エネルギーの境界 [eV] と、各 bin に積分した外向き数流束 [m^-2 s^-1]
plasma%emission = binned_emission( &
    energy_edges_ev=[0.0_dp, 1.0_dp, 3.0_dp, 8.0_dp], &
    bin_flux_m2_s=[2.0e12_dp, 5.0e11_dp, 1.0e11_dp])
```

| 引数 | 制約 |
| --- | --- |
| `density_m3` | 有限・非負。0 は放出なし |
| `temperature_ev` | 有限・正。放出なしでも必要（内部の電位尺度に使う） |
| `energy_edges_ev(N+1)` | 有限・非負・厳密な昇順、N ≥ 1。最初の境界は 0 より大きくてもよい |
| `bin_flux_m2_s(N)` | 有限・非負。全て 0 も可 |

bin 内は `g(K) = bin_flux/(K_high − K_low)` が一定、範囲外は 0 です（[放出電子](model.md#放出電子)）。
源は渡した配列をコピーして所有します。値の検査は求解・評価の呼び出し時に行い、不正な源は `SHEATH_INVALID_ARGUMENT` です。

| 型束縛手続き | 結果 |
| --- | --- |
| `source%is_binned()` | bin 源なら `.true.` |
| `source%mean_normal_energy()` | 流束で重み付けした平均法線運動エネルギー [eV]。Maxwell 源は温度、bin 源は bin 中点の加重平均、空の bin 源は 0 |
| `source%potential_scale(T_e)` | 内部の電位尺度 V_s [V]。Maxwell 源は温度、bin 源は T_e 以下で最大の 2 の冪。[接続残差の換算](#静的評価)に使う |
| `source%with_outward_flux(total_flux_m2_s, mass_kg)` | 分布の形を保って外向き数流束の合計を `total_flux_m2_s` にした源。Maxwell 源は規格化密度 N_em = 2√π Γ/v_th、bin 源は各 bin を比例配分 |

源の流束は `emitted_fluxes(source, mass_kg, barrier_v, outward, escape, returning)` で評価します。`outward` は境界を出る
数流束の合計、`escape` と `returning` は面から上流まで `barrier_v` [V] を登る分と戻る分です（Maxwell 源の外向き流束は
N_em v_th/(2√π)）。

### J=0 の入力

どちらも抽象型 `sheath_equilibrium_input` の派生型で、`branch`（既定 `'auto'`）を持ちます。

**`fixed_entry_equilibrium_input`** は法線の入口状態を直接与えます。成分は `branch` と `plasma`（`plasma_input`）で、
`plasma` の既定値は `plasma_input` と同じです。bin 源も使えます。

**`zhao_equilibrium_input`** は太陽高度と全風速から法線条件と光電子源を作ります（[変換規則](model.md#太陽風と照射角からの入力)）。放出源は Maxwell 源です。

| 成分 | 既定値 | 意味 |
| --- | --- | --- |
| `sun_elevation_deg` | 60 | 太陽高度 α。0〜90 |
| `ion_density_m3` | 8.7e6 | 上流イオン密度 |
| `photoelectron_reference_density_m3` | 64e6 | 垂直入射時の光電子規格化密度 n_pe,ref |
| `electron_temperature_ev`, `ion_temperature_ev`, `ion_pressure_factor` | 12, 0, 1 | `plasma_input` と同じ |
| `photoelectron_temperature_ev` | 2.2 | 光電子温度 |
| `solar_wind_speed_mps` | 468e3 | 全風速 v_sw |
| `ion_mass_kg`, `electron_mass_kg` | 陽子・電子質量 | |
| `electron_drift_mode` | `'normal'` | `'normal'`、`'full'`、`'zero'`（小文字で指定） |
| `ion_drift_mode` | `'normal'` | `'normal'`、`'full'`（小文字で指定） |

### E_H 指定の入力

`prescribed_field_input` は `plasma_input` を継承し、`branch`（既定 `'auto'`）と `electric_field_v_m`（既定 0）を加えます。
共通部分は `input%plasma_input = plasma` でまとめて設定するか、`input%electron_temperature_ev` のように直接設定します。
既定値は `plasma_input` と同じです。

## `sheath_solver`

数値設定と、任意で探索用の解マップを持つ型です。求解メソッドは solver を変更しません。

| 成分 | 型 | 内容 |
| --- | --- | --- |
| `search` | `sheath_search_options` | 根の探索 |
| `continuation` | `sheath_continuation_options` | 解マップからの continuation |
| `profile` | `sheath_profile_options` | プロファイルの標本化 |
| `equilibrium_atlas`, `field_atlas` | allocatable の解マップ | `solver%equilibrium_atlas = atlas` でコピーを持たせる。不要なら deallocate |

### 探索の設定 `search`

| 成分 | 既定値 | 意味 |
| --- | --- | --- |
| `method` | `'auto'` | `'auto'`、`'newton'`、`'lm'`、`'bracket'`。`'bracket'` は Type B/C を明示したときだけ |
| `residual_tolerance` | 1e-10 | 規格化した元の方程式の最大絶対残差。正 |
| `max_iterations`, `max_backtracks`, `max_starts` | 100, 24, 32 | 反復・減速・初期値の上限 |
| `use_default_guesses` | `.true.` | 標準の初期値を使う |
| `bracket_points` | 96 | スカラー探索の格子点数。2 以上 |
| `potential_extent` | 200 | 探索する規格化電位の範囲（電位尺度の倍数）。正 |
| `scalar_max_depth` | 8 | E_H 指定 B/C のスカラー探索での区間の細分化深さ。1〜16 |
| `upstream_band_tolerance` | 0 | ドリフトのある A/C の上流の帯の許容幅（割合）。0 以上 1 未満 |
| `upstream_field_floor_v_m` | 0 | 同じく場の不足の許容値 [V/m]。非負 |

上流の帯の二つの設定は J=0 の探索と `build_profile` に使います（[物理モデル](model.md#ドリフトのある-ac-と上流の帯)）。
探索法の詳細は [数値解法](numerics.md#根の探索) を参照してください。

### continuation の設定 `continuation`

| 成分 | 既定値 | 意味 |
| --- | --- | --- |
| `method` | `'parameter'` | `'parameter'` は入力の経路を順に追う。`'arclength'` は単純な折り返しも越える |
| `initial_step`, `min_step`, `max_step` | 0.25, 1e-4, 0.5 | 経路パラメータ（0→1）の刻み。`min ≤ initial ≤ max` |
| `max_steps` | 128 | 刻みの上限 |
| `max_root_distance` | 0.75 | 一刻みで許す解の移動（無次元の対数座標）。超えると刻みを縮めて再試行 |

### プロファイルの設定 `profile`

| 成分 | 既定値 | 意味 |
| --- | --- | --- |
| `points_per_segment` | 4000 | 単調区間あたりの電位点数。32 以上。A は 2 区間 |
| `max_distance_m` | 100 | 返す高さの上限 [m] |
| `potential_cutoff_v` | 2.2e-3 | 上流側の打ち切り電位 \|φ\| [V]。境界電位・極小の深さの半分を上限とする |

## 求解メソッド

```fortran
call solver%solve_equilibrium(input, root, status, message, diagnostics, initial_guesses)
```

| メソッド | 入力 → 出力 | 動作 |
| --- | --- | --- |
| `solve_equilibrium(input, root, status, message [, diagnostics, initial_guesses])` | J=0 入力 → `sheath_equilibrium_result` | 指定 Type、または auto の順で最初に見つかった物理解 |
| `solve_equilibrium_candidates(input, roots, status, message [, diagnostics, initial_guesses, deflation, max_roots])` | J=0 入力 → allocatable 配列 | 見つかった異なる物理解をすべて返す |
| `solve_prescribed_field(input, root, status, message [, diagnostics, initial_guesses, deflation, max_roots])` | `prescribed_field_input` → `prescribed_field_result` | 物理解が一つだけ見つかったとき返す。複数なら `SHEATH_AMBIGUOUS_SOLUTION` |
| `solve_prescribed_field_candidates(input, roots, status, message [, ...同上])` | `prescribed_field_input` → allocatable 配列 | 候補を選ばずに返す。順序に意味はない |
| `solve_profile(input, profile, status, message [, diagnostics, initial_guesses])` | J=0 入力 → `sheath_profile_result` | `solve_equilibrium` の後に `build_profile` |
| `build_profile(input, root, profile, status, message)` | J=0 入力と既存解 → `sheath_profile_result` | 探索せず、元の方程式と判定条件で検査してから再構成 |

| 省略可能な引数 | 既定値・意味 |
| --- | --- |
| `diagnostics` | `sheath_search_diagnostics`。呼び出しごとに初期化 |
| `initial_guesses(:)` | 同じ閉じ方の結果型の配列。標準の初期値に追加する。出力と同じ変数を渡さない |
| `deflation` | 既知の根を退ける deflation。J=0 の候補探索は `.true.`、E_H 指定は `.false.` |
| `max_roots` | Type ごとの候補の上限。既定 16、正 |

`auto` の試行順は次のとおりです。

| 閉じ方 | 順序 |
| --- | --- |
| J=0、`zhao_equilibrium_input` | 太陽高度 20° 未満で C→A→B、それ以外 A→B→C |
| J=0、`fixed_entry_equilibrium_input` | A→B→C |
| E_H 指定 | E_H ≥ 0 で A→B→C、E_H < 0 で C→A→B。候補探索は全 Type を探す |

失敗時は結果型が `valid = .false.`、プロファイルの配列は未確保です。
候補配列は、J=0 では長さ 0、E_H 指定では未確保になります。
プロファイルと局所密度の API は J=0 の解だけに対応します。E_H 指定の根の境界での量は、結果型のほか [静的評価](#静的評価) や [`evaluate_matching_conditions`](coupling.md#面上の条件を取り出す) で得られます。

## 結果

### J=0 の結果

`sheath_equilibrium_result`

| 成分 | 単位・意味 |
| --- | --- |
| `valid`, `branch` | 有効な解か、Type（`'A'`/`'B'`/`'C'`） |
| `surface_potential_v`, `minimum_potential_v` | φ_H、φ_min [V]。B は φ_min=0、C は φ_min=φ_H |
| `ambient_electron_density_m3` | 背景電子の規格化 N_e [m⁻³]。上流の全電子密度ではない |
| `debye_length_m` | 内部の長さ尺度 √(ε0 V_s/(e n)) [m]（[規格化](numerics.md#規格化)） |
| `residual_norm` | 規格化した残差の最大値（無次元） |
| `electron_inward_flux_m2_s`, `ion_inward_flux_m2_s` | 境界へ入る電子・イオンの数流束 |
| `emitted_escape_flux_m2_s` | 上流へ抜ける放出電子の数流束 |
| `net_current_a_m2` | +z 向きの通常電流 [A/m²]。J=0 なのでほぼ 0 |
| `upstream_negative_band_v`, `upstream_field_deficit_v_m` | 受理した上流の帯の幅 [V] と場の不足 [V/m]。ドリフトのある A/C を受理したときだけ正 |

境界電場は [静的評価](#静的評価) または [`evaluate_matching_conditions`](coupling.md#面上の条件を取り出す) で得られます。

### E_H 指定の結果

`prescribed_field_result`

| 成分 | 単位・意味 |
| --- | --- |
| `valid`, `branch` | 有効な解か、Type |
| `boundary_potential_v`, `minimum_potential_v` | φ_H、φ_min [V] |
| `ambient_electron_density_m3` | 背景電子の規格化 N_e [m⁻³] |
| `electron_inward_flux_m2_s`, `ion_inward_flux_m2_s` | 境界へ入る数流束 |
| `emitted_outward_flux_m2_s`, `emitted_escape_flux_m2_s`, `emitted_return_flux_m2_s` | 放出電子の外向き・上流到達・戻りの数流束 |
| `net_current_a_m2` | +z 向きの通常電流 [A/m²]。閉じ方として課していない応答 |
| `residual_norm`, `minimum_field_squared_hat` | 規格化残差、経路上の E² の最小値（無次元） |
| `nonlinear_iterations` | 非線形反復の回数 |

### プロファイル

`sheath_profile_result` は J=0 の解と、同じ格子上の配列を持ちます。

| 成分 | 内容 |
| --- | --- |
| `equilibrium` | 再構成した `sheath_equilibrium_result` |
| `z_m(:)`, `potential_v(:)`, `electric_field_v_m(:)` | 境界（z=0）から上流へ向かう高さ・電位・電場。間隔は非一様 |
| `density(:)` | 各点の `sheath_density_result` |
| `turning_height_m` | A の極小の高さ [m]。B/C は −1 |

プロファイルは半無限の Poisson 一次積分から作ります。`potential_cutoff_v` または `max_distance_m` で止め、
その先に人工的な 0 V の区間は加えません。受理した上流の帯がある場合は帯の手前で止めます。

### 局所密度

```fortran
call evaluate_density(input, root, potential_v, density, status, message, side)
```

J=0 の解 `root` と、それを解いた `input` から、電位 `potential_v` [V] での密度を返します。
A は `side='lower'`（境界から極小）または `'upper'`（極小から上流）が必要で、B/C は省略できます。
電位は選んだ区間の内側でなければなりません。

`sheath_density_result` の成分は `ion_m3`、`electron_free_m3`（境界へ届く背景電子）、`electron_reflected_m3`、
`emitted_free_m3`（上流へ抜ける放出電子）、`emitted_captured_m3`（戻る放出電子）[m⁻³] と、`charge_c_m3` [C/m³] です。

## 静的評価

閉じ方を課さず、与えた電位での状態を評価します。solver は不要で、状態も持ちません。
時間発展する外側の方程式や、利用側の電場・電流条件の中で使うための API です。

```fortran
program static_example
  use sheath_model
  implicit none
  type(plasma_input) :: plasma
  type(sheath_state_result) :: state
  integer(i32) :: status
  character(len=256) :: message

  plasma%electron_drift_mps = 0.0_dp
  plasma%emission = binned_emission( &
      [0.0_dp, 1.0_dp, 3.0_dp, 8.0_dp], [2.0e12_dp, 5.0e11_dp, 1.0e11_dp])
  call evaluate_sheath_state(plasma, 'B', 0.5_dp, state, status, message)
  if (status /= SHEATH_OK) stop 1
  print *, state%boundary_field_squared_v2_m2, state%net_current_a_m2
  if (state%admissible) print *, state%electric_field_v_m
end program static_example
```

`evaluate_sheath_state(plasma, branch, boundary_potential_v, state, status, message [, minimum_potential_v, electron_normalization_m3])`

| branch | 電位の入力 [V] | 条件 |
| --- | --- | --- |
| `'B'` | `boundary_potential_v` = φ_H ≥ 0 | φ_min=0。平坦な端点 φ_H=0 も評価できる |
| `'C'` | φ_H < 0 | φ_min=φ_H |
| `'A'` | φ_H と `minimum_potential_v` = φ_m < min(φ_H, 0) | A では必須、B/C では指定すると入力エラー |

`'auto'` は使えません。N_e は与えた電位での上流の準中性から求めます。

| `sheath_state_result` の成分 | 単位・意味 |
| --- | --- |
| `evaluated` | 有限の評価に成功したか |
| `admissible`, `physical_status`, `physical_message` | 物理解の判定に合格したか、その結果。A は接続も必要 |
| `branch`, `boundary_potential_v`, `minimum_potential_v` | 評価した Type と電位 [V] |
| `ambient_electron_density_m3` | N_e [m⁻³] |
| `neutrality_residual_m3` | 上流での n_i − n_e − n_em [m⁻³] |
| `boundary_field_squared_v2_m2` | 符号を保った境界の E² [V²/m²]。負でも 0 に置き換えない |
| `electric_field_v_m` | Type の符号を持つ境界電場 [V/m]。E² が負なら NaN |
| `connection_residual_v2_m2` | A の `−(2/ε0) ∫[φ_m..0] ρ_upper dφ` [V²/m²]。B/C は 0 |
| `connection_residual_normalized` | 同じ残差の無次元量（下記）。B/C は 0 |
| `electron_inward_flux_m2_s`, `ion_inward_flux_m2_s` | 境界へ入る数流束 |
| `emitted_outward_flux_m2_s`, `emitted_escape_flux_m2_s`, `emitted_return_flux_m2_s` | 放出電子の数流束 |
| `net_current_a_m2` | +z 向きの通常電流 [A/m²] |

**`status == SHEATH_OK` は「試行状態を評価できた」という意味で、解であることは意味しません。**
`admissible` が偽でも、符号付きの E² や残差は外側の方程式を解くのに使えます。
A の通常の試行点では接続残差は 0 でなく、完成した解ではありません。
`electric_field_v_m` を使う前に、`admissible` か E² ≥ 0 を確認してください。

A の外側の方程式には `connection_residual_normalized` R_A を使えます。
電位尺度 `V_s = plasma%emission%potential_scale(plasma%electron_temperature_ev)` とすると、SI の残差との関係は

```text
R_A = connection_residual_v2_m2 * eps0 * sqrt(V_s) / (e * n_i * (−φ_m)^(3/2))
```

です。浅い極小で積分区間が縮むだけで残差が小さく見える効果を、(−φ_m)^(3/2) で除いています。

`electron_normalization_m3` は、急峻なスペクトルの端点で外側の方程式の根が隣り合う binary64 の電位の間に入る場合のための引数で、
通常は省略します。B/C だけで指定でき、範囲は **現在の電位の直前・直後の表現可能値で準中性から得た N_e の間** に限ります。
指定した N_e で密度・電場・電流をすべて計算し直し、準中性の残差と判定条件はそのまま検査します。範囲外は入力エラーです。

## 解マップと continuation

解マップは、受理した根を無次元座標で記録する、呼び出し側所有の表です。探索の初期値と continuation の出発点にだけ使い、
問い合わせでは毎回元の方程式を解き直します。閉じ方ごとに `sheath_equilibrium_atlas`（J=0、6 次元のキー）と
`sheath_field_atlas`（E_H 指定、E_H を加えた 7 次元のキー）を使います。

```fortran
call solver%build_equilibrium_atlas(inputs, atlas, status, message [, report, deflation])
call solver%build_field_atlas(inputs, atlas, status, message [, report, deflation, max_roots])
solver%equilibrium_atlas = atlas          ! 以後の求解で初期値に使う
call solver%solve_equilibrium(query, root, status, message, diagnostics)
```

| 手続き | 動作 |
| --- | --- |
| `build_equilibrium_atlas`, `build_field_atlas` | 入力の配列を解いて既存の表に追加する。後から見つかった近傍解で未解決の点を再探索する。J=0 の `deflation` は既定 `.false.`（`.true.` で候補探索） |
| `add_equilibrium_to_atlas(input, root, atlas, status, message [, component])` | 一つの根を、元の方程式と判定条件で検査してから登録する |
| `add_field_to_atlas(input, root, atlas, status, message [, component])` | E_H 指定の根を同様に登録する |

`report(3, n)` は Type（A/B/C）× 入力ごとのステータスで、指定しなかった Type は `SHEATH_INVALID_ARGUMENT` です。
構築の `SHEATH_OK` は有限の探索が終わったことを意味し、全入力で解が見つかったことは意味しません。
求解は表を変更せず、自動登録もしません。

| 表の操作 | 意味 |
| --- | --- |
| `atlas%size()`, `atlas%point(i)` | 登録数と i 番目の `sheath_atlas_point`（`branch`、`component`、`key(:)`、`coordinates(3)`、`spectrum_shape(:)`） |
| `atlas%clear()` | すべて削除 |
| `atlas%write(unit, iostat)`, `atlas%read(unit, iostat)` | 版付きのテキスト形式。unit は formatted で開く。読み込みに失敗した場合、表は変わらない |
| `atlas%options` | `sheath_atlas_options`: `neighbors=8`、`max_distance=1`、`interpolate=.true.`、`component_distance=0.75` |

`component` は Type 内の解の族を区別する正の整数です。省略時は近さで自動に付けますが、数学的な枝の接続を証明するものではありません。
J=0 の表の形式は Python の `EquilibriumAtlas` と共通です。スペクトル形状の異なる記録は、別の源の問い合わせに使いません。

## 探索の診断

`sheath_search_diagnostics` の成分はすべて A/B/C 順の長さ 3 の配列です。

| 成分 | 意味 |
| --- | --- |
| `searched`, `excluded` | その Type を探索したか、解析的な条件で除外したか |
| `starts`, `unconverged`, `rejected`, `profile_failures` | 初期値の数、未収束、物理条件で棄却、プロファイル評価の失敗 |
| `roots_found`, `best_residual` | 見つかった根の数、最小の残差 |
| `evaluations`, `iterations`, `lm_steps`, `brackets` | 残差評価・反復・LM ステップ・括弧付けの回数 |
| `atlas_starts`, `atlas_hits`, `continuation_steps`, `continuation_retries`, `deflations` | 解マップと continuation、deflation の使用 |
| `subdivisions`, `tangencies`, `invalid_evaluations` | スカラー探索の細分化・接する根・無効な評価点 |

採用した解があっても未収束の試行は残ります。採用候補がなく、棄却と未収束が混在する場合は `SHEATH_NUMERICAL_FAILURE` です。

## 補助関数と定数

| 名前 | 内容 |
| --- | --- |
| `ion_density_ratio(potential_v, entry_energy_ev, pressure_energy_ev)` | イオン密度比 n_i/n_i∞。`entry_energy_ev = m_i u_i∞²/(2e)`、`pressure_energy_ev = ion_pressure_factor*T_i`。越えられない電位では NaN |
| `ion_critical_potential(entry_energy_ev, pressure_energy_ev)` | イオンが越えられる最大電位 [V]。入口が熱音速以下なら NaN |
| `plane_electron_access_energy_ev`, `plane_ion_access_energy_ev`, `plane_electron_escape_threshold_ev`, `plane_ion_escape_threshold_ev`, `plane_electron_inward_flux_m2_s` | matching plane での注入・反射・除去の判定と、到達条件で制限した上流電子の数流束（[結合](coupling.md#粒子の注入反射除去)） |
| `log_energy_edges`, `log_energy_bin_count`, `log_energy_bin`, `energy_bin`, `LOG_ENERGY_DEFAULT_LOWEST_EV`, `LOG_ENERGY_DEFAULT_DECADES`, `LOG_ENERGY_DEFAULT_BINS_PER_DECADE`, `LOG_ENERGY_DEFAULT_BINS` | 流出スペクトルの法線エネルギー bin（[結合](coupling.md#流出スペクトルから放出源を作る)） |
| `dp`, `i32` | `real64`、`int32` の kind |
| `SHEATH_OK` ほか | [ステータス](conventions.md#ステータスとエラー処理) |
