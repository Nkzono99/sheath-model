# Fortran API リファレンス

Fortran ライブラリの公開窓口は **`use sheath_model` のみ** です。`src/internal/` 内のモジュールや、このページおよび [粒子シミュレータとの結合](coupling.md) に記載のない型束縛手続きは内部実装であり、予告なく変更されます。

- 単位・座標・符号・`status` 定数 → [共通の規約](conventions.md)
- 方程式・Type A/B/C・物理解判定 → [物理モデル](model.md)
- 粒子シミュレータとの結合 API → [粒子シミュレータとの結合](coupling.md)

## 目次

1. [公開名の一覧](#公開名の一覧)
2. [物理入力と放出源](#物理入力)
3. [ソルバーと数値設定（`sheath_solver`）](#sheath_solver)
4. [求解メソッド](#求解メソッド)
5. [結果型と局所密度](#結果)
6. [静的評価（`evaluate_sheath_state`）](#静的評価)
7. [解マップと continuation](#解マップと-continuation)
8. [探索の診断（`sheath_search_diagnostics`）](#探索の診断)
9. [補助関数と定数](#補助関数と定数)

---

## 公開名の一覧

| 役割 | 公開シンボル |
| --- | --- |
| **物理入力** | [`plasma_input`](#plasma_input)、[`fixed_entry_equilibrium_input`](#j0-の入力)、[`zhao_equilibrium_input`](#j0-の入力)、[`prescribed_field_input`](#e_h-指定の入力)、抽象型 `sheath_equilibrium_input` |
| **放出源** | [`emission_source`](#放出源)、`maxwellian_emission`、`binned_emission`、`emitted_fluxes` |
| **ソルバー・設定** | [`sheath_solver`](#sheath_solver)、`sheath_search_options`、`sheath_continuation_options`、`sheath_profile_options` |
| **結果・密度・診断** | [`sheath_equilibrium_result`](#j0-の結果)、[`prescribed_field_result`](#e_h-指定の結果)、[`sheath_profile_result`](#プロファイル)、[`sheath_density_result`](#局所密度)、`evaluate_density`、[`sheath_search_diagnostics`](#探索の診断) |
| **静的評価** | [`evaluate_sheath_state`](#静的評価)、`sheath_state_result` |
| **解マップ** | [`sheath_equilibrium_atlas`](#解マップと-continuation)、`sheath_field_atlas`、`sheath_atlas_options`、`sheath_atlas_point` |
| **シミュレータ結合** | `sheath_matching_conditions`、`evaluate_matching_conditions`、`sheath_matching_iteration`、`matching_states_at`、`select_combined_matching_conditions`、`plane_electron_access_energy_ev`、`plane_ion_access_energy_ev`、`plane_electron_escape_threshold_ev`、`plane_ion_escape_threshold_ev`、`plane_electron_inward_flux_m2_s`、`log_energy_edges`、`log_energy_bin_count`、`log_energy_bin`、`energy_bin`、`observed_outflow_source`、`LOG_ENERGY_DEFAULT_*`（詳細は [結合 API](coupling.md)） |
| **補助関数・定数** | [`ion_density_ratio`、`ion_critical_potential`](#補助関数と定数)、`dp`、`i32`、`SHEATH_OK` ほか（[ステータス](conventions.md#ステータスとエラー処理)） |

---

<a id="物理入力"></a>

## 1. 物理入力と放出源

### `plasma_input`

上流プラズマの状態と境界の放出源をまとめる基本型です。$E_H$ 指定と静的評価にはこの型（または派生型）を、固定入口の $J_z = 0$ には `fixed_entry_equilibrium_input%plasma` として渡します。

| 成分 | 既定値 | 制約・意味 |
| --- | --- | --- |
| `ion_density_m3` | `8.7e6` | 正。上流イオン密度 $n_{i\infty}$ |
| `ion_entry_speed_mps` | `405299.88897111727` | 正。上流での内向き法線速度 $u_{i\infty}$（既定値は $468\text{ km/s}\cdot\sin 60^\circ$） |
| `ion_temperature_ev` | `0.0` | 非負。`0` で冷たいイオン |
| `ion_pressure_factor` | `1.0` | 正。温かいイオン圧力項の係数（[イオン](model.md#イオン)） |
| `electron_temperature_ev` | `12.0` | 正。背景電子温度 $T_e$ |
| `electron_drift_mps` | `0.0` | 背景電子の内向き法線ドリフト $v_d$。正にすると A / C は既定で棄却される（[詳細](model.md#ドリフトのある-ac-と上流の帯)） |
| `ion_mass_kg`, `electron_mass_kg` | 陽子・電子質量 | 正 |
| `emission` | 密度 0・温度 2.2 eV の Maxwell 源 | 境界からの放出源（[`emission_source`](#放出源)） |

すべての実数成分は有限値である必要があります。また入口でイオン熱音速を超える条件 $m_i u_{i\infty}^2 / e > (\text{\texttt{ion\_pressure\_factor}})\cdot T_i$ を満たさない場合、求解・評価時に `SHEATH_INVALID_ARGUMENT` を返します。

---

<a id="放出源"></a>

### 放出源（`emission_source`）

`emission_source` は境界から $+z$ 向きに出る電子の分布を表す不透明型（成分は `private`）です。コンストラクタ関数で作ります。

```fortran
! 半 Maxwell 源: 規格化密度 N_em [m^-3] と温度 T_em [eV]
plasma%emission = maxwellian_emission(density_m3=2.0e7_dp, temperature_ev=2.2_dp)

! bin 源: 昇順の法線エネルギー境界 [eV] と各 bin の積分数流束 [m^-2 s^-1]
plasma%emission = binned_emission( &
    energy_edges_ev=[0.0_dp, 1.0_dp, 3.0_dp, 8.0_dp], &
    bin_flux_m2_s=[2.0e12_dp, 5.0e11_dp, 1.0e11_dp])
```

| 引数 | 制約 |
| --- | --- |
| `density_m3` | 有限・非負（`0` は放出なし） |
| `temperature_ev` | 有限・正（放出なしでも内部の電位尺度に使うため必須） |
| `energy_edges_ev(N+1)` | 有限・非負・厳密に昇順、$N \ge 1$（先頭は $0$ より大きくても可） |
| `bin_flux_m2_s(N)` | 有限・非負（すべて `0` も可） |

bin 源では各 bin 内で $g(K) = \text{bin\_flux} / (K_{\mathrm{high}} - K_{\mathrm{low}})$ が一定、範囲外は $0$ です（[放出電子](model.md#放出電子)）。配列はコピーされて所有されます。不正な源は求解・評価時に `SHEATH_INVALID_ARGUMENT` になります。

**型束縛手続きと流束評価**：

| 手続き | 戻り値・動作 |
| --- | --- |
| `source%is_binned()` | bin 源なら `.true.` |
| `source%mean_normal_energy()` | 流束重み付き平均法線運動エネルギー [eV]。Maxwell 源は $T_{\mathrm{em}}$、bin 源は bin 中点の加重平均、空の bin 源は `0` |
| `source%potential_scale(t_e_ev)` | 内部の電位尺度 $V_s$ [V]。Maxwell 源は $T_{\mathrm{em}}$、bin 源は $T_e$ 以下で最大の $2$ の冪。[静的評価の接続残差換算](#静的評価)に使用 |
| `source%with_outward_flux(total_flux_m2_s, mass_kg)` | 分布の形状を保ったまま総外向き数流束を `total_flux_m2_s` にスケールした新しい `emission_source` |
| `call emitted_fluxes(source, mass_kg, barrier_v, outward, escape, returning)` | 検証済み源に対し、障壁 `barrier_v` [V]（非負）での外向き合計・脱出・戻りの各数流束 [$\mathrm{m^{-2}\,s^{-1}}$] を返す `pure subroutine` |

---

<a id="j0-の入力"></a>

### $J_z = 0$ の入力

どちらも抽象型 `sheath_equilibrium_input`（共通成分：`character(len=9) :: branch = 'auto'`）の派生型です。

#### `fixed_entry_equilibrium_input`

法線入口条件を直接指定する型です。bin 放出源にも対応します。成分は `branch` と `plasma`（型 `plasma_input`、既定値は `plasma_input` と同じ）です。

#### `zhao_equilibrium_input`

太陽高度と全太陽風速から法線条件と半 Maxwell 光電子源を自動構成する型です（変換規則は [物理モデル](model.md#太陽風と照射角からの入力)）。

| 成分 | 既定値 | 制約・意味 |
| --- | --- | --- |
| `branch` | `'auto'` | `'A'`、`'B'`、`'C'`、`'auto'`（大小文字不問） |
| `sun_elevation_deg` | `60.0` | 太陽高度 $\alpha$（`0`〜`90`） |
| `ion_density_m3` | `8.7e6` | 正。上流イオン密度 |
| `photoelectron_reference_density_m3` | `64.0e6` | 正。垂直入射時の光電子規格化密度 $n_{\mathrm{pe,ref}}$ |
| `electron_temperature_ev`, `ion_temperature_ev`, `ion_pressure_factor` | `12.0`, `0.0`, `1.0` | `plasma_input` と同じ |
| `photoelectron_temperature_ev` | `2.2` | 正。光電子温度 $T_{\mathrm{pe}}$ |
| `solar_wind_speed_mps` | `468.0e3` | 正。全太陽風速 $v_{\mathrm{sw}}$ |
| `ion_mass_kg`, `electron_mass_kg` | 陽子・電子質量 | 正 |
| `electron_drift_mode` | `'normal'` | `'normal'`、`'full'`、`'zero'`（**小文字**） |
| `ion_drift_mode` | `'normal'` | `'normal'`、`'full'`（**小文字**） |

---

<a id="e_h-指定の入力"></a>

### $E_H$ 指定の入力（`prescribed_field_input`）

`prescribed_field_input` は `plasma_input` を継承し、次の 2 成分を追加します。

| 追加成分 | 既定値 | 意味 |
| --- | --- | --- |
| `branch` | `'auto'` | `'A'`、`'B'`、`'C'`、`'auto'`（大小文字不問） |
| `electric_field_v_m` | `0.0` | 指定する境界電場 $E_H = -d\phi/dz\rvert_{z=0^+}$ [$\mathrm{V/m}$] |

共通部分は `input%plasma_input = plasma` でまとめて代入するか、`input%electron_temperature_ev = 10.0_dp` のように直接設定します。

---

<a id="sheath_solver"></a>

## 2. ソルバーと数値設定（`sheath_solver`）

`sheath_solver` は数値設定と（任意の）解マップを束ねる設定オブジェクトです。**求解メソッドはすべて `intent(in)` であり、ソルバー自身を変更せず、前回の解も保持しません。**

| 成分 | 型 | 内容 |
| --- | --- | --- |
| `search` | `sheath_search_options` | 非線形探索と上流帯の許容設定 |
| `continuation` | `sheath_continuation_options` | 解マップからの continuation 設定 |
| `profile` | `sheath_profile_options` | プロファイル格子の標本化設定 |
| `equilibrium_atlas`, `field_atlas` | `allocatable` 解マップ | `solver%equilibrium_atlas = atlas` のようにスナップショットを添付。外すときは `deallocate` |

### 探索設定（`sheath_search_options`）

| 成分 | 既定値 | 制約・意味 |
| --- | --- | --- |
| `method` | `'auto'` | `'auto'`、`'newton'`、`'lm'`、`'bracket'`（`'bracket'` は Type B / C を明示したときのみ有効） |
| `residual_tolerance` | `1e-10` | 正。規格化した元の方程式の最大絶対残差上限 |
| `max_iterations` | `100` | $0$ 以上。非線形反復の上限 |
| `max_backtracks` | `24` | 正。直線探索／減衰ステップの上限 |
| `max_starts` | `32` | 正。多点出発の初期値数上限 |
| `use_default_guesses` | `.true.` | 標準初期値を使うか |
| `bracket_points` | `96` | $2$ 以上。スカラー探索の初期格子点数 |
| `potential_extent` | `200.0` | 正。探索電位の無次元上限（電位尺度 $V_s$ の倍数） |
| `scalar_max_depth` | `8` | $1$〜$16$。$E_H$ 指定 B / C の適応細分化深さ |
| `upstream_band_tolerance` | `0.0` | $0 \le \text{値} < 1$。正ドリフト A / C の上流帯許容幅（割合） |
| `upstream_field_floor_v_m` | `0.0` | 非負。正ドリフト A / C の上流帯における場の不足上限 [$\mathrm{V/m}$] |

上流帯の 2 設定は $J_z = 0$・$E_H$ 指定の探索および `build_profile` に適用されます（[物理モデル](model.md#ドリフトのある-ac-と上流の帯)、探索法の詳細は [数値解法](numerics.md#根の探索)）。

### continuation 設定（`sheath_continuation_options`）

| 成分 | 既定値 | 制約・意味 |
| --- | --- | --- |
| `method` | `'parameter'` | `'parameter'`（入力経路を順に追跡）または `'arclength'`（単純な折り返しを越える擬似弧長法） |
| `initial_step`, `min_step`, `max_step` | `0.25`, `1e-4`, `0.5` | 経路パラメータ（$0 \to 1$）の刻み。$0 < \text{\texttt{min\_step}} \le \text{\texttt{initial\_step}} \le \text{\texttt{max\_step}}$ |
| `max_steps` | `128` | 正。ステップ数の上限 |
| `max_root_distance` | `0.75` | 正。1 ステップで許容する解の移動距離（無次元対数座標）。超えると刻みを縮めて再試行 |

### プロファイル設定（`sheath_profile_options`）

| 成分 | 既定値 | 制約・意味 |
| --- | --- | --- |
| `points_per_segment` | `4000` | $32$ 以上。単調区間あたりの電位点数（Type A は境界側と上流側の 2 区間） |
| `max_distance_m` | `100.0` | 正。出力する高さの上限 [m] |
| `potential_cutoff_v` | `2.2e-3` | 正。上流側の打ち切り電位 $\lvert \phi \rvert$ [V]（境界電位・極小深さの半分が上限） |
| `upstream_band_margin` | `0.05` | $0 < \text{値} < 1$。受理した上流の帯がある根で、帯の手前に残す余白（帯端〜極小／境界の深さに対する割合） |

---

<a id="求解メソッド"></a>

## 3. 求解メソッド

すべて `sheath_solver` の型束縛手続きです。

| メソッド | 入出力 | 動作 |
| --- | --- | --- |
| `solver%solve_equilibrium(input, root, status, message [, diagnostics, initial_guesses])` | $J_z = 0$ 入力 → `sheath_equilibrium_result` | 指定 Type、または `auto` の試行順で最初に見つかった物理解を返す |
| `solver%solve_equilibrium_candidates(input, roots, status, message [, diagnostics, initial_guesses, deflation, max_roots])` | $J_z = 0$ 入力 → `allocatable` 配列 | 見つかった異なる物理解をすべて返す |
| `solver%solve_prescribed_field(input, root, status, message [, diagnostics, initial_guesses, deflation, max_roots])` | `prescribed_field_input` → `prescribed_field_result` | 物理解が**ちょうど 1 つ**のとき成功。複数見つかると `SHEATH_AMBIGUOUS_SOLUTION` |
| `solver%solve_prescribed_field_candidates(input, roots, status, message [, diagnostics, initial_guesses, deflation, max_roots])` | `prescribed_field_input` → `allocatable` 配列 | 見つかった候補を選ばずにすべて返す（配列順は順位を意味しない） |
| `solver%solve_profile(input, profile, status, message [, diagnostics, initial_guesses])` | $J_z = 0$ 入力 → `sheath_profile_result` | `solve_equilibrium` に続けて `build_profile` を実行 |
| `solver%build_profile(input, root, profile, status, message)` | $J_z = 0$ 入力 + 既存解 → `sheath_profile_result` | 探索を行わず、元の方程式と物理解判定を再検査してからプロファイルを再構成 |

**省略可能な引数**：

| 引数 | 意味・既定値 |
| --- | --- |
| `diagnostics` | `sheath_search_diagnostics`（呼び出しごとに初期化される） |
| `initial_guesses(:)` | 同じ閉じ方の結果型配列。標準初期値に追加して探索する（**出力変数と同じ変数を渡さないこと**） |
| `deflation` | 既知根を退ける deflation を使うか。既定は $J_z = 0$ 候補探索で `.true.`、$E_H$ 指定で `.false.` |
| `max_roots` | Type ごとの候補数上限。既定 `16`、正の整数 |

**`branch = 'auto'` の試行順**：

| 閉じ方・入力型 | 試行順 |
| --- | --- |
| $J_z = 0$・`zhao_equilibrium_input` | $\alpha < 20^\circ$ で `C → A → B`、それ以外は `A → B → C` |
| $J_z = 0$・`fixed_entry_equilibrium_input` | `A → B → C` |
| $E_H$ 指定 | $E_H \ge 0$ で `A → B → C`、$E_H < 0$ で `C → A → B`（候補探索は全 Type を探索） |

**失敗時の出力と対応範囲**：

- 失敗時は結果型の `valid = .false.`、プロファイル結果の配列成分は未確保です。
- 候補配列は、$J_z = 0$ では**長さ 0 で確保済み**、$E_H$ 指定では**未確保**になります。
- プロファイルと局所密度 API は $J_z = 0$ 解専用です。$E_H$ 指定の根の境界量は、結果型本体・[静的評価](#静的評価)・[`evaluate_matching_conditions`](coupling.md#面上の条件を取り出す) から取得できます。

**使用例（$E_H$ 指定で全候補を取得）**：

```fortran
program field_candidates_example
  use sheath_model
  implicit none
  type(sheath_solver) :: solver
  type(prescribed_field_input) :: input
  type(prescribed_field_result), allocatable :: candidates(:)
  integer(i32) :: status
  integer :: i
  character(len=256) :: message

  input%electron_drift_mps = 0.0_dp
  input%electric_field_v_m = 1.62_dp
  input%emission = maxwellian_emission(55.42562584220407e6_dp, 2.2_dp)

  call solver%solve_prescribed_field_candidates(input, candidates, status, message)
  if (status /= SHEATH_OK) then
    print '(a)', trim(message)
    stop 1
  end if

  do i = 1, size(candidates)
    print '(a,1x,a,2(1x,es14.6))', 'Type', candidates(i)%branch, &
        candidates(i)%boundary_potential_v, candidates(i)%net_current_a_m2
  end do
end program field_candidates_example
```

---

<a id="結果"></a>

## 4. 結果型と局所密度

<a id="j0-の結果"></a>

### $J_z = 0$ の結果（`sheath_equilibrium_result`）

| 成分 | 単位・意味 |
| --- | --- |
| `valid`, `branch` | 有効な解か、Type（`'A'` / `'B'` / `'C'`） |
| `surface_potential_v`, `minimum_potential_v` | $\phi_H$、$\phi_{\min}$ [V]（B は $\phi_{\min} = 0$、C は $\phi_{\min} = \phi_H$） |
| `ambient_electron_density_m3` | 背景電子の規格化 $N_e$ [$\mathrm{m^{-3}}$]（上流の全電子密度ではない） |
| `debye_length_m` | 内部の長さ尺度 $\sqrt{\varepsilon_0 V_s / (e n_s)}$ [m]（[規格化](numerics.md#規格化)） |
| `residual_norm` | 規格化残差の最大絶対値（無次元） |
| `electron_inward_flux_m2_s`, `ion_inward_flux_m2_s` | 境界へ入る電子・イオンの数流束 [$\mathrm{m^{-2}\,s^{-1}}$] |
| `emitted_escape_flux_m2_s` | 上流へ脱出する放出電子の数流束 [$\mathrm{m^{-2}\,s^{-1}}$] |
| `net_current_a_m2` | $+z$ 向きの通常電流 [$\mathrm{A/m^2}$]（$J_z = 0$ 解のためほぼ $0$） |
| `upstream_negative_band_v`, `upstream_field_deficit_v_m` | 受理した上流帯の幅 [V] と場の不足 [$\mathrm{V/m}$]（正ドリフト A / C 受理時のみ正） |

境界電場 $E_H$ は [静的評価](#静的評価) または [`evaluate_matching_conditions`](coupling.md#面上の条件を取り出す) で取得します。

<a id="e_h-指定の結果"></a>

### $E_H$ 指定の結果（`prescribed_field_result`）

| 成分 | 単位・意味 |
| --- | --- |
| `valid`, `branch` | 有効な解か、Type |
| `boundary_potential_v`, `minimum_potential_v` | $\phi_H$、$\phi_{\min}$ [V] |
| `ambient_electron_density_m3` | 背景電子の規格化 $N_e$ [$\mathrm{m^{-3}}$] |
| `electron_inward_flux_m2_s`, `ion_inward_flux_m2_s` | 境界へ入る電子・イオンの数流束 [$\mathrm{m^{-2}\,s^{-1}}$] |
| `emitted_outward_flux_m2_s`, `emitted_escape_flux_m2_s`, `emitted_return_flux_m2_s` | 放出電子の外向き・脱出・戻りの数流束 [$\mathrm{m^{-2}\,s^{-1}}$] |
| `net_current_a_m2` | $+z$ 向きの通常電流 [$\mathrm{A/m^2}$]（閉じ方として固定しない応答） |
| `residual_norm`, `minimum_field_squared_hat` | 規格化残差、経路上の無次元 $E^2$ 最小値 |
| `nonlinear_iterations` | 非線形反復の回数（`integer(i32)`） |
| `upstream_negative_band_v`, `upstream_field_deficit_v_m` | 受理した上流帯の幅 [V] と場の不足 [$\mathrm{V/m}$] |

<a id="プロファイル"></a>

### プロファイル（`sheath_profile_result`）

| 成分 | 内容 |
| --- | --- |
| `equilibrium` | 再構成に用いた `sheath_equilibrium_result` |
| `z_m(:)`, `potential_v(:)`, `electric_field_v_m(:)` | 境界（$z = 0$）から上流へ向かう高さ [m]・電位 [V]・電場 [$\mathrm{V/m}$]（非一様格子） |
| `density(:)` | 各格子点の `sheath_density_result` |
| `turning_height_m` | Type A の極小高さ [m]（Type B / C は `-1.0`） |

プロファイルは `potential_cutoff_v` または `max_distance_m` で停止し、その先へ人工的な $0\text{ V}$ 区間を付け足しません。受理した上流帯がある根では、帯の端から `upstream_band_margin` の割合だけ手前で停止します（帯が狭い根で遠方電位まで欲しい場合は `upstream_band_margin` を小さくしてください）。

<a id="局所密度"></a>

### 局所密度（`evaluate_density` / `sheath_density_result`）

```fortran
call evaluate_density(input, root, potential_v, density, status, message, side)
```

$J_z = 0$ の解 `root` とそれを解いた `input` を渡し、指定電位 `potential_v` [V] での局所密度を評価します。Type A では `side = 'lower'`（境界〜極小）または `'upper'`（極小〜上流）が必須で、B / C では省略できます。電位は選択区間の内側である必要があります。

`sheath_density_result` の成分：

- `ion_m3`：イオン密度 [$\mathrm{m^{-3}}$]
- `electron_free_m3`：境界へ到達する背景電子密度 [$\mathrm{m^{-3}}$]
- `electron_reflected_m3`：反射する背景電子密度 [$\mathrm{m^{-3}}$]
- `emitted_free_m3`：上流へ脱出する放出電子密度 [$\mathrm{m^{-3}}$]
- `emitted_captured_m3`：境界へ戻る放出電子密度（往復の合計） [$\mathrm{m^{-3}}$]
- `charge_c_m3`：正味電荷密度 [$\mathrm{C/m^3}$]

---

<a id="静的評価"></a>

## 5. 静的評価（`evaluate_sheath_state`）

$J_z = 0$ や $E_H$ 指定などの閉じ方を課さず、**与えた試行電位での状態（符号付き $E^2$、接続残差、電流、流束）を直接評価する** 無状態 API です。`sheath_solver` は不要で、時間発展する外側の方程式や利用側独自の境界条件へ組み込む用途に使います。

```fortran
call evaluate_sheath_state(plasma, branch, boundary_potential_v, state, status, message &
    [, minimum_potential_v, electron_normalization_m3])
```

| `branch` | 必要な電位引数 [V] | 条件（`'auto'` は不可） |
| :---: | --- | --- |
| `'B'` | `boundary_potential_v` ($\phi_H \ge 0$) | $\phi_{\min} = 0$。平坦端点 $\phi_H = 0$ も評価可能。`minimum_potential_v` は指定不可 |
| `'C'` | `boundary_potential_v` ($\phi_H < 0$) | $\phi_{\min} = \phi_H$。`minimum_potential_v` は指定不可 |
| `'A'` | `boundary_potential_v` ($\phi_H$) と `minimum_potential_v` ($\phi_m$) | $\phi_m < \min(\phi_H, 0)$。`minimum_potential_v` は**必須** |

$N_e$ は与えた電位での上流準中性から自動で求めます。

| `sheath_state_result` の成分 | 単位・意味 |
| --- | --- |
| `evaluated` | 有限値としての評価に成功したか |
| `admissible`, `physical_status`, `physical_message` | 物理解の判定（Type A は上流接続も含む）に合格したか、そのステータスと理由 |
| `branch`, `boundary_potential_v`, `minimum_potential_v` | 評価した Type と電位 [V] |
| `ambient_electron_density_m3` | 背景電子の規格化 $N_e$ [$\mathrm{m^{-3}}$] |
| `neutrality_residual_m3` | 上流での電荷密度差 $n_i - n_e - n_{\mathrm{em}}$ [$\mathrm{m^{-3}}$] |
| `boundary_field_squared_v2_m2` | **符号を保持した**境界の $E^2$ [$\mathrm{V^2/m^2}$]（負でも 0 に丸めない） |
| `electric_field_v_m` | Type の符号を持つ境界電場 [$\mathrm{V/m}$]（**$E^2 < 0$ のときは `NaN`**） |
| `connection_residual_v2_m2` | Type A の $-(2/\varepsilon_0)\int_{\phi_m}^{0} \rho_{\mathrm{upper}}\,d\phi$ [$\mathrm{V^2/m^2}$]（B / C は `0`） |
| `connection_residual_normalized` | 同じ接続残差の無次元量 $R_A$（下式、B / C は `0`） |
| `electron_inward_flux_m2_s`, `ion_inward_flux_m2_s` | 境界へ入る電子・イオンの数流束 [$\mathrm{m^{-2}\,s^{-1}}$] |
| `emitted_outward_flux_m2_s`, `emitted_escape_flux_m2_s`, `emitted_return_flux_m2_s` | 放出電子の外向き・脱出・戻りの数流束 [$\mathrm{m^{-2}\,s^{-1}}$] |
| `net_current_a_m2` | $+z$ 向きの通常電流 [$\mathrm{A/m^2}$] |

> [!WARNING]
> **`status == SHEATH_OK` は「試行状態を有限値として評価できた」ことを意味し、物理的な完成解であることは意味しません。**
> - `admissible = .false.` でも、符号付き `boundary_field_squared_v2_m2` や残差は外側の方程式を解くのに利用できます。
> - `electric_field_v_m` を使う前に、必ず `state%admissible` または `state%boundary_field_squared_v2_m2 >= 0.0_dp` を確認してください。

**Type A の規格化接続残差**：電位尺度 $V_s = \text{\texttt{plasma\%emission\%potential\_scale(plasma\%electron\_temperature\_ev)}}$ に対して

$$
R_A = \frac{\text{\texttt{connection\_residual\_v2\_m2}} \cdot \varepsilon_0 \sqrt{V_s}}{e\,n_{i\infty}\,(-\phi_m)^{3/2}}
$$

です（浅い極小で積分区間が縮むだけで残差が小さく見える効果を $(-\phi_m)^{3/2}$ で除いています）。

**高度な引数 `electron_normalization_m3`**：急峻な bin スペクトルの端点等で、外側方程式の根が隣接する binary64 電位の間に落ちる場合のための引数です（通常は省略）。Type B / C（かつ $\phi_H \ne 0$）でのみ指定でき、**現在電位の直前・直後の binary64 値における中性密度 $N_e$ の間**に厳密に制限されます（範囲外は `SHEATH_INVALID_ARGUMENT`）。指定した $N_e$ で密度・電場・電流を一貫して再計算し、準中性残差と物理解判定の許容値は一切緩めません。

**使用例**：

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

---

<a id="解マップと-continuation"></a>

## 6. 解マップと continuation

解マップ（Atlas）は、受理済みの物理解を無次元座標で記録する呼び出し側所有の表です。問い合わせ時の初期値生成と continuation の出発点にのみ使い、**問い合わせでは毎回元の方程式を解き直します**（解マップを自動更新しません）。

- `sheath_equilibrium_atlas`：$J_z = 0$ 用（6 次元キー、Python の `EquilibriumAtlas` とテキスト形式を共有）
- `sheath_field_atlas`：$E_H$ 指定用（$E_H$ を加えた 7 次元キー）

```fortran
call solver%build_equilibrium_atlas(inputs, atlas, status, message [, report, deflation])
call solver%build_field_atlas(inputs, field_atlas, status, message [, report, deflation, max_roots])

solver%equilibrium_atlas = atlas   ! 以後の求解で初期値・continuation に使用
call solver%solve_equilibrium(query, root, status, message, diagnostics)
```

**`sheath_solver` の解マップ構築・登録メソッド**：

| メソッド | 動作 |
| --- | --- |
| `solver%build_equilibrium_atlas(inputs, atlas, status, message [, report, deflation])` | 入力配列を解いて既存の表に追加し、後から見つかった近傍解で未解決点を再探索する。`deflation` 既定は `.false.` |
| `solver%build_field_atlas(inputs, atlas, status, message [, report, deflation, max_roots])` | $E_H$ 指定版の表を構築・追加する |
| `solver%add_equilibrium_to_atlas(input, root, atlas, status, message [, component])` | 既存の $J_z = 0$ 解を元の方程式・物理解判定で検査してから登録する |
| `solver%add_field_to_atlas(input, root, atlas, status, message [, component])` | 既存の $E_H$ 指定解を同様に検査してから登録する |

- `report(3, n)`（`integer(i32), allocatable`）は Type（A/B/C）$\times$ 入力ごとのステータスを返します（指定しなかった Type は `SHEATH_INVALID_ARGUMENT`）。
- 構築の `SHEATH_OK` は有限の探索が完了したことを意味し、全入力で解が見つかったことは保証しません。
- `component` は Type 内の解の族を区別する正の整数です（省略時は近傍距離で自動付与。数学的な連結性の証明ではありません）。異なる bin スペクトル形状の記録は問い合わせに使用しません。

**解マップ型の操作**：

| 操作・成分 | 意味 |
| --- | --- |
| `atlas%size()` | 登録点数 |
| `atlas%point(i)` | $i$ 番目の `sheath_atlas_point`（成分：`branch`, `component`, `key(:)`, `coordinates(3)`, `spectrum_shape(:)`） |
| `atlas%clear()` | 全登録を削除 |
| `atlas%write(unit, iostat)` / `atlas%read(unit, iostat)` | 版付きテキスト形式での入出力（`unit` は `formatted` で開く）。読み込み失敗時は元の表を変更しない |
| `atlas%options` | `sheath_atlas_options`（既定：`neighbors = 8`, `max_distance = 1.0`, `interpolate = .true.`, `component_distance = 0.75`） |

実例は [example/solution_atlas.f90](../example/solution_atlas.f90) および [example/field_atlas.f90](../example/field_atlas.f90) を参照してください。

---

<a id="探索の診断"></a>

## 7. 探索の診断（`sheath_search_diagnostics`）

すべての成分は **A / B / C 順の長さ 3 の配列** です。

| 成分 | 意味 |
| --- | --- |
| `searched(3)`, `excluded(3)` | その Type を探索したか、解析条件で除外したか |
| `starts(3)`, `unconverged(3)`, `rejected(3)`, `profile_failures(3)` | 初期値試行数、未収束数、物理条件による棄却数、プロファイル評価失敗数 |
| `roots_found(3)`, `best_residual(3)` | 見つかった根の数、観測した最小残差 |
| `evaluations(3)`, `iterations(3)`, `lm_steps(3)`, `brackets(3)` | 残差評価回数、反復回数、LM ステップ数、括弧付け回数 |
| `atlas_starts(3)`, `atlas_hits(3)`, `continuation_steps(3)`, `continuation_retries(3)`, `deflations(3)` | 解マップ・continuation・deflation の利用回数 |
| `subdivisions(3)`, `tangencies(3)`, `invalid_evaluations(3)` | スカラー探索の細分化回数・接根検出数・無効評価点数 |

採用解が見つかった場合でも、別初期値での未収束回数はそのまま残ります。採用解がなく棄却と未収束が混在する場合は `SHEATH_NUMERICAL_FAILURE` です。

---

<a id="補助関数と定数"></a>

## 8. 補助関数と定数

| 名前 | 内容 |
| --- | --- |
| `ion_density_ratio(potential_v, entry_energy_ev, pressure_energy_ev)` | 超音速イオン流体密度比 $n_i / n_{i\infty}$。`entry_energy_ev` $= m_i u_{i\infty}^2 / (2e)$、`pressure_energy_ev` $= (\text{\texttt{ion\_pressure\_factor}})\cdot T_i$。到達不可の電位では `NaN` |
| `ion_critical_potential(entry_energy_ev, pressure_energy_ev)` | イオンが越えられる最大電位 $\phi_{\mathrm{crit}}$ [V]。入口が熱音速以下なら `NaN` |
| `plane_*`, `log_energy_*`, `energy_bin`, `observed_outflow_source`, `LOG_ENERGY_DEFAULT_*` | シミュレータ結合用の判定・流束・スペクトル集計関数と定数（[シミュレータとの結合](coupling.md)） |
| `dp`, `i32` | `real64`、`int32` の kind 定数 |
| `SHEATH_OK`, `SHEATH_INVALID_ARGUMENT`, `SHEATH_NO_PHYSICAL_SOLUTION`, `SHEATH_NUMERICAL_FAILURE`, `SHEATH_AMBIGUOUS_SOLUTION` | ステータス定数（[共通の規約](conventions.md#ステータスとエラー処理)） |
