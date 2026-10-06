# Fortran API

公開窓口は `use sheath_model` です。物理条件を入力型、数値設定を `sheath_solver`、
計算結果を結果型に分けています。Solver は前回解を保持せず、求解ごとに物理条件を渡します。
実数は `real(dp)`、ステータスは `integer(i32)`。SI 単位を使い、温度・光電子の法線運動エネルギーは eV です。
上流電位は 0 V、+z は表面から上流向き、電場は `E=-dphi/dz`、法線入口速度は内向きが正です。

## 求解とプロファイル

```fortran
use sheath_model
implicit none
type(sheath_solver) :: solver
type(fixed_entry_equilibrium_input) :: input
type(sheath_equilibrium_result) :: root
type(sheath_profile_result) :: profile
type(sheath_search_diagnostics) :: diagnostics
integer(i32) :: status
character(len=256) :: message

input%branch = 'A'
input%plasma%electron_drift_mps = 0.0_dp
input%plasma%photoelectrons = maxwellian_photoelectrons(55.42562584220407e6_dp, 2.2_dp)
input%plasma%ion_temperature_ev = 12.0_dp
input%plasma%ion_pressure_factor = 3.0_dp
solver%search%method = 'newton'
solver%profile%max_distance_m = 60.0_dp

call solver%solve_equilibrium(input, root, status, message, diagnostics)
if (status /= SHEATH_OK) stop 1
call solver%build_profile(input, root, profile, status, message)
if (status /= SHEATH_OK) stop 1
print *, root%surface_potential_v, profile%electric_field_v_m(1)
```

`build_profile` は既存解を元の方程式と物理条件で検査して再構成します。
まとめて実行する場合は `solver%solve_profile(input, profile, status, message, diagnostics)` を使います。
探索とプロファイルの数値設定は独立しており、入口速度に Bohm 補正を加える操作はありません。

| Solver のメソッド | 入力・出力 | 意味 |
| --- | --- | --- |
| `solve_equilibrium(input, root, status, message, diagnostics, initial_guesses)` | J=0 入力 → 単一解 | 指定 Type、または auto の順序で最初に採用した解 |
| `solve_equilibrium_candidates(input, roots, status, message, diagnostics, initial_guesses, deflation, max_roots)` | J=0 入力 → allocatable 配列 | 有限探索で発見した異なる物理解 |
| `solve_prescribed_field(input, root, status, message, diagnostics, initial_guesses, deflation, max_roots)` | 電場固定入力 → 単一解 | 発見した候補が一つの場合に返す |
| `solve_prescribed_field_candidates(input, roots, status, message, diagnostics, initial_guesses, deflation, max_roots)` | 電場固定入力 → allocatable 配列 | 複数候補を選択せず返す |
| `solve_profile(input, profile, status, message, diagnostics, initial_guesses)` | J=0 入力 → プロファイル | 求解してから再構成 |
| `build_profile(input, root, profile, status, message)` | J=0 入力と既存解 → プロファイル | 解探索を繰り返さず再構成 |

`diagnostics` 以降の引数は optional。`initial_guesses(:)` は各 closure の結果型の配列で、
標準初期値に追加されます。出力配列と同じ変数を渡さないでください。
deflation の既定値は J=0 候補探索で true、電場固定で false。`max_roots` は各 Type の上限で既定値 16 です。

## 物理入力

`plasma_input` は入口状態と光電子源を固定する型です。
`fixed_entry_equilibrium_input%plasma` にこれを渡すと J=0、
`prescribed_field_input` に同じ値を渡すと電場固定を使えます。
どちらも Maxwell 源と bin スペクトルに対応します。

| 共通フィールド | 既定値 | 制約・意味 |
| --- | --- | --- |
| `ion_density_m3` | 8.7e6 | 正、上流イオン密度 |
| `ion_entry_speed_mps` | 405299.88897111727 | 正、内向き法線入口速度 |
| `electron_temperature_ev` | 12 | 正、背景電子温度 |
| `electron_drift_mps` | 405299.88897111727 | 内向き法線ドリフト。固定入口 J=0 の plasma 既定値は 0 |
| `ion_temperature_ev` | 0 | 非負、0 で冷たいイオン |
| `ion_pressure_factor` | 1 | 正、イオン流体圧力項の係数 |
| `ion_mass_kg`, `electron_mass_kg` | 陽子・電子質量 | 正 |
| `photoelectrons` | 密度 0、温度 2.2 eV の Maxwell 源 | [源の指定と制約](spectral-api.md#光電子源を指定する) |

すべて有限値が必要です。温かいイオンでは `m_i u_i²/e > ion_pressure_factor*T_i` が必要です。
背景電子の規格化密度は入力で固定せず、準中性条件と closure から解きます。

`fixed_entry_equilibrium_input` は `branch='auto'` と `plasma` を持ちます。
`prescribed_field_input` は `plasma_input` を継承し、`branch='auto'` と `electric_field_v_m=0` を追加します。
`field%plasma_input = plasma` で共通状態を設定できます。電場固定では正味電流を出力し、J=0 を課しません。

太陽風と照射角から入口を指定する場合は `zhao_equilibrium_input` を使います。
既定値は高度 60 度、イオン密度 8.7e6 m⁻³、垂直入射光電子密度 64e6 m⁻³、
電子温度 12 eV、光電子温度 2.2 eV、イオン温度 0、風速 468e3 m/s です。
`electron_drift_mode` は normal/full/zero、`ion_drift_mode` は normal/full、いずれも既定値 normal。
normal は風速に `sin(sun_elevation_deg)` を掛け、光電子源密度にも同じ照射係数を掛けます。
入力角度は 0〜90 度、法線イオン速度ゼロは入力エラーです。

branch は A/B/C/auto、大文字小文字は不問です。J=0 の auto は Zhao 入力の高度 20 度未満で C→A→B、
その他で A→B→C の順に探索します。複数解の比較には候補探索を使ってください。

## 数値設定と解マップ

Solver の `search`、`continuation`、`profile` に設定します。

| `search` の設定 | 既定値 | 意味 |
| --- | --- | --- |
| `method` | auto | auto / newton / lm。J=0 の明示した B/C のみ bracket も可 |
| `residual_tolerance` | 1e-10 | 元の規格化方程式の最大絶対残差 |
| `max_iterations`, `max_backtracks`, `max_starts` | 100, 24, 32 | 有限探索の反復・減速・開始点予算 |
| `use_default_guesses` | true | 標準初期値を追加 |
| `bracket_points`, `potential_extent` | 96, 200 | bracket 格子数、規格化電位の探索範囲 |

`continuation%method` は parameter（既定）または arclength。
初期・最小・最大刻みは 0.25 / 1e-4 / 0.5、`max_steps=128`、`max_root_distance=0.75` です。
profile は `points_per_segment=4000`、`max_distance_m=100`、`potential_cutoff_v=2.2e-3`。
格子点は 32 以上、距離・cutoff は有限かつ正です。

`sheath_equilibrium_atlas` と `sheath_field_atlas` は closure ごとに分かれた表です。
`solver%build_equilibrium_atlas(inputs, atlas, status, message, report, deflation)`、
`solver%build_field_atlas(inputs, atlas, status, message, report, deflation, max_roots)` で構築します。
`report(3,n)` は A/B/C × 入力の結果で、未指定 Type は INVALID_ARGUMENT。
構築の OK は有限探索の完了であり、全入力の解発見を意味しません。
既存表に追加し、後から見つかった近傍解で未解決点を再探索します。

`solver%equilibrium_atlas = atlas` または `solver%field_atlas = atlas` で探索用の表をコピーして保持します。
不要なら deallocate。求解は表を変更しません。
登録は `solver%add_equilibrium_to_atlas(input, root, atlas, status, message, component)` または
`solver%add_field_to_atlas(...)` で明示し、元の方程式とプロファイルを再検査します。
component は optional の正整数。自動ラベルは近接性の推定で、数学的な枝接続の証明ではありません。

表の `options` は neighbors=8、max_distance=1、interpolate=true、component_distance=0.75。
`atlas%size()`、`%point(i)`、`%clear()`、`%write(unit,iostat)`、`%read(unit,iostat)` を使えます。
J=0 は 6、電場固定は 7 次元の無次元キーを使い、スペクトル形状が異なる表は流用しません。
表は初期値と continuation に使い、対象の方程式を解き直します。

## 結果と診断

`sheath_equilibrium_result` は `valid`、`branch`、`surface_potential_v`、`minimum_potential_v`、
`ambient_electron_density_m3`、`debye_length_m`、`residual_norm`、電子・イオン内向き流束、
光電子 escape 流束、`net_current_a_m2` を返します。電子密度は Maxwell 分布の規格化で、全上流密度とは異なります。
`prescribed_field_result` は表面電位を `boundary_potential_v` で返し、
光電子 outward/return 流束、`nonlinear_iterations`、`minimum_field_squared_hat` も含みます。

`sheath_profile_result` の equilibrium と、同じ格子上の `z_m(:)`、`potential_v(:)`、
`electric_field_v_m(:)`、`density(:)` を使います。`turning_height_m` は A の内部極小位置、B/C では -1。
`sheath_density_result` の単位は m⁻³、`charge_c_m3` は C/m³。
局所密度は `evaluate_density(input, root, potential_v, density, status, message, side)` で評価でき、A は lower/upper を明示します。

`sheath_search_diagnostics` はすべて A/B/C 順の長さ 3 の配列です。
searched/excluded、starts/unconverged/rejected/profile_failures、roots_found、
evaluations/iterations/lm_steps/brackets/best_residual、
atlas_starts/atlas_hits/continuation_steps/continuation_retries/deflations を返します。
呼び出しごとに初期化し、採用解があっても未収束の試行を残します。

成功は `status == SHEATH_OK` で判定します。INVALID_ARGUMENT は入力不正、
NO_PHYSICAL_SOLUTION は探索で物理候補を採用できなかった結果、NUMERICAL_FAILURE は未解決の数値探索です。
電場固定の単一解探索で複数候補を発見した場合は AMBIGUOUS_SOLUTION。
失敗時は valid=false、プロファイルは未確保。J=0 候補配列は長さ 0、電場固定の候補配列は未確保です。
有限探索の成功・失敗から、全根の発見・解の不存在・動的安定性は保証しません。

静的電位評価は Solver を必要としない [スペクトル API](spectral-api.md) を参照してください。
利用例は [examples.md](examples.md)、モデルの適用条件は [kinetic-model.md](kinetic-model.md) にあります。
