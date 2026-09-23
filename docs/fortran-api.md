# Fortran API

この文書は軌道保存・物理解判定を修正したソース版の API です（v0.1.0 とは解の存在域が変わります）。
詳細な導出と適用範囲は [運動論モデル](kinetic-model.md) を参照してください。

`use sheath_model` が公開窓口です。`src/internal/` は実装詳細です。
実数は `real(dp)`（real64）、ステータスは `integer(i32)`（int32）を使います。
単位は SI、温度のみ eV。上流の `phi(infinity)=0 V` を電位基準とし、+z は境界から上流へ向かいます。
電場は `E=-dphi/dz`、粒子数流束は指定方向への正の大きさです。

## モデルと呼び出し方

| 関数 | 条件 | 入力型 | 結果型 |
| --- | --- | --- | --- |
| `solve_equilibrium(input, result, status, message)` | J=0 | `zhao_equilibrium_input` | `zhao_equilibrium_result` |
| `solve_prescribed_field(input, result, status, message)` | E_H 指定 | `zhao_field_input` | `zhao_field_result` |

両方とも平面・一次元・半無限領域の定常シースです。冷たいイオンビーム、ドリフト Maxwell 電子、Maxwell 光電子源を仮定します。
J=0 は上流準中性・零電流条件から解き、A は極小から上流の電場条件も満たします。
両モデルとも、代数根に加えてイオン到達条件・実数電場の接続・上流漸近条件を検査します。
背景電子の内向きドリフトが正で反射低速集団を持つ A/C は、半無限上流条件に接続できないため不採用です。
E_H 指定では零電流式を電場条件に置き換え、正味電流を出力します。
A の電場条件は `E_H²=-(2/epsilon_0)*integral(phi_m..phi_H, rho_lower dphi)`、
B/C は `E_H²=(2/epsilon_0)*integral(phi_H..0, rho dphi)` です。
A の上側領域では `E(infinity)=0` を満たす極小電位を求めます。

型に値を設定して関数を呼ぶだけで、初期化や前回解の管理は不要です。結果は毎回上書きされます。
成功は `status == sheath_ok` で判定し、失敗時は `result%valid=.false.` になります。

```fortran
use sheath_model
implicit none
type(zhao_equilibrium_input) :: stationary_input
type(zhao_equilibrium_result) :: stationary
type(zhao_field_input) :: field_input
type(zhao_field_result) :: field_solution
integer(i32) :: status
character(len=256) :: message

stationary_input%branch = 'A'
stationary_input%electron_drift_mode = 'zero'
call solve_equilibrium(stationary_input, stationary, status, message)
if (status /= sheath_ok) stop 1

field_input%branch = 'A'
field_input%electron_drift_mps = 0.0_dp
field_input%root_selection = 'max_field_energy'
field_input%electric_field_v_m = 1.62_dp
field_input%photoelectron_source_density_m3 = 5.5425625842204072e7_dp
call solve_prescribed_field(field_input, field_solution, status, message)
if (status /= sheath_ok) stop 1
```

## J=0 の入力: `zhao_equilibrium_input`

| フィールド | 既定値 | 意味・制約 |
| --- | --- | --- |
| `branch` | `'auto'` | A/B/C/auto。大文字小文字は不問 |
| `sun_elevation_deg` | 60 | 太陽高度、0〜90 度 |
| `ion_density_m3` | 8.7e6 | 上流イオン密度、正 |
| `photoelectron_reference_density_m3` | 64e6 | 垂直入射時の光電子源規格化密度、正 |
| `electron_temperature_ev` | 12 | 上流電子温度、正 |
| `photoelectron_temperature_ev` | 2.2 | 光電子温度、正 |
| `solar_wind_speed_mps` | 468e3 | 全太陽風速度、正 |
| `ion_mass_kg` | 1.67262192369e-27 | イオン質量、正 |
| `electron_mass_kg` | 9.1093837015e-31 | 電子質量、正 |
| `electron_drift_mode` | `'normal'` | normal: `v_sw sin(alpha)`、full: 全速度、zero: 無ドリフト背景電子 |
| `ion_drift_mode` | `'normal'` | normal: `v_sw sin(alpha)`、full: 全速度 |

すべての数値は有限値が必要です。光電子源密度は `n_phe0=n_phe_ref sin(alpha)`。
法線イオン速度ゼロはモデルが未定義なので入力エラーです。
`auto` は高度 20 度未満で C→A→B、それ以外では A→B→C の順に探索し、プロファイルの物理条件も満たす最初の枝を返します。

## E_H 指定の入力: `zhao_field_input`

| フィールド | 既定値 | 意味・制約 |
| --- | --- | --- |
| `branch` | `'auto'` | A/B/C/auto。大文字小文字は不問 |
| `root_selection` | `'require_unique'` | 物理解が複数ある場合の選択方針 |
| `electric_field_v_m` | 0 | 指定する法線電場 E_H [V/m]。+z 向きが正 |
| `ion_density_m3` | 8.7e6 | 上流イオン密度、正 |
| `photoelectron_source_density_m3` | 0 | 境界での光電子源規格化密度 n_phe0、非負 |
| `electron_temperature_ev` | 12 | 上流電子温度、正 |
| `photoelectron_temperature_ev` | 2.2 | 光電子温度、正 |
| `electron_drift_mps` | 405299.88897111727 | 内向きを正とする電子の法線ドリフト |
| `ion_drift_mps` | 405299.88897111727 | 内向きのイオンビーム速度、正 |
| `ion_mass_kg`, `electron_mass_kg` | J=0 と同じ | 正の質量 |

すべての数値は有限値が必要です。光電子源ゼロでも温度は正を指定します。
ドリフトは法線方向へ射影済みの値を渡します。絶対高度 H は平面モデルの方程式に入りません。
光電子源密度は半 Maxwell 分布の規格化密度で、総放出流束との関係は
`Gamma_pe,emit=n_phe0*sqrt(2 e T_pe/m_e)/(2 sqrt(pi))` です。局所の光電子全密度とは異なります。

BEACH の `zhao_online` と同じく、背景電子・イオンの外向き流束から速度分布を再構成するモデルではありません。
BEACH の共通インタフェースにある `electron_outward_flux_m2_s` と `ion_outward_flux_m2_s` は
Zhao のシース解に作用しないため、本 API の入力には含めていません。

この API は上流イオン状態と光電子源を固定し、背景電子 Maxwell 分布の規格化を準中性から解きます。
背景電子 VDF の規格化を固定した応答ではなく、E_H の変更に伴い規格化密度も変わり得ます。
`fixed_ambient` を同じ半無限条件へ追加すると一般に過剰決定になるため、別モードは設けていません。

根の選択方針は二つです。

- `require_unique`: 検出された物理解が一意なら採用。複数なら `sheath_ambiguous_solution`。
- `max_field_energy`: 正の電場エネルギー `+(epsilon_0/2)*integral(E² dz)` が最大の候補を選択。数値的な同順位は曖昧性を返す。

後者はヒューリスティックであり、安定性を意味しません。

`solve_prescribed_field_candidates(input, results, status, message)` は
`type(zhao_field_result), allocatable :: results(:)` に発見した物理候補を返します。
`root_selection` による選択は行いません。失敗時は配列を未確保に戻します。
ゼロ電場でも非平坦解を探索し、平坦解は正の電子規格化密度を持つ場合だけ候補に加えます。
A の極小が境界へ合流したゼロ電場端点は C として一度だけ返します。

複数の物理解を扱うための指定であり、有限個の初期値による探索で全パラメータ域の存在・一意性を保証するものではありません。
探索は指定枝に限定します。ただし A/C のゼロ電場端点は共通の C 表現にまとめます。

## 結果

両結果型に共通する物理量です。

| フィールド | 単位 | 意味 |
| --- | --- | --- |
| `valid`, `branch` | — | 成功状態と採用枝 |
| `minimum_potential_v` | V | 領域全体の最小電位。A は内部極小、B は遠方の 0、C は境界値 |
| `ambient_electron_density_m3` | m⁻³ | 上流 Maxwell 電子集団の規格化密度 N_e（実際の上流総電子密度ではない） |
| `electron_inward_flux_m2_s` | m⁻² s⁻¹ | 境界に到達する電子流束 |
| `ion_inward_flux_m2_s` | m⁻² s⁻¹ | 冷たいイオン流束 n_i u_i |
| `photoelectron_escape_flux_m2_s` | m⁻² s⁻¹ | 戻り光電子を除いた上流到達流束 |
| `net_current_a_m2` | A/m² | +z 向きの通常の電流 J_z |
| `residual_norm` | 無次元 | 解いた方程式の規格化残差の最大絶対値 |

`J_z=e*(Gamma_e,in-Gamma_i,in-Gamma_pe,escape)` です。表面に正電荷を蓄積させる電流は `-J_z`。
J=0 モデルの電流は数値誤差の範囲でゼロ、E_H 指定モデルでは条件に応じた非ゼロ値も返します。

`zhao_equilibrium_result` は追加で `surface_potential_v`（表面電位）と `debye_length_m`（光電子参照密度・温度による Debye 長）を持ちます。
`zhao_field_result` は追加で `boundary_potential_v`（指定電場の位置の電位）、`nonlinear_iterations`（反復数）、
`minimum_field_squared_hat`（確認した経路上の最小無次元電場二乗）、`field_energy_j_m2`（正の静電場エネルギー）を持ちます。
E_H 側は密度を n_i、電位を T_pe、長さを `sqrt(epsilon_0 T_pe/(n_i e))` で規格化します。
成功した候補の電場エネルギーは選択方針によらず計算します。エネルギー・残差・最小電場二乗は未計算時に `huge()` です。
数値ゼロと失敗を区別するため、必ず `status` を確認してください。

## J=0 解の密度・プロファイル

`evaluate_density(input, solution, potential_v, density, status, message, side)` には、解いたときと同じ `zhao_equilibrium_input` と結果を渡します。
A は `side='lower'`（表面〜極小）または `'upper'`（極小〜上流）が必須、B/C は省略できます。
許される電位範囲は A lower が `[phi_m,phi0]`、A upper が `[phi_m,0]`、B が `[0,phi0]`、C が `[phi0,0]`。
`zhao_density_result` は `ion_m3`, `electron_free_m3`, `electron_reflected_m3`,
`photoelectron_free_m3`, `photoelectron_captured_m3`, `charge_c_m3` を返します。

`solve_profile(input, options, profile, status, message)` は J=0 の半無限プロファイルを再構成します。
`zhao_profile_options` の設定は次の三つです。

- `points_per_segment=4000`: 電位空間の積分点数、32 以上。A では上下それぞれに使用。
- `max_distance_m=100`: 返却する高さの上限、有限かつ正。
- `potential_cutoff_v=0.0022`: 遠方のゼロ電位手前での打ち切りの目安、有限かつ正。枝の電位振幅の半分を上限として調整。

`zhao_profile_result` の `z_m(:)`, `potential_v(:)`, `electric_field_v_m(:)`, `density(:)` が同じ非一様格子上の結果です。
`equilibrium` は J=0 解、`turning_height_m` は A の極小高度（B/C は -1）。
高さは 0 から厳密に増加し、A の極小点は一度だけ格納します。遠方に人工的なゼロ電位の尾部は追加しません。
実際の返却範囲は `z_m(size(z_m))` で確認してください。高さ上限内に 2 点未満しかない場合は入力エラーです。
失敗時は配列を未確保に戻します。実数の電場を構成できない代数解は `solve_equilibrium` の時点で不採用です。
負の表面電位を持つ A でも lower は `[phi_m,phi0]`、upper は `[phi_m,0]` です。
Python の B/C も同じ半無限条件の一次積分です。`n_profile_grid` は B/C の電位格子の点数に使い、
`profile_phi_tol_hat` は全枝で遠方の打ち切りに使います。返却する格子は非一様で、`zmax_hat` まで必ず到達するとは限りません。

## ステータス

| 定数 | 値 | 意味 |
| --- | --- | --- |
| `sheath_ok` | 0 | 成功 |
| `sheath_invalid_argument` | 1 | 不正な入力、設定、評価範囲 |
| `sheath_no_physical_solution` | 2 | 指定電場・枝で物理解が成立しない |
| `sheath_numerical_failure` | 3 | 収束失敗、非有限値、プロファイル積分失敗等 |
| `sheath_ambiguous_solution` | 4 | 複数解を指定方針では選べない |

`message` は呼び出し側の文字列へ書き込みます（256 文字以上を推奨）。成功時にも縮退解の説明が入る場合があります。
数値探索失敗は物理解の不存在を証明しません。ライブラリはログ出力・ファイル操作・プロセス終了を行いません。
