# 粒子シミュレータとの結合

粒子シミュレータの計算領域の上端面を、その上に広がる平面・定常の外部シースへつなぐための Fortran API です。
読み終えると、外部解が上端面に課す条件の取り出し方、上端面を外へ出た電子から外部解を更新する方法、
その状態を checkpoint に保存する方法が分かります。粒子種、メッシュ・幾何、面積、粒子重み、MPI は扱わず、
条件をどう課すかは利用側が決めます。

## 考え方

- **matching plane**: 結合する上端面。外部モデルはこの面を自らの境界 z=0 とみなし、上方を 1 次元・定常・無衝突として解きます。
  面より下の横方向の構造が、外部シースから見て一様な壁として扱える高さに置いてください。
- **電位**: 上流を 0 V とし、面の電位 φ_H は外部根の境界電位です。+z は面から上流向き、電場は `E=-dphi/dz` です。
- **粒子集団**: 外部モデルは、上流の drift Maxwell 電子、上流のイオンビーム、面から外へ出る放出電子の 3 集団で構成します。
  放出電子は由来を問いません。光電子、二次電子、領域内で反射して面を外へ抜けた上流電子を、すべて放出源に入れます。
- **零電流条件**: J=0 の根では、面を通る正味電流が 0 になります。面へ入る上流電子束は外部モデルでは吸収として数え、
  反射して出ていく電子は放出電子の escape として数えます。

## 面上の条件を取り出す

`evaluate_matching_conditions` は、J=0 の根または E_H 指定の根から `sheath_matching_conditions` を作ります。
探索はせず、根の電位と電子規格化を、解いたときの入力で評価し直します。束は求解と同じ式で計算します。

```fortran
use sheath_model
type(sheath_solver) :: solver
type(fixed_entry_equilibrium_input) :: input
type(sheath_equilibrium_result) :: root
type(sheath_matching_conditions) :: plane
integer(i32) :: status
character(len=256) :: message

input%plasma%emission = maxwellian_emission(55.4e6_dp, 2.2_dp)
call solver%solve_equilibrium(input, root, status, message)
if (status /= SHEATH_OK) stop 1
call evaluate_matching_conditions(input, root, plane, status, message)
if (status /= SHEATH_OK) stop 1
print *, plane%branch, plane%plane_potential_v, plane%electron_path_minimum_v
```

E_H 指定では `prescribed_field_input` と `prescribed_field_result` を渡します。
根が無効な場合は `SHEATH_INVALID_ARGUMENT`、束が非有限・負になる場合は `SHEATH_NUMERICAL_FAILURE` を返し、`valid=.false.` です。

| `sheath_matching_conditions` | 単位・意味 |
| --- | --- |
| `valid`, `branch` | 有効な条件か、根の Type（A/B/C） |
| `plane_potential_v` | 面の電位 φ_H [V] |
| `plane_field_v_m` | 面のすぐ上の電場 E(H+) [V/m]。A/B は正、C は負 |
| `electron_path_minimum_v` | 面と上流の間で、面の電位とは別に電子が越える電位 [V]。A は内部極小 φ_m、B/C は 0 |
| `ion_path_maximum_v` | 同じくイオンが越える電位 [V]。常に 0 |
| `electron_reservoir_density_m3`, `electron_temperature_ev`, `electron_drift_mps` | 上流電子 Maxwell 分布の規格化密度 N_e [m⁻³]・温度 [eV]・内向き drift [m/s]。N_e は上流の全電子密度ではない |
| `ion_reservoir_density_m3`, `ion_entry_speed_mps`, `ion_temperature_ev` | 上流イオンの密度 [m⁻³]・内向き入口速度 [m/s]・温度 [eV] |
| `electron_inward_flux_m2_s`, `ion_inward_flux_m2_s` | 面へ入る電子・イオンの数流束 [m⁻² s⁻¹] |
| `emitted_outward_flux_m2_s` | 放出電子の外向き数流束 [m⁻² s⁻¹] |
| `emitted_escape_flux_m2_s`, `emitted_return_flux_m2_s` | そのうち上流へ抜ける分と面へ戻る分 [m⁻² s⁻¹] |
| `net_current_a_m2` | +z 向きの通常電流 `e*(Gamma_e,in-Gamma_i,in-Gamma_em,escape)` [A/m²]。J=0 の根ではほぼ 0 |
| `upstream_negative_band_v`, `upstream_field_deficit_v_m` | drift のある A/C 根を受理した上流の帯 [V] と場の不足 [V/m]。それ以外は 0 |

### 条件の使い方

経路の電位は、利用側が持つ面の局所電位 φ_loc と組み合わせて使います。エネルギーは法線運動エネルギー [eV] です。

| 判定 | 条件 |
| --- | --- |
| 上流の電子が面に届く | 上流での K∞ ≥ −min(φ_loc, `electron_path_minimum_v`, 0) |
| 上流のイオンが面に届く | 上流での K∞ ≥ max(φ_loc, `ion_path_maximum_v`) |
| 面を外へ出た電子が上流へ抜ける | 面での K ≥ φ_loc − min(φ_loc, `electron_path_minimum_v`)。満たさなければ面へ戻る |
| 面を外へ出たイオンが上流へ抜ける | 面での K ≥ max(φ_loc, `ion_path_maximum_v`) − φ_loc |

`min(φ_H, electron_path_minimum_v)` と `max(φ_H, ion_path_maximum_v)` は、外部プロファイル全体の電位の最小値・最大値です。
場の境界には、面の平均電位を `plane_potential_v` に固定する方法と、法線電場を `plane_field_v_m` に合わせる方法があります。
どちらを使うか、また束を目標値として課すか整合の診断に使うかは利用側で選びます。

## 流出スペクトルから放出源を作る

面を外向きに通過した電子を法線運動エネルギーの bin に集計し、`observed_outflow_source` で放出源にします。

```fortran
real(dp), allocatable :: edges(:), weight(:)
type(emission_source) :: source

edges = log_energy_edges()                 ! [0, 1 meV) と 1 meV〜1 keV の 24 bin/decade、計 145 bin
allocate (weight(size(edges) - 1), source=0.0_dp)
! 面を外向きに通過した電子ごとに（K は法線運動エネルギー [eV]、w は物理粒子数）
weight(energy_bin(edges, K)) = weight(energy_bin(edges, K)) + w
! 窓の終わりに、利用側で MPI 合算してから面積と時間で割る
call observed_outflow_source(edges, weight/(area_m2*duration_s), source, status, message)
```

| 手続き | 意味 |
| --- | --- |
| `log_energy_edges(lowest_edge_ev=1e-3, decades=6, bins_per_decade=24)` | 先頭 bin [0, lowest) と対数 bin の境界 [eV]。不正な引数では空配列 |
| `energy_bin(edges_ev, energy_ev)` | エネルギーが属する bin。範囲外と NaN は端の bin、bin がなければ 0 |
| `observed_outflow_source(edges_ev, flux_m2_s, source, status, message)` | bin ごとの外向き数流束 [m⁻² s⁻¹] から bin 放出源を作る。空の端 bin は除き、内部の空 bin は残す |

流束は有限・非負で、合計が正でなければなりません。満たさない場合は `SHEATH_INVALID_ARGUMENT` です。
bin 内は `dGamma/dK` 一定として扱います（[放出源の指定](spectral-api.md#放出源を指定する)）。

## 外部解を更新する

`sheath_matching_iteration` は J=0 の外部根を保持し、観測した放出源で解き直します。
上流の plasma と solver の設定は利用側の設定から毎回渡し、状態には含めません。

```fortran
type(sheath_matching_iteration) :: outer
type(plasma_input) :: plasma

plasma%emission = maxwellian_emission(n_em, 2.2_dp)   ! 初回の放出源
call outer%initialize(plasma, solver, status, message, initial_order='CAB')
plane = outer%conditions()
! 窓ごとに
call outer%update(source, status, message)
if (status == SHEATH_OK) plane = outer%conditions()
```

| 手続き | 意味 |
| --- | --- |
| `initialize(plasma, solver, status, message, branch, initial_order)` | `plasma%emission` を放出源として最初の根を解く。`branch` は auto（既定）/A/B/C、`initial_order` は auto の試行順で ABC の並べ替え（既定 `'ABC'`） |
| `update(source, status, message)` | 新しい放出源で解き直す。失敗すると放出源・根・条件を前のまま保ち、理由を status で返す |
| `conditions()` | 受理した根の `sheath_matching_conditions`。初期化前は `valid=.false.` |
| `source()` | 受理した放出源 |
| `accepted_solves()` | 初回を含む受理した求解の回数 |
| `relative_change()` | 最後に受理した更新での、外向き束と平均法線エネルギーの相対変化の大きい方 |
| `write_state(unit, status, message)` | 状態を unformatted の unit へ書く |
| `read_state(unit, plasma, solver, status, message)` | `write_state` の状態を、渡した plasma と solver で復元する |

branch を指定すると、その branch だけを試します。auto では、初回は `initial_order` の順に、
更新では現在の branch を先に、残りを A, B, C の順に試し、現在の根を初期値に使います。
branch の継続は数値的な選択であり、安定性による選択ではありません。
drift のある A/C 根を受理する上流の帯の許容値は、`solver%search` で設定します（[Fortran API](fortran-api.md#数値設定と解マップ)）。

`write_state` は版番号、branch の方針、根の Type・電位・電子規格化、受理回数、相対変化、放出源を書きます。
unit は stream・sequential のどちらでも構いません。開閉と位置は利用側が管理します。
`read_state` は保存した放出源と渡した plasma で、保存した Type だけを保存した根を初期値に解き直します。
見つからない場合や版が異なる場合はエラーで、別の Type へは移りません。

## 利用側の責務

| 責務 | 内容 |
| --- | --- |
| 粒子種と集団の対応 | どの粒子種を上流電子・イオン・放出電子として扱うか、上流条件を `plasma_input` に詰める |
| 面の幾何 | 面の位置・法線の向き・面積 |
| 集計と並列化 | 外向き通過の bin 集計、窓の長さ、MPI 合算、物理粒子数への換算 |
| 条件の課し方 | 上流からの注入、面での反射・除去、電位または電場の境界条件、束を目標にするか診断にするか |
| 更新の判断 | 更新の間隔、失敗時の扱い、`relative_change` による収束判定 |

外部解は定常解であり、時間発展するシースではありません。更新は弱い連成で、収束や安定性を保証しません。
J=0 以外の closure（E_H 指定）による反復は提供していません。
