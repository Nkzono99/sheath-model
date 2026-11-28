# 粒子シミュレータとの結合

このページでは、PIC や BEM などの粒子シミュレータの計算領域の上端面（**matching plane**）を、その上に広がる平面・定常の外部 1 次元シースへ接続するための **Fortran API** と、利用側が担う責務を説明します。

本ライブラリはシミュレータ固有の概念（粒子種 ID・メッシュ幾何・面積・超粒子重み・MPI 通信など）を持ちません。BEACH で採用されているほか、EMSES など任意の粒子シミュレータから同じ手順で利用できます。

## 目次

1. [結合の仕組みと 2 つの更新方式](#構成)
2. [基本ワークフロー（6 ステップ）](#結合の流れ)
3. [面上の境界条件を取り出して適用する](#面上の条件を取り出す)
   - [粒子の注入・反射・除去](#粒子の注入反射除去)
   - [場の境界条件](#場の境界条件)
4. [流出スペクトルから放出源を作る](#流出スペクトルから放出源を作る)
5. [更新方式 A：零電流での反復と checkpoint](#外部解を更新する)
6. [更新方式 B：電場と電流を合わせる候補選択](#combined-matching)
7. [MPI での運用と利用側の責務](#mpi-での使い方)

---

<a id="構成"></a>

## 1. 結合の仕組みと 2 つの更新方式

![結合の概念図。物体表面の上にシミュレータの計算領域、その上端の matching plane（外部モデルの z=0）、さらに外部シースと上流がある。外部モデルから面の条件を渡し、シミュレータから流出電子のエネルギー別流束を返して窓ごとに更新する。](figures/matching_plane_coupling.png)

> 左が領域の分担、右が観測窓ごとの弱連成ループの概念図です。位置・粒子数・矢印は縮尺を持ちません。

### 押さえるべき物理的な前提

- **matching plane（外部モデルの $z = 0$）**：シミュレータ計算領域の上端面です。外部モデルはこの面を平面境界とみなし、上方を半無限の 1 次元シースとして解きます。面より下の横方向構造が外部から見て一様とみなせる高さに配置してください。
- **電位基準**：上流（$z \to \infty$）が $0\text{ V}$、面の電位 $\phi_H$ が外部根の境界電位です。シミュレータの電位基準が異なる場合は利用側でオフセット換算します。
- **放出電子の定義（重要）**：外部モデルにとって、**面を外向き（$+z$ 向き）に通過する電子は由来を問わずすべて放出電子** です。表面由来の光電子・二次電子だけでなく、**計算領域内で反射して面を外へ抜けた上流電子も含めます**。面より内側で表面へ戻った電子は足し戻しません（[検証記録](../outputs/boundary_validation_20260923/REPORT.md)）。
- **零電流と粒子種への配分**：$J_z = 0$ の外部根では、面へ入る上流電子を外部モデル側で吸収として数え、反射して面を出る電子を放出源に含めます（障壁を越える分が `escape`）。両者を合わせると面の正味電流収支と整合します。外部モデルは電子の由来を区別しないため、`emitted_escape_flux_m2_s` / `emitted_return_flux_m2_s` を各粒子種へどう配分するかは利用側で決めます（追跡で決まる二次電子・反射電子に観測値を充て、残りを光電子に割り当てると窓内でも電荷収支が閉じます）。
- **弱連成**：一定の観測期間（窓）に計測した流出から定常の外部解を解き直し、次の窓に適用します。外部解は各窓での定常解であり、結合ループの収束や動的安定性は保証しません。

### 2 つの更新方式の選び方

| 方式 | 使う API | 特徴・使いどころ |
| --- | --- | --- |
| **方式 A：零電流（$J_z = 0$）での反復** | `sheath_matching_iteration` | 観測した流出スペクトルから $J_z = 0$ の外部根を更新する標準方式。状態管理と checkpoint 機能付き（[第 5 節](#外部解を更新する)） |
| **方式 B：電場と電流を合わせる候補選択** | `matching_states_at` + `select_combined_matching_conditions` | 面の直下で測定した内側電場 $E(H^-)$ も合わせ、電場残差と電流残差の重み付き距離が最小の外部候補へ緩和更新する方式（[第 6 節](#combined-matching)） |

なお、利用側から $E_H$ を厳密に指定したい場合は、窓ごとに `solver%solve_prescribed_field` を呼んで `evaluate_matching_conditions` で条件を作ることもできます。

---

<a id="結合の流れ"></a>

## 2. 基本ワークフロー（方式 A の 6 ステップ）

1. 上流条件を `plasma_input` に設定し、初回の放出源（表面放出の半 Maxwell 源など）を `plasma%emission` に入れる。
2. `outer%initialize(plasma, solver, status, message)` で最初の外部根を解き、`outer%conditions()` で面上の条件を得る。
3. 観測窓の間、上端面に場と粒子の境界条件を課し、面を外向きに通過した電子を法線運動エネルギー $K$ [eV] の bin ごとに物理粒子数で集計する（反射処理の前に数える）。
4. 窓の終わりに集計を MPI で合算し、面の面積 [$\mathrm{m^2}$] と窓の長さ [s] で割って数流束にし、`observed_outflow_source` で bin 放出源を作る。
5. `outer%update(source, status, message)` で外部根を解き直す。成功時は新しい条件、失敗時は前の条件のまま 3 へ戻る。
6. checkpoint 時は `outer%write_state`、再開時は `outer%read_state` を呼ぶ。

初期化から流出集計・更新・checkpoint 往復までを通しで実行できるサンプルが [example/matching_plane.f90](../example/matching_plane.f90) にあります。

```bash
fpm run --example matching_plane
```

---

<a id="面上の条件を取り出す"></a>

## 3. 面上の境界条件を取り出して適用する

`sheath_matching_conditions` は外部根が matching plane に課す条件一式を保持する型です。`outer%conditions()` で得るほか、任意の $J_z = 0$ 解または $E_H$ 指定解から直接生成できます。

```fortran
call evaluate_matching_conditions(input, root, plane, status, message)
```

`input` と `root` の組は、`sheath_equilibrium_input` + `sheath_equilibrium_result` または `prescribed_field_input` + `prescribed_field_result` のどちらにも対応します。探索は行わず、根の電位と $N_e$ を入力で再評価します（無効な根は `SHEATH_INVALID_ARGUMENT`、非有限・不正符号の流束は `SHEATH_NUMERICAL_FAILURE` で `valid = .false.`）。

### `sheath_matching_conditions` の成分

| 成分 | 単位・意味 |
| --- | --- |
| `valid`, `branch` | 有効な条件か、外部根の Type（`'A'` / `'B'` / `'C'`） |
| `plane_potential_v` | 面の電位 $\phi_H$ [V] |
| `plane_field_v_m` | 面のすぐ上の電場 $E(H^+) = -d\phi/dz$ [$\mathrm{V/m}$]（A / B は正、C は負） |
| `electron_path_minimum_v` | 面と上流の間で面電位とは別に電子が越える電位 [V]（A は $\phi_m$、B / C は `0`） |
| `ion_path_maximum_v` | 同じくイオンが越える電位 [V]（常に `0`） |
| `electron_reservoir_density_m3`, `electron_temperature_ev`, `electron_drift_mps` | 上流電子 Maxwell 分布の規格化 $N_e$（全密度ではない）・温度・内向きドリフト |
| `ion_reservoir_density_m3`, `ion_entry_speed_mps`, `ion_temperature_ev` | 上流イオンの密度・内向き入口速度・温度 |
| `electron_inward_flux_m2_s`, `ion_inward_flux_m2_s` | 外部モデルで面へ入る電子・イオンの数流束 [$\mathrm{m^{-2}\,s^{-1}}$] |
| `emitted_outward_flux_m2_s`, `emitted_escape_flux_m2_s`, `emitted_return_flux_m2_s` | 放出源の外向き・脱出・戻りの数流束 [$\mathrm{m^{-2}\,s^{-1}}$] |
| `net_current_a_m2` | $+z$ 向きの通常電流 [$\mathrm{A/m^2}$]（$J_z = 0$ 根ではほぼ $0$） |
| `upstream_negative_band_v`, `upstream_field_deficit_v_m` | 正ドリフト A / C で受理した上流帯 [V] と場の不足 [$\mathrm{V/m}$]（それ以外は `0`） |

<a id="粒子の注入反射除去"></a>

### 粒子の注入・反射・除去

経路電位は、利用側が持つ面上の**局所電位 $\phi_{\mathrm{loc}}$** [V] と組み合わせて使います（外部プロファイル全体の最小・最大電位は $\min(\phi_{\mathrm{loc}}, \text{\texttt{electron\_path\_minimum\_v}}, 0)$ と $\max(\phi_{\mathrm{loc}}, \text{\texttt{ion\_path\_maximum\_v}}, 0)$ です）。$K$ は法線運動エネルギー [eV] です。

| 判定対象 | 到達・脱出の条件 | 面での法線エネルギー / 満たさないとき |
| --- | --- | --- |
| 上流の電子が面に届く | $K_\infty \ge K_{\mathrm{acc},e} = -\min(\phi_{\mathrm{loc}}, \phi_{e,\mathrm{path}}, 0)$ | 面でのエネルギー $K = K_\infty + \phi_{\mathrm{loc}}$ |
| 上流のイオンが面に届く | $K_\infty \ge K_{\mathrm{acc},i} = \max(\phi_{\mathrm{loc}}, \phi_{i,\mathrm{path}}, 0)$ | 面でのエネルギー $K = K_\infty - \phi_{\mathrm{loc}}$ |
| 面を外へ出た電子が上流へ抜ける | $K \ge K_{\mathrm{esc},e} = \phi_{\mathrm{loc}} - \min(\phi_{\mathrm{loc}}, \phi_{e,\mathrm{path}})$ | 満たさなければ同じ $K$ で面へ戻す（反射） |
| 面を外へ出たイオンが上流へ抜ける | $K \ge K_{\mathrm{esc},i} = \max(\phi_{\mathrm{loc}}, \phi_{i,\mathrm{path}}) - \phi_{\mathrm{loc}}$ | 満たさなければ面へ戻す |

これらのしきい値と局所電位に応じた上流電子数流束は、次の **純関数（`pure`）** で直接取得できます。

| 純関数 | 戻り値 |
| --- | --- |
| `plane_electron_access_energy_ev(conditions, phi_loc_v)` | 上流電子の到達下限エネルギー $K_{\mathrm{acc},e}$ [eV] |
| `plane_ion_access_energy_ev(conditions, phi_loc_v)` | 上流イオンの到達下限エネルギー $K_{\mathrm{acc},i}$ [eV] |
| `plane_electron_escape_threshold_ev(conditions, phi_loc_v)` | 流出電子の脱出しきい値 $K_{\mathrm{esc},e}$ [eV] |
| `plane_ion_escape_threshold_ev(conditions, phi_loc_v)` | 流出イオンの脱出しきい値 $K_{\mathrm{esc},i}$ [eV] |
| `plane_electron_inward_flux_m2_s(conditions, phi_loc_v, electron_mass_kg)` | 到達条件で制限した上流電子の数流束 [$\mathrm{m^{-2}\,s^{-1}}$]（温度または質量 $\le 0$ なら `NaN`） |

```fortran
! 面の局所電位 phi_loc_v での注入数流束と、流出電子の脱出／反射判定の例
k_acc_e = plane_electron_access_energy_ev(plane, phi_loc_v)
gamma_e_loc = plane_electron_inward_flux_m2_s(plane, phi_loc_v, plasma%electron_mass_kg)
k_esc_e = plane_electron_escape_threshold_ev(plane, phi_loc_v)

if (electron_normal_energy_ev >= k_esc_e) then
  ! 上流へ脱出（計算領域から除去）
else
  ! 外部シースで折り返して同じ法線エネルギーで面へ戻す
end if
```

- **上流電子の注入サンプリング**：内向きドリフト Maxwell 分布 $g(a) \propto e^{-(a - u)^2}$ を $K_\infty \ge K_{\mathrm{acc},e}$ の範囲に制限し、流束重み付けして法線速度を標本化します。その数流束が `plane_electron_inward_flux_m2_s` であり、$\phi_{\mathrm{loc}} = \phi_H$ では `plane%electron_inward_flux_m2_s` に一致します。
- **接線速度・イオン**：外部モデルは法線方向のみを定めるため、接線速度分布は利用側で与えます。温かいイオンは流体モデルで速度分布を持たず、冷たいイオンは入口速度ビームです。また外部モデルは外向きイオンを含まないため、面を外へ出るイオンが無視できない状況は適用外です。

<a id="場の境界条件"></a>

### 場の境界条件

面の平均電位を `plane_potential_v` に固定する **Dirichlet 条件** と、法線電場を `plane_field_v_m` に合わせる **Neumann 条件** の 2 通りがあります（面に面電荷がなければ $E(H^-) = E(H^+)$）。どちらを課し、残る量や流束を整合診断に使うかは利用側で設計します。

---

<a id="流出スペクトルから放出源を作る"></a>

## 4. 流出スペクトルから放出源を作る

面を外向きに通過した電子の法線運動エネルギー $K$ [eV] をヒストグラム集計し、bin 放出源（`emission_source`）へ変換する補助 API です。平均エネルギーだけの半 Maxwell 分布へ縮約しないため、計算領域内の加速や電位極小で変形したスペクトル形状がそのまま外部シースへ伝わります。

| 手続き・定数 | 意味 |
| --- | --- |
| `LOG_ENERGY_DEFAULT_LOWEST_EV`, `LOG_ENERGY_DEFAULT_DECADES`, `LOG_ENERGY_DEFAULT_BINS_PER_DECADE`, `LOG_ENERGY_DEFAULT_BINS` | 既定対数格子の定数：下端 `1e-3` eV（1 meV）、`6` decade（1 keV まで）、`24` bin/decade、計 `145` bin。固定長配列の宣言に利用可能 |
| `log_energy_edges([lowest_edge_ev, decades, bins_per_decade])` | 先頭 bin $[0, \text{lowest})$ と対数 bin の境界配列 [eV] を返す。不正引数では空配列 |
| `log_energy_bin_count([decades, bins_per_decade])` | `log_energy_edges` が作る bin 数（`decades * bins_per_decade + 1`）。不正引数では `0` |
| `log_energy_bin(energy_ev [, lowest_edge_ev, decades, bins_per_decade])` | 境界配列を作らず $O(1)$ で bin 番号を返す（`energy_bin(log_energy_edges(...), energy_ev)` と厳密一致）。粒子通過ごとの集計向き |
| `energy_bin(edges_ev, energy_ev)` | 任意の昇順境界配列に対する bin 番号。範囲外・`NaN` は端の bin、bin なしは `0` |
| `call observed_outflow_source(edges_ev, flux_m2_s, source, status, message)` | bin ごとの外向き数流束 [$\mathrm{m^{-2}\,s^{-1}}$] から `emission_source` を構築。両端の空 bin は自動で除き、内部の空 bin は残す |

`flux_m2_s` は有限・非負かつ合計が正である必要があります（満たさない場合は `SHEATH_INVALID_ARGUMENT`）。bin の定義は任意で、`log_energy_*` を使わずに独自のエネルギー格子を渡すこともできます。

```fortran
! 既定の対数格子（145 bin）で粒子通過を O(1) 集計し、放出源を作る例
real(dp) :: weight(LOG_ENERGY_DEFAULT_BINS)
real(dp), allocatable :: edges(:)
type(emission_source) :: source

weight = 0.0_dp
! --- 観測窓内の通過ループ ---
! b = log_energy_bin(energy_ev)
! weight(b) = weight(b) + particle_weight

! --- 窓の終わり（MPI なら weight を全ランクで合算後）---
edges = log_energy_edges()
call observed_outflow_source(edges, weight / (area_m2 * duration_s), source, status, message)
```

---

<a id="外部解を更新する"></a>

## 5. 更新方式 A：零電流での反復と checkpoint（`sheath_matching_iteration`）

`sheath_matching_iteration` は、現在受理している $J_z = 0$ 外部根と放出源を保持し、観測窓ごとに更新するオブジェクトです。上流の `plasma` と `solver` 設定は利用側の設定とみなし、状態には保存せず `initialize` と `read_state` で毎回渡します。

| 型束縛手続き | 意味 |
| --- | --- |
| `outer%initialize(plasma, solver, status, message [, branch, initial_order])` | `plasma%emission` を初回源として最初の外部根を解く。`branch` は `'auto'`（既定）/`'A'`/`'B'`/`'C'`、`initial_order` は `auto` 初回順（`'ABC'` の並べ替え、既定 `'ABC'`） |
| `outer%update(source, status, message)` | 新しい放出源で解き直す。**失敗した場合は直前の放出源・根・条件をそのまま保持**し、`status`/`message` に理由を返す |
| `outer%conditions()` | 受理済みの `sheath_matching_conditions`（初期化前は `valid = .false.`） |
| `outer%source()` | 受理済みの `emission_source` |
| `outer%accepted_solves()` | 初回を含む受理済み求解回数（`integer(i32)`） |
| `outer%relative_change()` | 最後に受理した更新での「外向き数流束の相対変化」と「平均法線エネルギーの相対変化」の大きい方 |
| `outer%write_state(unit, status, message)` | 状態を `unformatted` の `unit` へ書き出す |
| `outer%read_state(unit, plasma, solver, status, message)` | `write_state` で書いた状態を、渡した `plasma` と `solver` で復元する |

- **Type の試行順**：`branch` を明示するとその Type のみ試します。`'auto'` では初回に `initial_order` 順（例：低太陽高度で C から始めたい場合は `initial_order = 'CAB'`）、更新では「現在の Type を先頭、残りを A, B, C 順」に現在の根を初期値として試します。この継続順は数値的な選択であり、動的安定性の選択ではありません。
- **上流帯の設定**：正ドリフト A / C を許容する場合は、`solver%search` に `upstream_band_tolerance` または `upstream_field_floor_v_m` を設定して渡します。
- **`message` 長**：試行した各 Type の失敗理由を連結するため、512 文字程度を推奨します。
- **収束判定と失敗時対応**：`relative_change()` のしきい値、更新間隔、`update` 失敗時に前条件で継続するか停止するかは利用側が決めます。

### checkpoint（`write_state` / `read_state`）

- `write_state` は版番号・`branch` 方針・根の Type／電位／$N_e$・受理回数・相対変化・放出源を書き出します（`plasma` と `solver` は書きません）。`unit` のアクセス方式は `stream` / `sequential` のどちらも可で、開閉・位置管理は利用側が行います。
- `read_state` は保存した放出源と渡した `plasma` を用いて、**保存した Type だけ**を保存した根を初期値に解き直します。別の Type への自動フォールバックは行わず、再現できない場合や版不一致はエラーになります。再開時は checkpoint 作成時と同じ `plasma`・`solver` を渡してください。

---

<a id="combined-matching"></a>

## 6. 更新方式 B：電場と電流を合わせる候補選択

接合面に面電荷を置かない場合、面の直下で観測した内側電場 $E(H^-)$ と外部解の電場 $E(H^+)$ が一致し、かつ正味電流 $J_z = 0$ となる状態が目標です。

![接合候補の選択の模式図。左で規格化した電場残差と電流残差が原点に最も近い候補を目標にする。右で前の電位から提案電位を作り、目標と同じ Type の候補を採用する。](figures/matching_selection_schematic.png)

> 左が各候補の規格化残差、右が同じ候補を面電位 $\phi_H$ 軸上に並べた図です。目標（Target）が Type B のため、前の電位から緩和・上限適用した提案電位（Proposal）に最も近い Type B 候補を採用（Selected）します。

### 手順と目的関数

1. **候補の列挙（`matching_states_at`）**：前の面電位を含む電位格子を用意し、各電位 $\phi_H$ で最新の放出源を持つ `plasma_input` から接続可能な外部候補（$\phi_H$ の符号に応じた B または C、および上流接続を満たす A）を列挙します。
2. **目標の決定と緩和選択（`select_combined_matching_conditions`）**：全電位の候補を 1 つの配列にまとめて渡し、規格化残差
   $$
   S_i = \sqrt{\left(\frac{E_i(H^+) - E(H^-)}{\sigma_E}\right)^2 + \left(\frac{J_i / (e\Gamma_{i,\mathrm{in}})}{\sigma_J}\right)^2}
   $$
   が最小となる候補を `target_index` とします。次に、前の面電位 $\phi_{\mathrm{prev}}$ から目標電位までの差に緩和係数 `relaxation` を掛け、上限 `max_step_v` を適用した提案電位 $\phi_{\mathrm{prop}}$ を作ります。目標と**同じ Type** の候補のうち
   $$
   \lvert \phi_{H,i} - \phi_{\mathrm{prop}} \rvert + 10^{-3}\,\lvert \phi_{e,\mathrm{path},i} - \phi_{e,\mathrm{path},\mathrm{target}} \rvert
   $$
   が最小の候補を `selected_index` として返します（同点は配列先頭側を優先。上限は提案電位に適用するため、採用電位には離散格子への丸めが入ります）。
3. **採用**：`status == SHEATH_OK` を確認し、`candidates(selected_index)` を次の窓の条件に使います。

### 手続きと引数一覧

#### `matching_states_at`

```fortran
call matching_states_at(input, plane_potential_v, candidates, status, message &
    [, minimum_gaps_v, search])
```

| 引数 | 制約・意味 |
| --- | --- |
| `input` | 最新の放出源を含む `plasma_input` |
| `plane_potential_v` | 試行する面電位 $\phi_H$ [V]（有限値） |
| `candidates(:)` | `type(sheath_matching_conditions), allocatable, intent(out)`。候補なしは長さ 0・`SHEATH_NO_PHYSICAL_SOLUTION` |
| `minimum_gaps_v(:)` | （省略可）Type A の極小を走査する $\min(\phi_H, 0) - \phi_m$ [V] の正・厳密増加・2 点以上の配列。既定は $0.1\text{ mV}$〜$\sqrt{10}\text{ V}$ の対数 33 点 |
| `search` | （省略可）`sheath_search_options`。正ドリフト A / C の上流帯許容値などを指定（既定は厳密判定） |

#### `select_combined_matching_conditions`

```fortran
call select_combined_matching_conditions(candidates, previous, inner_field_v_m, &
    sigma_field_v_m, sigma_current, relaxation, max_step_v, &
    selected_index, target_index, status, message)
```

| 引数 | 制約・意味 |
| --- | --- |
| `candidates(:)` | 全電位から集めた `sheath_matching_conditions` の配列 |
| `previous` | 前の窓の有効な `sheath_matching_conditions`（`previous%valid` 必須） |
| `inner_field_v_m` | 面の直下で測定した内側電場 $E(H^-)$ [$\mathrm{V/m}$] |
| `sigma_field_v_m` | 電場不確かさの尺度 $\sigma_E$ [$\mathrm{V/m}$]（正） |
| `sigma_current` | 無次元電流 $J / (e\Gamma_{i,\mathrm{in}})$ の尺度 $\sigma_J$（正） |
| `relaxation` | 緩和係数（$0 < \text{\texttt{relaxation}} \le 1$） |
| `max_step_v` | 1 回の提案電位変化量の上限 [V]（非負。`0` で上限無効化） |
| `selected_index`, `target_index` | 採用候補・目標候補の 1 始まり添字（見つからない場合は `0`） |

### 呼び出し例

```fortran
program combined_matching_example
  use sheath_model
  implicit none
  type(plasma_input) :: plasma
  type(sheath_matching_conditions) :: previous
  type(sheath_matching_conditions), allocatable :: at_phi(:), all_candidates(:)
  integer(i32) :: status
  integer :: k, sel, tgt
  real(dp) :: phi
  character(len=256) :: message

  plasma%electron_drift_mps = 0.0_dp
  plasma%emission = maxwellian_emission(55.42562584220407e6_dp, 2.2_dp)

  allocate (all_candidates(0))
  do k = 0, 20
    phi = 2.0_dp + 0.2_dp*real(k, dp)
    call matching_states_at(plasma, phi, at_phi, status, message)
    if (status == SHEATH_OK) all_candidates = [all_candidates, at_phi]
  end do

  ! ここでは例として先頭候補を前状態とし、観測内側電場 1.62 V/m に向かって選択する
  previous = all_candidates(1)
  call select_combined_matching_conditions( &
      candidates=all_candidates, previous=previous, &
      inner_field_v_m=1.62_dp, sigma_field_v_m=0.05_dp, sigma_current=0.05_dp, &
      relaxation=0.5_dp, max_step_v=0.5_dp, &
      selected_index=sel, target_index=tgt, status=status, message=message)
  if (status /= SHEATH_OK) stop 1
  print '(a,a,2f10.4)', 'Selected Type ', all_candidates(sel)%branch, &
      all_candidates(sel)%plane_potential_v, all_candidates(tgt)%plane_potential_v
end program combined_matching_example
```

電位格子の分担・収集、内側電場と不確かさの推定、更新間隔、選択状態の保存は利用側で管理します。検証テストは [test/test_matching.f90](../test/test_matching.f90) にあります。

> **実測の目安**：この閉じ方を用いた EMSES の準 1 次元柱（$4 \times 4$ セル）では、matching plane を極小の $2.5\text{ m}$ 下からシースの外（約 $10\lambda_D$）まで動かしても、面電位は単体解と $0.01$〜$0.05\text{ V}$ で一致しました。シース外では $E(H^+)$ が小さくなるため、差は $E(H^-)$ の窓平均雑音に敏感になります。

---

<a id="mpi-での使い方"></a>

## 7. MPI での運用と利用側の責務

### MPI での推奨分担

本ライブラリ自身は MPI 通信を行いません。並列シミュレータでは次の運用を推奨します。

- **流出の集計**：各ランクで物理粒子重みを bin 集計し、窓の終わりに全ランクで `allreduce` 合算する。
- **方式 A の更新**：全ランクが同じ放出源で `outer%update` を呼ぶ（同じバイナリ・入力なら全ランクで同じ根が得られる。念のため `status` を照合すると不一致を検出できる）。
- **checkpoint**：代表 1 ランクが `write_state` で書き、再開時は全ランクが同じファイルを `read_state` で読む。
- **方式 B の候補列挙**：電位格子をランク間に分割して `matching_states_at` を並列実行し、候補配列を一定順序で集約してから `select_combined_matching_conditions` を呼ぶ。

### 利用側が担う責務チェックリスト

| 責務 | 実装・確認すべき事項 |
| --- | --- |
| **幾何と単位換算** | 面の位置・法線・面積を定義し、シミュレータ規格化単位と SI / eV を相互換算する |
| **入射と反射** | 粒子種を上流電子・イオン・放出電子に対応づけ、接線方向を含めた注入・反射・除去を実装し、入射数流束を `electron_inward_flux_m2_s`・`ion_inward_flux_m2_s` と照合する |
| **電位・電場** | $\phi_H$ の Dirichlet か $E(H^+)$ の Neumann かを選び、符号と基準を `plane_potential_v`・`plane_field_v_m` と揃える |
| **流出の集計** | 由来を問わず外向き通過電子を**反射処理の前**に数え（Type A で上から戻る遅い電子も放出源に含まれる）、`emitted_*_flux_m2_s` と照合する |
| **更新の制御** | 時間積分・更新間隔・失敗時の扱い・`relative_change()` による収束判断を管理する |
| **checkpoint** | ファイル unit の開閉を管理し、保存前と再開直後の条件が一致することを確認する（[example/matching_plane.f90](../example/matching_plane.f90)） |

> 参考：BEACH での結合実装は、BEACH リポジトリの `src/runtime/sheath/bem_surface_current_model.f90`（条件の写像）と `src/runtime/sheath/bem_zhao_outflow_refresh.f90`（窓集計・更新・checkpoint）にあります。
