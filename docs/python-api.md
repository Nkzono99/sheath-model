# Python API リファレンス

Python パッケージ `sheath_model` は NumPy / SciPy による独立実装で、次の **2 つの物理モデル** を提供します。

1. **Zhao モデル（$J_z = 0$）**：`SheathSolver` による平衡解・プロファイル・局所流束・1 次元速度分布（VDF）・解マップ
2. **固定入射モデル（Fixed-Reservoir）**：遠方 $0\text{ V}$ の電子分布を固定した分岐判定・有限箱・準中性外側領域付きプロファイル

> [!NOTE]
> Zhao モデルの $E_H$ 指定、静的評価、bin 放出源、および粒子シミュレータ結合 API は [Fortran API](fortran-api.md) / [結合 API](coupling.md) で提供しています。

- すべての入力・ソルバー・結果・プロファイルは **不変（`frozen=True`）の dataclass** です。一部だけ変更して作り直すには `dataclasses.replace` を使います。
- `SheathSolver` は前回の解を保持せず、異なる入力にそのまま再利用できます。
- 単位・符号・例外の基本方針は [共通の規約](conventions.md) を参照してください。

## 目次

1. [公開シンボル一覧](#公開シンボル一覧)
2. [Part 1：Zhao モデル（$J_z = 0$）](#part-1zhao-モデル)
   - [物理入力（`FixedEntryParams` / `ZhaoParams`）](#物理入力)
   - [ソルバーと設定（`SheathSolver`）](#sheathsolver)
   - [結果・プロファイル・VDF](#結果)
   - [解マップ（`EquilibriumAtlas`）](#解マップ)
3. [Part 2：遠方の電子分布を固定するモデル](#fixed-reservoir)
4. [Part 3：補助関数とコマンドライン](#補助関数とコマンドライン)

---

## 公開シンボル一覧

`from sheath_model import ...` で直接インポートできる全シンボル（`__all__`）です。

| カテゴリ | シンボル |
| --- | --- |
| **Zhao モデル入力** | `FixedEntryParams`, `ZhaoParams` |
| **Zhao モデル求解・設定** | `SheathSolver`, `SearchOptions`, `ProfileOptions`, `ContinuationOptions` |
| **Zhao モデル結果・診断** | `EquilibriumResult`, `CandidateSet`, `DensityResult`, `SheathProfile`, `SearchDiagnostics`, `SearchFailure` |
| **解マップ** | `EquilibriumAtlas`, `AtlasOptions`, `AtlasPoint` |
| **固定入射モデル入力** | `FixedReservoir`, `MaxwellianElectrons`, `KappaElectrons`, `MaxwellianEmission`, `BinnedEmission`, `EmissionSpectrum` |
| **固定入射モデル診断・求解** | `classify_branch`, `BranchCriterion`, `turning_field_difference`, `solve_finite_column`, `FiniteColumnSolution`, `solve_presheath`, `PresheathSolution`, `OuterWellSeed`, `replacement_targets` |
| **共通補助関数** | `ion_density_ratio`, `ion_critical_potential` |

---

<a id="part-1zhao-モデル"></a>

## Part 1：Zhao モデル（$J_z = 0$）

### 基本的な使い方

```python
from sheath_model import ZhaoParams, SheathSolver

params = ZhaoParams(sun_elevation_deg=60.0, electron_drift_mode="zero")
profile = SheathSolver().solve_profile(params, branch="A")

eq = profile.equilibrium
flux = profile.fluxes(0.0)
vdf = profile.vdf(0.0, species="swe")
print(f"phi_H={eq.surface_potential_v:.3f} V, phi_min={eq.minimum_potential_v:.3f} V")
print(f"inward ion flux={-flux['Gamma_swi_signed_m2s']:.3e} m^-2 s^-1")
print(f"VDF grid points={len(vdf['swe']['vz_mps'])}")
```

固定入口・温かいイオンの例は [examples/fixed_entry.py](../examples/fixed_entry.py) を参照してください。

---

<a id="物理入力"></a>

### 物理入力

<a id="fixedentryparams"></a>

#### `FixedEntryParams`

法線入口条件と半 Maxwell 放出源を直接指定します。

| 属性 | 既定値 | 単位・意味 |
| --- | --- | --- |
| `ion_density_m3` | `8.7e6` | $\mathrm{m^{-3}}$。上流イオン密度 $n_{i\infty}$ |
| `ion_entry_speed_mps` | `405299.88897111727` | $\mathrm{m/s}$。内向き法線入口速度 $u_{i\infty}$ |
| `ion_temperature_ev`, `ion_pressure_factor` | `0.0`, `1.0` | $\mathrm{eV}$, 無次元。イオン温度と圧力項係数 |
| `electron_temperature_ev` | `12.0` | $\mathrm{eV}$。背景電子温度 $T_e$ |
| `electron_drift_mps` | `0.0` | $\mathrm{m/s}$。背景電子の内向き法線ドリフト $v_d$ |
| `emission_density_m3`, `emission_temperature_ev` | `0.0`, `2.2` | $\mathrm{m^{-3}}$, $\mathrm{eV}$。放出電子 Maxwell 規格化 $N_{\mathrm{em}}$（外向き半分の密度は $N_{\mathrm{em}}/2$）と温度 |
| `ion_mass_kg`, `electron_mass_kg` | 陽子・電子質量 | $\mathrm{kg}$ |

<a id="zhaoparams"></a>

#### `ZhaoParams`

太陽高度と全風速から法線条件と光電子源を構成します（[変換規則](model.md#太陽風と照射角からの入力)）。属性と既定値は Fortran の `zhao_equilibrium_input` と同一です。

| 属性 | 既定値 | 意味 |
| --- | --- | --- |
| `sun_elevation_deg` | `60.0` | 太陽高度 $\alpha$（`0`〜`90`） |
| `ion_density_m3` | `8.7e6` | 上流イオン密度 |
| `photoelectron_reference_density_m3` | `64e6` | 垂直入射時の光電子規格化密度 $n_{\mathrm{pe,ref}}$ |
| `electron_temperature_ev`, `ion_temperature_ev`, `ion_pressure_factor` | `12.0`, `0.0`, `1.0` | `FixedEntryParams` と同じ |
| `photoelectron_temperature_ev` | `2.2` | 光電子温度 $T_{\mathrm{pe}}$ |
| `solar_wind_speed_mps` | `468e3` | 全太陽風速 $v_{\mathrm{sw}}$ |
| `ion_mass_kg`, `electron_mass_kg` | 陽子・電子質量 | 粒子質量 |
| `electron_drift_mode` | `"normal"` | `"normal"`, `"full"`, `"zero"`（無ドリフトの A / C には `"zero"` を指定） |
| `ion_drift_mode` | `"normal"` | `"normal"`, `"full"` |

両入力型とも、読み取り専用の導出プロパティとして `length_scale_m`、`density_scale_m3`、`v_swe_th_mps`、`v_emission_th_mps`、`cs_mps`、`mach`、`u`、`tau` を持ちます（`ZhaoParams` はさらに `alpha_rad`, `normal_wind_speed_mps`, `electron_drift_mps`, `ion_entry_speed_mps`, `emission_density_m3`, `emission_temperature_ev` を提供します）。

---

<a id="sheathsolver"></a>

### ソルバーと設定（`SheathSolver`）

```python
SheathSolver(
    search=SearchOptions(),
    profile=ProfileOptions(),
    continuation=ContinuationOptions(),
    equilibrium_atlas=None,
)
```

| メソッド | 戻り値 | 動作 |
| --- | --- | --- |
| `solve_equilibrium(inputs, *, branch="auto", initial_guess=None)` | `EquilibriumResult` | 指定 Type、または `auto` 順で最初の物理解。`initial_guess` に過去の `EquilibriumResult` を渡すと同 Type の初期値に追加 |
| `solve_equilibrium_candidates(inputs, *, branch="auto", deflation=True, max_roots=16)` | `CandidateSet` | 見つかった全物理解（`candidates` タプルと `diagnostics`）。`max_roots` は Type ごとの上限 |
| `build_profile(equilibrium)` | `SheathProfile` | 探索を行わず、`equilibrium.inputs` で元の方程式と物理解判定を検査して再構成 |
| `solve_profile(inputs, *, branch="auto", initial_guess=None)` | `SheathProfile` | `solve_equilibrium` に続けて `build_profile` を実行 |
| `build_equilibrium_atlas(parameters, *, branches=("A", "B", "C"), options=None, deflation=False)` | `EquilibriumAtlas` | 新しい解マップを構築して返す（ソルバーに添付済みの表は変更しない） |

- `branch="auto"` の試行順：`ZhaoParams` で `sun_elevation_deg < 20` のとき `C → A → B`、それ以外は `A → B → C`。
- 探索に失敗すると `SearchFailure` を送出し、`exc.diagnostics` に A / B / C 順の `SearchDiagnostics` を保持します。

#### `SearchOptions`

Fortran の `sheath_search_options` と同じ意味です（$E_H$ 指定専用の `scalar_max_depth` を除く）。

| 属性 | 既定値 | 制約・意味 |
| --- | --- | --- |
| `method` | `"auto"` | `"auto"`, `"newton"`, `"lm"`, `"bracket"`（`"bracket"` は明示的な `"B"` / `"C"` のみ） |
| `residual_tolerance` | `1e-10` | 正。規格化残差の許容値 |
| `max_iterations`, `max_backtracks`, `max_starts` | `100`, `24`, `32` | 反復（$\ge 0$）、直線探索（$\ge 1$）、初期値数（$\ge 1$）の上限 |
| `use_default_guesses` | `True` | 標準初期値を使うか |
| `bracket_points` | `96` | $\ge 2$。スカラー探索格子点数 |
| `potential_extent` | `200.0` | 正。探索電位の無次元上限 |
| `upstream_band_tolerance` | `0.0` | $[0, 1)$。正ドリフト A / C の上流帯許容幅（割合） |
| `upstream_field_floor_v_m` | `0.0` | $\ge 0$。正ドリフト A / C の上流帯の場不足上限 [$\mathrm{V/m}$] |

#### `ProfileOptions`

位置と電位は無次元量（長さ尺度 `inputs.length_scale_m`、電位尺度 `inputs.emission_temperature_ev`）で指定します。

| 属性 | 既定値 | 制約・意味 |
| --- | --- | --- |
| `zmax_hat` | `80.0` | 正。返す高さの上限（`length_scale_m` の倍数） |
| `n_profile_grid` | `600` | $\ge 32$。B / C の電位点数（内部では最低 600 点を使用） |
| `n_type_a_grid` | `8000` | $\ge 32$。A の上流側の電位点数（内部では最低 2000 点を使用） |
| `profile_phi_tol_hat` | `1e-3` | 正。上流側の打ち切り電位 $\lvert \hat{\phi} \rvert$（B / C は $\lvert \hat{\phi}_H \rvert / 2$ が上限） |
| `type_a_phi_m_eps_hat` | `1e-5` | 正。A の格子を極小からわずかにずらして始める幅（$\lvert \hat{\phi}_m \rvert$ の 5% が上限） |
| `upstream_band_margin` | `0.05` | $(0, 1)$。受理した上流帯がある根で帯の手前に残す余白の割合 |

#### `ContinuationOptions`

Fortran と同一です（`method="parameter"` または `"arclength"`, `initial_step=0.25`, `min_step=1e-4`, `max_step=0.5`, `max_steps=128`, `max_root_distance=0.75`）。

---

<a id="結果"></a>

### 結果・プロファイル・VDF

#### `EquilibriumResult` / `CandidateSet` / `DensityResult`

- **`EquilibriumResult`** の属性・メソッド：
  - `inputs`, `branch`, `surface_potential_v`, `minimum_potential_v`, `ambient_electron_density_m3`（規格化 $N_e$）, `residual_norm`, `diagnostics`（`SearchDiagnostics`）, `upstream_negative_band_v`, `upstream_field_deficit_v_m`
  - `density(potential_v, *, side="upper") -> DensityResult`：指定電位 [V]（スカラーまたは配列）での局所密度。Type A は `side="lower"` または `"upper"` を選択。
- **`CandidateSet`**：`candidates: tuple[EquilibriumResult, ...]` と `diagnostics: SearchDiagnostics`。
- **`DensityResult`**：`ion_m3`, `electron_free_m3`, `electron_reflected_m3`, `emitted_free_m3`, `emitted_captured_m3` とプロパティ `charge_c_m3` [$\mathrm{C/m^3}$]。

#### `SheathProfile`

| 属性・メソッド | 意味 |
| --- | --- |
| `equilibrium` | 元の `EquilibriumResult` |
| `z_m`, `z_hat`, `potential_v`, `electric_field_v_m` | 境界から上流への読み取り専用 NumPy 配列（非一様間隔） |
| `density` | 配列を保持する `DensityResult` |
| `turning_height_m` | Type A の極小高さ [m]（B / C は `-1.0`） |
| `sample(z, *, unit="m") -> dict` | 位置 `z` での局所状態（電位・電場・枝の区間・カットオフ速度・各集団の密度） |
| `fluxes(z, *, unit="m") -> dict` | 位置 `z` での正確な軌道流束・電流・平均速度（表示格子ではなく分布積分で計算） |
| `vdf(z, *, species="all", unit="m", n_v=4001, n_sigma=6.0, ion_sigma_frac=0.03) -> dict` | 位置 `z` での 1 次元速度分布関数 $g(v_z)$（$\int g\,dv_z = n(z)$） |

- 位置 `z` の単位は既定で `"m"` です。無次元位置を渡す場合は `unit="hat"` を明示します。計算区間外は `ValueError` です。
- **`sample` / `fluxes` の主なキー**（集団略称：`swe`＝背景電子、`swi`＝イオン、`phe`＝放出電子）：
  - `sample`：`branch`, `side`, `z_m`, `z_hat`, `phi_V`, `E_Vpm`, `vcut_swe_mps`, `vcut_phe_mps`, `v_i_mps`, `swe_reflected_active`, `phe_captured_active`, `n_swi_m3`, `n_swe_f_m3`, `n_swe_r_m3`, `n_phe_f_m3`, `n_phe_c_m3`, `n_total_m3`, `rho_m3`（および対応する `*_hat`）
  - `fluxes`：`sample` の全キーに加え、`Gamma_net_m2s`, `J_net_Apm2`、集団別の `Gamma_{swi,swe,phe}_signed_m2s`, `J_{swi,swe,phe}_signed_Apm2`, `mean_v_{swi,swe,phe}_mps`、および成分別の `Gamma_*_m2s`, `J_*_Apm2`。**`*_signed_m2s` と `*_Apm2` はすべて $+z$ 向きを正とします**（例：境界へ入るイオン数流束は `-fluxes["Gamma_swi_signed_m2s"]`）。
- **`vdf` の戻り値と制約**：
  - `species` は `"swe"`, `"phe"`, `"swi"`, `"all"`。`n_v` は 201 以上の整数（偶数は +1 して奇数化）。
  - 戻り値 dict は `"state"`、`"species"` と、要求した各集団の dict（`"vz_mps"`, `"g_total"`、さらに `"swe"` は `"g_free_incoming"`, `"g_reflected_incoming"`, `"g_reflected_outgoing"`、`"phe"` は `"g_free_outgoing"`, `"g_captured_outgoing"`, `"g_captured_returning"`）を含みます。
  - イオン VDF は冷たいビームを幅 `ion_sigma_frac` の Gauss 分布で表示用に正則化したものです。**温かいイオン（`ion_temperature_ev > 0`）では定義されないため `ValueError`** になります。温かいイオンでは `species="swe"` または `"phe"` を指定してください。

---

<a id="解マップ"></a>

### 解マップ（`EquilibriumAtlas`）

構築・保存・読込と候補探索の実行例は [examples/solution_atlas.py](../examples/solution_atlas.py) にあります。

| 名前 | 意味 |
| --- | --- |
| `EquilibriumAtlas(points=(), *, options=None)` | 呼び出し側所有の解マップ。`points` は `AtlasPoint` のタプル |
| `atlas.add(result, *, component=None, search=None)` | 元の方程式と物理解判定を検査して登録する（求解による自動登録は行わない） |
| `atlas.save(path)` / `EquilibriumAtlas.load(path, *, options=None)` | Fortran の `sheath_equilibrium_atlas` と互換の版付きテキスト形式で保存・読込 |
| `atlas.attempts` | `build_equilibrium_atlas` が記録する入力・Type ごとの結果（`"accepted"` / `"excluded"` / `"unresolved"` と `diagnostics`） |
| `AtlasOptions` | 既定：`neighbors=8`, `max_distance=1.0`, `interpolate=True`, `component_distance=0.75` |
| `AtlasPoint` | 記録点（`branch`, `component`, `key`, `coordinates`, `spectrum_shape`） |

Fortran が保存した bin スペクトルの記録も読み込めますが、Python の Maxwell 源の問い合わせには使用しません。

---

<a id="fixed-reservoir"></a>

## Part 2：遠方の電子分布を固定するモデル

遠方 $0\text{ V}$ のリザーバーで全速度域の電子分布と密度を固定するモデル群です（物理モデルと模式図は [物理モデル：固定入射モデル](model.md#fixed-reservoir) を参照してください）。

- 入射速度は壁向き（$-z$ 向き）を正、温度は eV、放出は外向き数流束 [$\mathrm{m^{-2}\,s^{-1}}$] で指定します。
- 有限箱では $0\text{ V}$ の上端からの入射分を用い、外側領域付きモデルでは遠方で全電子密度 $=$ イオン密度（準中性）である必要があります。

### 計算例

次の条件では、仮の中性谷底の局所判定は Type B 側（`"B"`）を示します。高さ $56.5\text{ m}$ の有限箱では上端付近に谷が現れ `A` となり、外側領域付きモデルではシースの外側に谷を持つ `B_out` が得られます。

```python
from scipy.constants import elementary_charge
from sheath_model import (
    FixedReservoir, MaxwellianElectrons, MaxwellianEmission,
    classify_branch, solve_finite_column, solve_presheath,
)

wind = FixedReservoir(
    (MaxwellianElectrons(density_m3=20e6, temperature_ev=4.0, drift_mps=400e3),),
    ion_density_m3=20e6,
    ion_speed_mps=400e3,
    ion_temperature_ev=10.0,
)
emission = MaxwellianEmission(flux_m2_s=36e-6 / elementary_charge, temperature_ev=2.2)

criterion = classify_branch(wind, emission)
column = solve_finite_column(wind, emission, height_m=56.5)
outer = solve_presheath(wind, emission, "B_out")

print("criterion:", criterion.outcome, f"(R={criterion.ratio:.3f})")
print("finite column:", column.branch, f"z_min={column.minimum_height_m:.2f} m")
print("presheath:", outer.branch, f"phi_band={outer.band_potential_v:.3f} V, phi_min={outer.minimum_potential_v:.3f} V")
```

### 入力クラス

| クラス | 属性と制約 |
| --- | --- |
| `MaxwellianElectrons(density_m3, temperature_ev, drift_mps=0.0)` | 全速度域の密度（正）、温度（正、eV）、内向き法線ドリフト [m/s]。複数成分をタプルで足し合わせ可能 |
| `KappaElectrons(density_m3, core_temperature_ev, kappa)` | 全速度域の密度（正）、コア温度（正、eV）、$\kappa > 1$（中心静止） |
| `FixedReservoir(electrons, ion_density_m3, ion_speed_mps, ion_temperature_ev=0.0, ion_pressure_factor=1.0, ion_mass_kg=..., electron_mass_kg=...)` | 電子成分の列（空不可）、正のイオン密度・速度・質量。温イオンの入口は超音速が必要 |
| `MaxwellianEmission(flux_m2_s, temperature_ev)` | 半 Maxwell 放出の総外向き数流束（非負、プロファイルソルバーは正）と温度（正、eV） |
| `BinnedEmission(energy_edges_ev, bin_flux_m2_s)` | 厳密に増加する非負のエネルギー境界 [eV] と各 bin の積分数流束（非負）。bin 内で微分数流束一定 |
| `EmissionSpectrum`（`Protocol`） | 任意の放出スペクトル用プロトコル。`flux_above(barrier_ev)` [$\mathrm{m^{-2}\,s^{-1}}$] と `differential_flux(energy_ev)` [$\mathrm{m^{-2}\,s^{-1}\,eV^{-1}}$]（スカラー／配列対応）を実装する |

`FixedReservoir` の補助メソッド（規格化速度 $u = v / \sqrt{2e/m_e}$、単位 $\sqrt{\mathrm{eV}}$）：

- `reservoir.phase(u)`：規格化速度 $u$ での電子分布（$u$ で積分すると数密度 [$\mathrm{m^{-3}}$]）
- `reservoir.inward_flux(depth)`：深さ `depth` [V] を越えて到達する入射電子数流束 [$\mathrm{m^{-2}\,s^{-1}}$]
- `reservoir.passing_density(depth)`：深さ `depth` [V] での通過背景電子密度 [$\mathrm{m^{-3}}$]
- `reservoir.ion_density(potential_v)`：電位 `potential_v` [V] でのイオン密度 [$\mathrm{m^{-3}}$]

### 診断・プロファイル関数

| 関数 | 動作と引数の制約 |
| --- | --- |
| `classify_branch(reservoir, emission, *, ratio_tolerance=1e-7, scan_points=96) -> BranchCriterion` | 仮の中性谷底を探し、位相空間密度比 $R$ で判定する。`ratio_tolerance > 0`, `scan_points >= 8` |
| `turning_field_difference(reservoir, emission, depth_v, barrier_ev, offset_v) -> float` | 谷から `offset_v` [V] 高い両側での $E_{\mathrm{wall}}^2 - E_{\mathrm{up}}^2$ [$\mathrm{V^2/m^2}$]。$0 \le \text{\texttt{offset\_v}} \le \min(\text{\texttt{depth\_v}}, \text{\texttt{barrier\_ev}})$ |
| `solve_finite_column(reservoir, emission, height_m, *, branch=None, points=2049, order=80) -> FiniteColumnSolution` | 高さ `height_m > 0`・上端 $0\text{ V}$（電場自由）の零電流解。`branch` は `"A"`, `"B"`, `"C"`, または単調解から先に試す `None`。`points >= 129`, `order >= 24`。正の `MaxwellianEmission` が必須 |
| `solve_presheath(reservoir, emission, branch, *, seed=None, nodes=65, order=96, far_tau=16.0) -> PresheathSolution` | 外側領域付きの零電流解。`branch` は `"A"`, `"B_out"`, `"B_nw"`, `"C"`。遠方は準中性・正の `MaxwellianEmission` が必須。`nodes` は 9 以上の奇数、`order >= 24`、`far_tau` は $1$〜$50$ |
| `replacement_targets(reservoir, energy_ev, outer_depth_v=0.0) -> tuple` | 外側領域の置換先（外向き、内向き）分布（単位は `reservoir.phase` と共通）。負エネルギーには速度 $0$ の値、`outer_depth_v > 0` で反射軌道に一次混合を適用 |

### 戻り値クラスと例外

- **`BranchCriterion`**：
  - `outcome`：`"A"` / `"B"` / `"marginal"` / `"no_current"`（放出 0 または障壁なし電子流束 $\le$ イオン流束）/ `"no_minimum"`（走査点で電子過剰がすべて負）/ `"unresolved"`（候補を一意に絞れなかった）
  - `neutral_depth_v`, `emission_barrier_ev`, `ratio`（比 $R$）, `effective_emission_temperature_ev`（$\Gamma_{\mathrm{esc}} / g(B)$）
- **`FiniteColumnSolution`**：
  - `branch`, `z_m`（壁 $0\text{ m} \to$ 上端 $H$）, `potential_v`, `field_v_m`, `escaping_flux_m2_s`, `minimum_height_m`, `monotone_max_height_m`（有限上限がない場合は `inf`）
- **`PresheathSolution`**：
  - `branch`, `wall_potential_v`, `band_potential_v`, `minimum_potential_v`（`B_nw` では `None`）
  - 外側領域：`outer_tau`（準中性接合点 $\tau = 0$ 起点）, `outer_potential_v`
  - 無衝突シース：`sheath_z_m`, `sheath_potential_v`, `sheath_field_v_m`（壁から準中性域へ向かい漸近端の手前で停止）
  - 残差・継続：`current_residual_a_m2`, `outer_neutrality_residual_m3`, `outer_well_seed`（`B_out` のみ）
- **`OuterWellSeed(minimum_depth_v, band_above_minimum_v, minimum_tau, escaping_flux_m2_s, turning_path=())`**：
  - `B_out` の継続用シード。極浅い谷では近隣解の `outer.outer_well_seed` を次の `solve_presheath(..., seed=...)` へ渡すのが有効です。`turning_path` は谷を $0$、準中性域を $1$ とする昇順座標と光学距離の組です。
- **例外**：両プロファイルソルバーは引数不正で `ValueError`、数値探索失敗で `ArithmeticError` を送出します。

---

<a id="補助関数とコマンドライン"></a>

## Part 3：補助関数とコマンドライン

### イオン流体補助関数

- `ion_density_ratio(potential_v, entry_energy_ev, pressure_energy_ev=0.0)`：密度比 $n_i / n_{i\infty}$（スカラーまたは配列対応）
- `ion_critical_potential(entry_energy_ev, pressure_energy_ev=0.0)`：臨界電位 $\phi_{\mathrm{crit}}$ [V]

Fortran と同じ物理式を評価し、定義域外では `ValueError` を送出します。

### コマンドライン（`python -m sheath_model`）

`ZhaoParams` の $J_z = 0$ 解を端末に表示する簡易 CLI です。

```bash
python -m sheath_model --branch A --alpha 60 --electron-drift-mode zero --ion-drift-mode normal --zmax-hat 120
```

| オプション | 選択肢・型 | 既定値 |
| --- | --- | --- |
| `--branch` | `A`, `B`, `C`, `auto` | `auto` |
| `--alpha` | float（太陽高度 [deg]） | `60.0` |
| `--zmax-hat` | float（無次元高さ上限） | `80.0` |
| `--electron-drift-mode` | `full`, `normal`, `zero` | **`full`** |
| `--ion-drift-mode` | `full`, `normal` | **`full`** |

> [!WARNING]
> CLI のドリフトモードの既定値は **`full`** であり、`ZhaoParams` の既定値（`normal`）とも異なります。正ドリフトの Type A / C は棄却されるため（[物理モデル](model.md#ドリフトのある-ac-と上流の帯)）、通常は `--electron-drift-mode zero` を明示してください。
