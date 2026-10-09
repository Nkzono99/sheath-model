# Python API

Python は Maxwell 光電子源の J=0 モデルに対応します。
物理条件を `FixedEntryParams` または `ZhaoParams`、数値設定を `SheathSolver` に渡します。
温度は eV、固定入口の密度・速度・質量は SI。入口速度は内向きが正で、Bohm 速度への自動補正はありません。

```python
from sheath_model import FixedEntryParams, SheathSolver, SearchOptions, ProfileOptions

inputs = FixedEntryParams(
    ion_density_m3=8.7e6,
    ion_entry_speed_mps=405299.88897111727,
    ion_temperature_ev=12., ion_pressure_factor=3.,
    electron_temperature_ev=12., electron_drift_mps=0.,
    photoelectron_density_m3=55.42562584220407e6,
    photoelectron_temperature_ev=2.2,
)
solver = SheathSolver(search=SearchOptions(method="newton"),
                      profile=ProfileOptions(zmax_hat=120.))
root = solver.solve_equilibrium(inputs, branch="A")
profile = solver.build_profile(root)
print(root.surface_potential_v, root.minimum_potential_v, root.residual_norm)
state = profile.sample(0.)
fluxes = profile.fluxes(0.)
vdf = profile.vdf(0., species="swe")
```

Solver は別の物理入力にも再利用でき、前回解を保持しません。
`solver.solve_profile(inputs, branch="A")` は求解と再構成をまとめて実行します。
`build_profile(root)` は解探索を繰り返さず、root に保持した物理条件で再構成します。
入力、Solver、解、プロファイルは属性の書き換えを禁止し、変更には `dataclasses.replace` を使います。

## 入力・出力

`FixedEntryParams` は法線状態と光電子 Maxwell 分布の規格化密度を直接指定します。
`ZhaoParams` は太陽高度・全風速・垂直入射光電子密度から同じ法線状態を計算します。
例えば `ZhaoParams(sun_elevation_deg=60., electron_drift_mode="zero")` を同じ Solver に渡せます。
Zhao の密度は `ion_density_m3` と `photoelectron_reference_density_m3`、温度は
`electron_temperature_ev`、`ion_temperature_ev`、`photoelectron_temperature_ev` です。
粒子質量は両入力とも `ion_mass_kg` と `electron_mass_kg`。既定値は陽子・電子質量です。
物理条件と制約は [運動論モデル](kinetic-model.md) を参照してください。

`EquilibriumResult` は inputs、branch、surface_potential_v、minimum_potential_v、
ambient_electron_density_m3、residual_norm、diagnostics、upstream_negative_band_v を持ちます。
upstream_negative_band_v は上流に接する $E^2<0$ の電位幅 [V] で、upstream_band_tolerance で受理した根だけが 0 より大きくなります。
電子密度は Maxwell 分布の規格化密度です。
`root.density(potential_v, side="upper")` で局所密度を SI 単位で評価できます。A は lower/upper を選びます。

`SheathProfile` は equilibrium、z_m、potential_v、electric_field_v_m、density、turning_height_m を持ちます。
配列は読み取り専用です。密度は ion_m3、electron_free_m3、electron_reflected_m3、
photoelectron_free_m3、photoelectron_captured_m3 と charge_c_m3。
sample/fluxes/vdf の位置は既定で m、無次元位置には `unit="hat"` を明示します。
計算した区間外の位置は ValueError。温かいイオンの流体モデルはイオン VDF を定義しません。

## 探索と解マップ

`SearchOptions` の method は auto/newton/lm、明示した B/C のみ bracket も可。
主要な既定値は residual_tolerance=1e-10、max_iterations=100、max_starts=32、use_default_guesses=True、
upstream_band_tolerance=0（Fortran と同じ意味）。
`ProfileOptions` は zmax_hat=80、n_profile_grid=600、n_type_a_grid=8000、
profile_phi_tol_hat=1e-3、type_a_phi_m_eps_hat=1e-5。
`ContinuationOptions(method="parameter")` または method="arclength" を Solver に設定します。

```python
from dataclasses import replace
from sheath_model import EquilibriumAtlas, ContinuationOptions

sweep = [replace(inputs, photoelectron_density_m3=value) for value in (54e6, 57e6)]
atlas = solver.build_equilibrium_atlas(sweep, branches=("A",))
atlas.save("equilibrium-atlas.txt")
worker = replace(solver, equilibrium_atlas=EquilibriumAtlas.load("equilibrium-atlas.txt"),
                 continuation=ContinuationOptions(method="arclength"))
query = worker.solve_equilibrium(inputs, branch="A", initial_guess=root)
found = worker.solve_equilibrium_candidates(inputs, branch="A", deflation=True)
print(len(found.candidates), found.diagnostics.atlas_hits)
```

initial_guess は以前の EquilibriumResult。マップの推定値とともに、新しい方程式で解き直します。
auto は Zhao 高度 20 度未満で C→A→B、その他で A→B→C の最初の物理解を返します。
候補探索は `CandidateSet(candidates, diagnostics)` を返し、候補の既定上限は Type ごとに 16。
deflation の既定値は true で、有限探索による候補集合に安定性の順位はありません。

マップは呼び出し側が所有し、Solver は参照します。求解による自動登録はありません。
`atlas.add(root)` は元の方程式とプロファイルを検査して明示登録します。
構築後の atlas.attempts は accepted/excluded/unresolved を区別し、空白から不存在を推定しません。
表の無次元形式は Fortran の J=0 表と共通です。bin スペクトルの記録は読み込めますが、Maxwell の問い合わせに流用しません。

失敗は `SearchFailure`。exception.diagnostics と解の diagnostics は A/B/C 順の検索結果です。
数値収束、物理プロファイルの存在、動的安定性は別の概念です。
