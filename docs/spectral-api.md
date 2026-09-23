# 任意光電子スペクトルと静的シース評価

Fortran の `use sheath_model` から、光電子源の分布を指定し、試行電位における
Type A/B/C の密度規格化・電場積分・流束・電流を評価できます。
前回状態や時刻を持たない API です。電位基準は上流の 0 V、+z は境界から上流向きで、
`E_z=-dphi/dz` とします。

## 光電子源を指定する

```fortran
type(zhao_plasma_input) :: plasma

! 境界を外向きに通過する法線運動エネルギーの分布
plasma%photoelectrons = binned_photoelectrons( &
    energy_edges_ev=[0.0_dp, 1.0_dp, 3.0_dp, 8.0_dp], &
    bin_flux_m2_s=[2.0e12_dp, 5.0e11_dp, 1.0e11_dp])

! 解析的な半 Maxwell 源を使う場合
plasma%photoelectrons = maxwellian_photoelectrons( &
    density_m3=2.0e7_dp, temperature_ev=2.2_dp)
```

`photoelectron_source` は入力を所有します。コンストラクタで渡した配列を後から変更しても
源は変わりません。入力の検査はモデルの評価・求解時に行い、不正な源には
`SHEATH_INVALID_ARGUMENT` を返します。

| 入力 | 単位・意味 | 制約 |
| --- | --- | --- |
| `energy_edges_ev(N+1)` | 法線運動エネルギー K の bin 境界 [eV] | 有限・非負・厳密な昇順、N ≥ 1 |
| `bin_flux_m2_s(N)` | 各 bin に**積分された**外向き数流束 [m⁻² s⁻¹] | 有限・非負、全てゼロも可 |
| Maxwell `density_m3` | 半 Maxwell 分布の規格化 N_PE [m⁻³] | 有限・非負 |
| Maxwell `temperature_ev` | 光電子温度 [eV] | 有限・正、零放出時も必要 |

bin 内は `g(K)=dGamma/dK=bin_flux/(K_high-K_low)` が一定、指定範囲外はゼロです。
bin 幅・間隔は任意で、最初の境界が 0 より大きくても構いません。
Maxwell 源の放出流束は `N_PE sqrt(2 e T_PE/m_e)/(2 sqrt(pi))` です。
bin 源から密度・温度を推定して Maxwell に近似する処理はありません。

障壁 `B=phi_H-phi_min` を超える `K≥B` が escape、それ未満が return です。
`Gamma_out=Gamma_escape+Gamma_return` で、return 流束は境界への戻りを一回だけ数えます。
一方、A の lower と B の局所密度には、戻る粒子の往路・復路の**両方**を含めます。
同じ `g(K)` を密度・流束・Poisson 積分に使います。

## 電位から評価する

```fortran
program static_example
  use sheath_model
  implicit none
  type(zhao_plasma_input) :: plasma
  type(zhao_state_result) :: state
  integer(i32) :: status
  character(len=256) :: message

  plasma%electron_drift_mps = 0.0_dp
  plasma%photoelectrons = binned_photoelectrons( &
      [0.0_dp, 1.0_dp, 3.0_dp, 8.0_dp], [2.0e12_dp, 5.0e11_dp, 1.0e11_dp])
  call evaluate_sheath_state(plasma, 'B', 0.5_dp, state, status, message)
  if (status /= SHEATH_OK) stop 1
  print *, state%boundary_field_squared_v2_m2, state%net_current_a_m2
  if (state%admissible) print *, state%electric_field_v_m
end program
```

| 枝 | 電位入力 [V] | 出力と条件 |
| --- | --- | --- |
| B | `boundary_potential_v=phi_H>0` | `phi_min=0`、境界電場は正。平坦端点 `phi_H=0` も評価可能 |
| C | `phi_H<0` | `phi_min=phi_H`、境界電場は負 |
| A | `phi_H` と `minimum_potential_v=phi_min<min(phi_H,0)` | lower の境界電場は正、upper の接続残差も返す |

A では `minimum_potential_v` が必須です。B/C への指定や `branch='auto'` は入力エラーです。
A の呼び出し例は `call evaluate_sheath_state(plasma, 'A', phi_H, state, status, message, minimum_potential_v=phi_m)` です。

`zhao_plasma_input` で固定するのは、上流イオン密度・速度、電子温度・ドリフト、粒子質量、光電子源です。
既定値と制約は [E_H 入力表](fortran-api.md#e_h-指定の入力-zhao_field_input) の共通項と同じです。
背景電子 Maxwell 集団の規格化 `N_e` は、指定した電位での上流準中性から求めます。
実際の上流総電子密度を固定するモデルではありません。

| `zhao_state_result` の出力 | 単位・意味 |
| --- | --- |
| `evaluated` | 有限の評価に成功したか |
| `admissible` | 共通の物理解判定に合格したか。A は upper 接続も必要 |
| `physical_status`, `physical_message` | 不採用理由または物理解判定の成功 |
| `branch`, `boundary_potential_v`, `minimum_potential_v` | 評価した枝と電位 [V] |
| `ambient_electron_density_m3` | 上流電子集団の規格化 N_e [m⁻³] |
| `neutrality_residual_m3` | 上流での `n_i-n_e-n_PE` [m⁻³] |
| `boundary_field_squared_v2_m2` | 符号を保持した境界の E² 積分 [V²/m²] |
| `connection_residual_v2_m2` | A の `-2/eps0 ∫(phi_min..0) rho_upper dphi` [V²/m²]、B/C は 0 |
| `electric_field_v_m` | 枝の符号を持つ境界電場 [V/m]。E² 積分が負なら NaN |
| `electron_inward_flux_m2_s`, `ion_inward_flux_m2_s` | 境界への電子・イオン数流束 [m⁻² s⁻¹] |
| `photoelectron_outward_flux_m2_s` | 総放出数流束 [m⁻² s⁻¹] |
| `photoelectron_escape_flux_m2_s`, `photoelectron_return_flux_m2_s` | 上流到達・境界へ戻る数流束 [m⁻² s⁻¹] |
| `net_current_a_m2` | 外向き通常電流 `e*(Gamma_e,in-Gamma_i,in-Gamma_PE,escape)` [A/m²] |

**`status==SHEATH_OK` は、試行状態を評価できたという意味です。**
`state%admissible` が偽でも、符号付き積分や残差は外側の方程式を解くために使用できます。
負の E² をゼロに置き換えません。A は通常の試行点では upper 接続残差が非ゼロで、完成した解ではありません。
許容誤差の範囲で判定するため、`electric_field_v_m` を使う処理では実数電場が得られたことも確認してください。
不正入力・数値評価失敗時には `evaluated=.false.` です。

## E_H 指定の解法にも同じ分布を使う

`zhao_field_input` は `zhao_plasma_input` を継承し、`branch` と `electric_field_v_m` を追加します。

```fortran
type(zhao_field_input) :: input
type(zhao_field_result), allocatable :: candidates(:)

input%zhao_plasma_input = plasma
input%electric_field_v_m = 0.1_dp
call solve_prescribed_field_candidates(input, candidates, status, message)
```

各共通項を `input%electron_temperature_ev` などで直接設定することもできます。
従来の `photoelectron_source_density_m3` と E_H 用の `photoelectron_temperature_ev` は
`input%photoelectrons=maxwellian_photoelectrons(N_PE,T_PE)` に置き換えました。旧フィールドは残していません。
太陽高度を入力する `zhao_equilibrium_input` の J=0 API は、従来どおり Maxwell 源を使います。
全ての解法が同じ密度・流束・積分・物理解判定を呼びます。

## 浮動小数点端点の扱い

狭い bin の平方根差・3/2 乗差は差分を因数分解して評価します。
bin 源の内部電位尺度は T_e 以下で最大の 2 の冪を使い、規格化の往復で bin 端が 1 ulp ずれることを防ぎます。
Type A のスペクトル源の探索・採用判定では、浅い極小の upper 残差を `(-phi_min_hat)^(3/2)` で規格化し、
極小が浅いだけで残差が小さく見える問題を避けます。電子ドリフトが厳密にゼロなら背景電子密度は解析式です。

非常に急な応答では、利用側の方程式を満たす電位が隣接する binary64 値の間に入り得ます。
この場合のために、静的評価は省略可能な `electron_normalization_m3` を受け取ります。
指定できる範囲は、B/C の**現在の電位の直前・直後の表現可能値**で準中性から得た N_e の間だけです。
A と平坦端点には指定できません。通常は省略してください。

この補正を使う場合も、同じ N_e で密度・電場・電流を全て再計算します。
準中性残差と既存の物理許容値はそのまま検査し、電流だけを書き換えません。
範囲外の N_e は入力エラーです。利用側は自身の方程式の残差も、規格化した元の許容値で検査する必要があります。
BEACH の batch-154 スペクトルと独立な高精度静的参照値を回帰テストに含めています。

## 対応範囲と利用側の責務

任意スペクトル、E_H 指定、静的電位評価は Fortran API です。Python は Maxwell 源の J=0 とプロファイルに対応し、
共通モデルの軌道分布・単位・Type B 上流条件・無ドリフト解析式を Fortran と揃えています。

ヒストグラムの生成・蓄積、時間更新式とその根探索、前回状態の保存、候補の選択、MPI・checkpoint は利用側で扱います。
ライブラリの判定は有限の積分・経路検査を含み、全パラメータ域の存在証明や動的安定性の判定ではありません。
電位を走査する実行例は [potential_scan.f90](../example/potential_scan.f90) です。

```bash
fpm run --example potential_scan
```
