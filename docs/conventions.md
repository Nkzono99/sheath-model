# 共通の規約

このページでは、Fortran・Python のすべての API に共通する **単位・座標・符号・解の分類・エラー処理・保証範囲・副作用** のルールをまとめます。各 API リファレンスはこの規約を前提としています。

---

## 1. 単位と命名規則

物理量は原則として **SI 単位** です。例外は次の 3 つだけです。

- 温度（eV）
- 法線運動エネルギー（eV）
- 太陽高度（度）

引数・構造体成分・属性の単位は、名前の末尾の接尾辞で明示しています。

| 接尾辞 | 単位 | 例 |
| --- | --- | --- |
| `_v` | $\mathrm{V}$ | `surface_potential_v`, `minimum_potential_v` |
| `_v_m` | $\mathrm{V/m}$ | `plane_field_v_m`, `electric_field_v_m` |
| `_v2_m2` | $\mathrm{V^2/m^2}$ | `boundary_field_squared_v2_m2`, `connection_residual_v2_m2` |
| `_ev` | $\mathrm{eV}$（温度・法線運動エネルギー） | `electron_temperature_ev`, `energy_edges_ev` |
| `_m3`, `_c_m3` | $\mathrm{m^{-3}}$, $\mathrm{C/m^3}$ | `ion_density_m3`, `charge_c_m3` |
| `_mps` | $\mathrm{m/s}$ | `ion_entry_speed_mps`, `electron_drift_mps` |
| `_m2_s` | $\mathrm{m^{-2}\,s^{-1}}$（粒子数流束） | `ion_inward_flux_m2_s`, `emitted_escape_flux_m2_s` |
| `_a_m2` | $\mathrm{A/m^2}$（電流密度） | `net_current_a_m2` |
| `_m`, `_kg`, `_deg` | $\mathrm{m}$, $\mathrm{kg}$, 度 | `z_m`, `ion_mass_kg`, `sun_elevation_deg` |
| `_hat` | 内部の無次元量 | `minimum_field_squared_hat`, Python の `zmax_hat` |

Fortran の実数型は `real(dp)`（IEEE 754 binary64）、ステータス整数は `integer(i32)`（32 bit 符号付き整数）です。`dp` と `i32` は `use sheath_model` から利用できます。

---

## 2. 座標系・電位基準・符号

```text
  上流（プラズマ側）  z -> +∞   (Zhao モデルでは φ = 0 V, E = 0)
        ^  +z 軸（電場 E_z, 正味電流 J_z, outward / escape 流束は +z 向きが正）
        |
        |   ↓ 背景電子ドリフト v_d, イオン入口速度 u_i∞, inward / return 流束（境界向きが正）
        |
  ======+======  境界 z = 0  （単体解析では電子放出面、結合では matching plane）
                 境界電位 φ_H、境界すぐ上流側の電場 E_H = -dφ/dz|_{z=0+}
```

| 項目 | 規約 |
| --- | --- |
| **境界の位置** | $z = 0$。単体解析では電子を放出する平面、シミュレータ結合では matching plane |
| **$z$ 軸の向き** | $+z$ は境界から上流（プラズマ側）向き |
| **電位の基準（$0\text{ V}$）** | ・**Zhao モデル**：上流の準中性域（$z \to \infty$）<br>・**有限箱**：上端（$z = H$）<br>・**外側領域付きモデル**：遠方リザーバー |
| **電場** | $E_z = -d\phi/dz$（$+z$ 向きが正）。境界電場 $E_H$ は境界のすぐ上流側 $z = 0^+$ の値 |
| **ドリフト・入口速度** | 上流から境界へ向かう（$-z$ 向き）法線速度を正とする（`electron_drift_mps`, `ion_entry_speed_mps`） |
| **粒子数流束** | 名前の向きを正とする（`inward`＝境界へ、`outward`・`escape`＝上流へ、`return`＝境界へ戻る） |
| **正味電流 $J_z$** | $+z$ 向きを正とする通常電流：$J_z = e(\Gamma_{e,\mathrm{in}} - \Gamma_{i,\mathrm{in}} - \Gamma_{\mathrm{em},\mathrm{escape}})$ |

> [!NOTE]
> **正味電流の符号に注意**：$J_z > 0$ は「$+z$ 向きに正電荷が流れる（＝境界へ**負電荷**が流れ込む）」ことを意味します。利用側のコードで「表面へ流れ込む電流（電子流束で負、イオン流束で正）」を正としている場合は、符号を反転して換算してください。
>
> また Python の `SheathProfile.fluxes(z)` が返す `*_signed_m2s` はすべて $+z$ 向きを正とする符号付き流束です（境界へ向かうイオン流束は負値になります）。

---

## 3. 解の分類（Type A / B / C）

Zhao モデルの解は、電位プロファイルの形によって **Type A / B / C** の 3 種類に分類されます。ここで $\phi_H$ は境界電位、$\phi_{\min}$ は境界から上流までを含めた電位の最小値です。

| Type | 電位の形（境界 → 上流） | $\phi_H$ と $\phi_{\min}$ の関係 | 境界電場 $E_H$ の符号 | よくある状況 |
| :---: | --- | --- | :---: | --- |
| **A** | 内部に電位の極小（谷）を持つ（非単調） | $\phi_{\min} < \min(\phi_H, 0)$（$\phi_H$ は負でも可） | 正（$E_H > 0$） | 電子放出が強いとき |
| **B** | 正の境界電位から $0\text{ V}$ へ単調減少 | $\phi_H > 0,\ \phi_{\min} = 0$（上流） | $0$ 以上（$E_H \ge 0$） | 電子放出が中程度のとき |
| **C** | 負の境界電位から $0\text{ V}$ へ単調増加 | $\phi_H < 0,\ \phi_{\min} = \phi_H$（境界） | $0$ 以下（$E_H \le 0$） | 電子放出が弱い／ないとき |

- `branch` 引数には `'A'`、`'B'`、`'C'`、`'auto'` を指定します。
  - Fortran は大文字・小文字を区別しません。
  - Python の `SheathSolver` は `"A"`、`"B"`、`"C"`、`"auto"` を受け付けます。
- `'auto'` の試行順序は閉じ方・入力型によって異なります（[Fortran API](fortran-api.md#求解メソッド)・[Python API](python-api.md#sheathsolver)）。
- 各 Type に存在する粒子集団の詳細は [物理モデル](model.md#電位の形type-abc) を、固定入射モデルの枝（`A`, `B_out`, `B_nw`, `C`）は [遠方の電子分布を固定するモデル](model.md#fixed-reservoir) を参照してください。

---

<a id="ステータスとエラー処理"></a>

## 4. ステータスとエラー処理

### Fortran（ステータスコード方式）

Fortran の公開手続きは例外を投げず、プロセス終了（`stop`）も行いません。代わりに `status`（`integer(i32)`）と `message`（`character(len=*)`）を返します。

> [!IMPORTANT]
> **結果を参照する前に、必ず `status == SHEATH_OK` を確認してください。**

| 定数 | 値 | 意味 |
| --- | ---: | --- |
| `SHEATH_OK` | `0` | 成功。`message` に探索に関する補足説明が入ることがある |
| `SHEATH_INVALID_ARGUMENT` | `1` | 入力値・設定・評価点が対応範囲外 |
| `SHEATH_NO_PHYSICAL_SOLUTION` | `2` | 探索した Type や候補がすべて物理条件で除外された（大域的な不存在の証明ではない） |
| `SHEATH_NUMERICAL_FAILURE` | `3` | 非線形反復が未収束、または有限値検査・数値積分に失敗した |
| `SHEATH_AMBIGUOUS_SOLUTION` | `4` | 単一解を要求したが、物理解が複数見つかった（`solve_prescribed_field`） |

**失敗時の出力と `message` の長さ**：

- 単一の結果型は `valid = .false.`、プロファイル結果の配列成分は未確保になります。
- 候補配列は、$J_z = 0$ の `solve_equilibrium_candidates` と `matching_states_at` では**長さ 0 で確保済み**、$E_H$ 指定の `solve_prescribed_field_candidates` では**未確保**になります（[Fortran API](fortran-api.md#求解メソッド)）。
- `message` は渡した文字列長で切り詰められます。通常は **256 文字以上**、試行した Type ごとの理由を連結する [`sheath_matching_iteration`](coupling.md#外部解を更新する) では **512 文字程度** を推奨します。

### Python（例外方式）

Python API は失敗を例外で通知します。

| 例外 | 発生する場面 |
| --- | --- |
| `SearchFailure` | `SheathSolver` による Zhao モデルの探索失敗。診断オブジェクトを `exc.diagnostics` に保持する |
| `ArithmeticError` | 固定入射モデルのプロファイルソルバー（`solve_finite_column`, `solve_presheath`）での数値探索失敗 |
| `ValueError` / `TypeError` | 引数の値域外・型不一致・物理制約違反 |

---

## 5. 収束・物理解・安定性の区別

本ライブラリは、次の 3 つを明確に区別します。

1. **代数方程式の収束**  
   上流の準中性条件や閉じ方（零電流または電場指定）の残差が許容値以下になった状態。これだけでは有効な解としません。
2. **物理解（admissible solution）**  
   収束した根が、上流へ連続するイオン流体の枝と、モデルの境界条件を満たす実数の電場（$E^2 \ge 0$ 等）を持つ状態（判定条件の詳細は [物理モデル](model.md#物理解の判定)）。**ライブラリが返す解は必ずこの条件を満たします。**
3. **動的安定性**  
   **判定しません。** 同じ条件に複数の物理解が共存する場合（例：Type A と Type B の共存）、どれが物理的に実現するかは選択しません。

> [!NOTE]
> **有限探索の限界**：根の探索は有限個の初期値・反復回数・候補上限で行います。したがって、探索の失敗（や Type マップの灰色領域）は「解が存在しないことの証明」ではなく、候補列挙の結果も「すべての根を網羅したことの証明」ではありません。

---

## 6. 状態・並列性・副作用

- **グローバル状態なし**：ライブラリはモジュール変数や `save` 変数を一切持ちません。標準出力・ファイル出力・`stop` も行いません。
- **ソルバーは無状態・読み取り専用**：`sheath_solver`（Fortran）および `SheathSolver`（Python）は数値設定と明示的に渡した解マップだけを保持します。求解メソッドはソルバー自身を変更せず、前回の解を記憶することもありません。物理入力は呼び出しごとに渡します。
- **状態を持つオブジェクト**：呼び出し側が明示的に所有する [`sheath_matching_iteration`](coupling.md#外部解を更新する) と解マップ（Atlas）だけです。
- **スレッド安全・並列呼び出し可能**：独立な入力は OpenMP 等で並列に解けます（実例：[example/readme_data.f90](../example/readme_data.f90)）。
- **I/O の管理**：ファイル読み書きを行うのは解マップの `write`/`read` と `sheath_matching_iteration` の `write_state`/`read_state` のみで、呼び出し側が開閉管理する unit（またはパス）を使用します。
