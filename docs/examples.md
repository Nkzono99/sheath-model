# 図と計算例

[README](../README.md) の図は、公開 Fortran API の計算結果を Matplotlib で描画したものです。
ここでは計算条件、元データ、再生成手順をまとめます。

## 共通の計算条件

| 量 | 値・定義 |
| --- | --- |
| 上流イオン密度 `n_i` | 8.7 cm⁻³ |
| 背景電子温度 `T_e` | 12 eV |
| 光電子温度 `T_pe` | 2.2 eV |
| イオン種 | 陽子 |
| 太陽風速度 `v_sw` | 468 km/s |
| 法線イオン速度 `v_i` | `v_sw sin(alpha)` |
| 電子の法線ドリフト | 0 |
| 光電子源規格化密度 `n_pe,0` | `n_pe,ref sin(alpha)` |
| 電位基準と座標 | 上流で 0 V、+z は表面から上流向き、`E_z=-dphi/dz` |

`alpha` は太陽高度です。背景電子 Maxwell 分布の規格化は両モデルとも未知量で、
実際の上流総電子密度とは異なります。
電子ドリフトを正にすると A/C の半無限上流条件が成立しなくなるため、図の範囲をそのまま適用できません。
仮定と制約は [運動論モデル](kinetic-model.md) を参照してください。

<a id="profiles"></a>

## Type A/B/C のプロファイル

`J=0`、`n_pe,ref=64 cm⁻³` として、A/B/C をそれぞれ指定して解いた例です。

| Type | 太陽高度 | 表面電位 | 最小電位 |
| --- | ---: | ---: | --- |
| A | 60° | 3.840 V | −0.332 V（内部） |
| B | 20° | 1.616 V | 0 V（上流） |
| C | 10° | −4.238 V | −4.238 V（表面） |

A は `phi_min < min(phi_H,0)` を満たす非単調解で、負の表面電位も許します。
掲載したのは正の表面電位を持つ例です。
図は最初の 60 m を表示しています。計算は遠方の電位打ち切り `|phi|≈2.2×10⁻⁴ V` まで行い、
その先に人工的なゼロ電位の線は追加していません。保存された全範囲は CSV で確認できます。

| ファイル | 内容 |
| --- | --- |
| [PNG](figures/sheath_profiles.png) / [PDF](figures/sheath_profiles.pdf) | 3 つの Type の電位・電場の比較図 |
| [Type A](figures/data/profile_A.csv) / [Type B](figures/data/profile_B.csv) / [Type C](figures/data/profile_C.csv) | 高さごとの電位・電場・電荷密度 |
| [profile_metadata.csv](figures/data/profile_metadata.csv) | 各例の条件、極小点、電流、解の残差 |

<a id="maps"></a>

## 解の Type マップ

縦軸は光電子源の強さ `r=n_pe,ref/n_i`、対数軸です。
各パネルは **257 × 129 点**、合計 66,306 点の計算結果です。

| | 左：J=0 | 右：E_H 指定 |
| --- | --- | --- |
| 横軸 | 太陽高度 1〜89° | 法線電場 −2〜2 V/m |
| 横軸の間隔 | 0.34375° | 0.015625 V/m |
| 縦軸 | `r=0.5〜16`、129 点の対数等間隔 | 同左 |
| 太陽高度 | 横軸で走査 | 20° に固定 |
| 探索 | `solve_equilibrium` で A/B/C を個別に指定 | `solve_prescribed_field_candidates` で候補を取得 |
| 電流 | 零電流を条件として課す | 結果として返す |

色は**この探索で見つかった物理解の Type**を示します。
A+B などは同じ条件で複数の Type が得られたことを表し、安定性による選択はしていません。
左の A/B/C マーカーはプロファイル例の位置です。

- **灰色（Candidates rejected）**：採用解がなく、試みた候補が物理条件で棄却された点。
- **薄灰色（No root resolved）**：採用解がなく、少なくとも一つの枝の数値探索が未解決の点。
- **Type の色**：採用解が見つかった点。他の枝の探索が未解決の場合もあります。

有限個の初期値からの探索結果です。灰色領域も含めて、解の不存在や全根の発見を保証しません。
代数根の収束と物理プロファイルの存在は別に判定しています。

| ファイル | 内容 |
| --- | --- |
| [PNG](figures/sheath_type_maps.png) / [PDF](figures/sheath_type_maps.pdf) | 2 つの境界条件の Type マップ |
| [equilibrium_map.csv](figures/data/equilibrium_map.csv) | J=0 の格子データ |
| [field_map.csv](figures/data/field_map.csv) | E_H 指定の格子データ |
| [summary.json](figures/data/summary.json) | 固定入力、代表解、Type ごとの格子点数 |

CSV の Type はビット値 `A=1, B=2, C=4` の和です。例えば `types_found=3` は A+B を表します。
`unresolved_types` と `rejected_types` も同じビット表現を使い、採用解の有無とは別に探索結果を記録します。
`candidate_count` は報告された候補数です。
E_H 側は枝ごとの `zhao_field_search_diagnostics` から未解決・棄却のビットを記録し、
採用候補がある格子点でも未解決の初期値を保持します。
掲載マップでは各格子点を独立に探索し、近隣解は初期値に使っていません。
低電場の浅い Type A や A/B の境界付近では初期値による検出差が残ります。
連続したパラメータ掃引では [field_sweep.f90](../example/field_sweep.f90) のように
近隣解も渡すことで、独立探索で漏れる枝を回収できる場合があります。

## 図を再生成する

リポジトリを取得し、Fortran 2008 対応コンパイラ、fpm、Python 3.10+ を用意してください。
以下はリポジトリ直下で実行します。

```bash
python -m pip install -e '.[plot]'
```

**保存済みの CSV から描画するだけなら、次のコマンドで再生成できます。**

```bash
python examples/plot_readme_figures.py
```

電位・電場の PNG / PDF、Type マップの PNG / PDF、集計 JSON を `docs/figures/` に書き出します。
別の出力先を使う場合は `--output build/readme-figures` を追加してください。
入力データの場所は `--data` で変更できます。

計算から再実行する場合は、公開 API を呼ぶ [readme_data.f90](../example/readme_data.f90) を使用します。
次の例は GNU Fortran / OpenMP で 16 コアを使用します。

```bash
mkdir -p docs/figures/data
OMP_NUM_THREADS=16 OMP_PROC_BIND=false \
  fpm run --example readme_data --compiler gfortran --profile release \
  --flag "-fopenmp" -- docs/figures/data 257 129
python examples/plot_readme_figures.py
```

計算コマンドの末尾に `equilibrium` または `field` を付けると、指定したカラーマップだけを再計算できます。
引数を省略した既定の格子も 257 × 129 点です。描画処理は
[plot_readme_figures.py](../examples/plot_readme_figures.py) にあります。

KUDPC 等の共有環境では、ビルド・ソルバー・テスト・描画を計算ノード割当内で実行してください。
上記の並列計算では、割り当てたコア数と `OMP_NUM_THREADS` を揃えます。

## その他の実行例

### Fortran

```bash
fpm build
fpm test
fpm run --example compare_closures
fpm run --example field_sweep
fpm run --example equilibrium_profile
fpm install --prefix ./install
```

[compare_closures.f90](../example/compare_closures.f90) は J=0 と E_H 指定の比較、
[field_sweep.f90](../example/field_sweep.f90) は近隣解を初期値に加える電場掃引、
[equilibrium_profile.f90](../example/equilibrium_profile.f90) は J=0 の高さ・電位・電場・電荷密度の CSV 出力です。
後者は標準のプロファイル設定を使うため、README の図とは遠方の打ち切りが異なります。

E_H 指定の `solve_prescribed_field` は、検出した物理解が一つの場合に結果を返し、
複数ある場合は `sheath_ambiguous_solution` を返します。安定性による選択は行いません。
候補を比較したい場合は `solve_prescribed_field_candidates` を使ってください。
ライブラリ自体はプロセス終了やファイル出力を行わず、ステータスと結果を返します。

ローカルのソースを別の fpm プロジェクトから使う場合は、Git 依存の代わりにパスを指定できます。

```toml
[dependencies]
sheath-model = { path = "../sheath-model" }
```

### Python

```bash
python -m unittest discover -s tests -v
python -m sheath_model --branch A --alpha 60 \
  --electron-drift-mode zero --ion-drift-mode normal --zmax-hat 120
python examples/plot_profiles.py
```

Python と Fortran はともに半無限領域の一次積分からプロファイルを構成します。
局所流束・速度分布の診断の式と仮定は [Algorithm notes](algorithm.md) を参照してください。
