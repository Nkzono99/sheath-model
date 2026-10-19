# 図と計算例

[README](../README.md) と各文書の説明図、および公開 Fortran API の計算結果を Matplotlib で描画したグラフをまとめます。
説明図の用途と、計算グラフの条件・元データ・再生成手順は以下のとおりです。

## 概念図

| 画像 | 読み方・関連する文書 |
| --- | --- |
| [境界と粒子集団](figures/sheath_overview.png) | Type A の入射・反射・放出・脱出。[物理モデル](model.md#対象とする系) |
| [計算領域と外部シースの結合](figures/matching_plane_coupling.png) | matching plane を境にした領域の分担と、観測窓ごとの条件・流束の受け渡し。[シミュレータとの結合](coupling.md#構成) |

概念図は画像生成で作成し、本文の粒子集団・座標・結合手順と照合しています。
位置・粒子数・矢印は縮尺を持たず、矢印の横方向の広がりは集団や流れを区別するための表現です。
以下の計算グラフは、保存済みの数値データから再生成できます。

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
仮定と制約は [物理モデル](model.md) を参照してください。

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
E_H 側は枝ごとの `sheath_search_diagnostics` から未解決・棄却のビットを記録し、
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

## 実行例の一覧

リポジトリ直下で実行します。KUDPC 等の共有環境では計算ノード割当内で実行してください。

### Fortran（`fpm run --example <名前>`）

| 名前 | 内容 | 関連する文書 |
| --- | --- | --- |
| [fixed_entry](../example/fixed_entry.f90) | 固定入口・温かいイオンの J=0 を解き、静的評価で境界電場を得る | [J=0 の入力](fortran-api.md#j0-の入力) |
| [compare_closures](../example/compare_closures.f90) | 同じ条件の J=0 と E_H 指定を比べる | [求解メソッド](fortran-api.md#求解メソッド) |
| [equilibrium_profile](../example/equilibrium_profile.f90) | J=0 のプロファイル（高さ・電位・電場・電荷密度）を CSV で出力 | [プロファイル](fortran-api.md#プロファイル) |
| [field_sweep](../example/field_sweep.f90) | E_H を掃引し、前の候補を初期値に加えて全候補と診断を出力 | [探索の診断](fortran-api.md#探索の診断) |
| [potential_scan](../example/potential_scan.f90) | bin 放出源の Type B を電位で走査し、符号付き E²・電場・N_e・電流・判定を CSV で出力 | [静的評価](fortran-api.md#静的評価) |
| [solution_atlas](../example/solution_atlas.f90) | J=0 の解マップを作り（引数のファイルへ保存・読込）、問い合わせと候補探索に使う | [解マップ](fortran-api.md#解マップと-continuation) |
| [field_atlas](../example/field_atlas.f90) | E_H 指定の解マップと arclength continuation | 同上 |
| [matching_plane](../example/matching_plane.f90) | 結合: 面上の条件、流出スペクトルからの放出源、外部解の更新と checkpoint の往復 | [シミュレータとの結合](coupling.md) |
| [readme_data](../example/readme_data.f90) | README の図のデータを計算する（[図を再生成する](#図を再生成する)） | |

`equilibrium_profile` は標準のプロファイル設定を使うため、README の図とは遠方の打ち切りが異なります。

### Python（`python examples/<名前>.py`）

| 名前 | 内容 |
| --- | --- |
| [fixed_entry.py](../examples/fixed_entry.py) | Fortran の `fixed_entry` と同じ条件を解き、境界の流束を表示 |
| [plot_profiles.py](../examples/plot_profiles.py) | Zhao 入力の A/B/C プロファイルを `profiles.png` に描く |
| [solution_atlas.py](../examples/solution_atlas.py) | 解マップの構築・保存・読込と候補探索 |
| [plot_readme_figures.py](../examples/plot_readme_figures.py) | README の図を CSV から描く |
