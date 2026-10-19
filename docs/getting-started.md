# 導入と最初の計算

sheath-model を自分の Fortran・Python のコードから使えるようにし、最初のシースを解くまでの手順です。
単位・符号・ステータスの扱いは [共通の規約](conventions.md) にまとめています。

## 必要なもの

| 用途 | 必要なもの |
| --- | --- |
| Fortran ライブラリ | Fortran 2008（submodule を含む）に対応したコンパイラと [fpm](https://fpm.fortran-lang.org/) |
| Python パッケージ | Python 3.10 以上、NumPy 1.24 以上、SciPy 1.10 以上（図は Matplotlib 3.8 以上） |

GNU Fortran 8.5 で `fpm test` の全件が通ることを確認しています。Intel ifx 2023.2 でもビルド・実行できますが、`test_delegated_physics` の "spectral A field inversion"（bin 源の Type A を、静的評価で得た境界電場を指定して解き直す検査）が失敗する既知の問題があります。
外部ライブラリには依存しません。

## Fortran プロジェクトから使う

### fpm の依存として追加する

`fpm.toml` に依存を追加します。再現性のため、`rev` で commit を固定してください。

```toml
[dependencies]
sheath-model = { git = "https://github.com/Nkzono99/sheath-model.git", rev = "<commit>" }
```

ローカルのソースを使う場合は相対パスで指定します（fpm は絶対パスを正しく扱わないことがあります）。

```toml
[dependencies]
sheath-model = { path = "../sheath-model" }
```

公開済みのタグ `v0.1.0` は、軌道保存・物理解の判定・スペクトル源・結合 API を含まない初期版です。`main` の commit を使ってください。

### fpm 以外のビルドから使う

Make や CMake でビルドするシミュレータでは、fpm でライブラリとモジュールを書き出してからリンクします。
ホストのプログラムと同じコンパイラ・同じ版でビルドしてください（`.mod` はコンパイラ間で互換性がありません）。

```bash
fpm install --profile release --compiler ifx --prefix /path/to/sheath-model-install
```

`<prefix>/include` に `.mod`、`<prefix>/lib/libsheath-model.a` にライブラリが入ります。

```bash
ifx -I/path/to/sheath-model-install/include -c host.f90
ifx host.o /path/to/sheath-model-install/lib/libsheath-model.a -o host
```

利用側のコードは `use sheath_model` だけを書きます。他のモジュールは内部用です。

## 最初の計算

太陽風と光電子の固定入口条件で、零電流（J=0）の Type A を解き、プロファイルを作る例です。

```fortran
program first_sheath
  use sheath_model
  implicit none
  type(sheath_solver) :: solver
  type(fixed_entry_equilibrium_input) :: input
  type(sheath_equilibrium_result) :: root
  type(sheath_profile_result) :: profile
  integer(i32) :: status
  character(len=256) :: message

  ! 物理条件（SI、温度は eV）。速度は境界へ向かう向きが正
  input%branch = 'A'
  input%plasma%ion_density_m3 = 8.7e6_dp
  input%plasma%ion_entry_speed_mps = 405299.88897111727_dp
  input%plasma%electron_temperature_ev = 12.0_dp
  input%plasma%electron_drift_mps = 0.0_dp
  input%plasma%emission = maxwellian_emission(55.42562584220407e6_dp, 2.2_dp)

  ! 数値設定は solver に置く
  solver%profile%max_distance_m = 60.0_dp

  call solver%solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_OK) then
    print *, trim(message)
    error stop 1
  end if
  call solver%build_profile(input, root, profile, status, message)
  if (status /= SHEATH_OK) then
    print *, trim(message)
    error stop 1
  end if

  print '(a,f10.5)', 'Surface potential [V]: ', root%surface_potential_v
  print '(a,f10.5)', 'Minimum potential [V]: ', root%minimum_potential_v
  print '(a,es12.4)', 'Surface field [V/m]:   ', profile%electric_field_v_m(1)
  print '(a,es12.4)', 'Net current [A/m^2]:   ', root%net_current_a_m2
end program first_sheath
```

表面電位約 3.84 V、極小約 −0.33 V の Type A が得られます。

- 物理条件は入力型、数値設定は `sheath_solver`、結果は結果型に分かれています。solver は前回の解を持たないので、別の条件にも使い回せます。
- `branch='auto'` にすると A→B→C の順に試し、最初に見つかった物理解を返します。複数の解を比べるには
  `solve_equilibrium_candidates` を使います（[求解メソッド](fortran-api.md#求解メソッド)）。
- 背景電子のドリフト（既定値 0）を正にすると、ドリフトのある A/C は既定では棄却されます（[物理モデル](model.md#ドリフトのある-ac-と上流の帯)）。

太陽高度と全風速から条件を作る場合は `zhao_equilibrium_input` を、境界電場を指定する場合は `prescribed_field_input` を使います。
任意の放出電子スペクトルは `binned_emission` で与えます（[Fortran API](fortran-api.md#物理入力)）。

## Python から使う

リポジトリ直下でインストールします。

```bash
python -m pip install -e '.[plot]'
```

```python
from sheath_model import ZhaoParams, SheathSolver, ProfileOptions

inputs = ZhaoParams(sun_elevation_deg=60.0, electron_drift_mode="zero")
solver = SheathSolver(profile=ProfileOptions(zmax_hat=120.0))
profile = solver.solve_profile(inputs, branch="A")
print(profile.equilibrium.surface_potential_v, profile.equilibrium.minimum_potential_v)
```

Python は Maxwell 放出源の J=0 モデルだけを持ちます。詳細は [Python API](python-api.md) を参照してください。

## テストと実行例

```bash
fpm test
python -m unittest discover -s tests -v
fpm run --example compare_closures
```

KUDPC などの共有環境では、ビルド・テスト・計算をログインノードではなく計算ノードの割当内で実行してください。
実行例の一覧は [図と計算例](examples.md#実行例の一覧) にあります。

## 次に読む

| 目的 | 文書 |
| --- | --- |
| 方程式と仮定、解の判定条件を知る | [物理モデル](model.md) |
| 入力・結果・設定の全項目を調べる | [Fortran API](fortran-api.md) / [Python API](python-api.md) |
| 粒子シミュレータの境界につなぐ | [シミュレータとの結合](coupling.md) |
| 探索がうまくいかないときに設定を調べる | [数値解法と検証](numerics.md) |
