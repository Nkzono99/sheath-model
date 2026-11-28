# 導入と最初の計算

## Fortran

Fortran 2008（submodule を含む）に対応したコンパイラと [fpm](https://fpm.fortran-lang.org/) が必要です。
ライブラリに外部依存はありません。

### 最初の計算

リポジトリ直下で実行します。

```bash
fpm run --example fixed_entry
```

固定入口・温かいイオンの条件で零電流の Type A を解き、表面電位・極小電位・表面電場・イオンの入射数流束を表示します。
この例の表面電位は約 3.840 V、極小電位は約 −0.332 V です。
コードでは `fixed_entry_equilibrium_input` に物理入力、`sheath_solver` に探索設定を指定しています。[fixed_entry.f90](../example/fixed_entry.f90)

高さごとの電位・電場・電荷密度を出力する例もあります。[equilibrium_profile.f90](../example/equilibrium_profile.f90)

### fpm の依存として追加する

利用側のコードでは `use sheath_model` を使います。
`fpm.toml` で依存の版をリリースタグ（`tag`）かコミット（`rev`）に固定します。以下はタグの指定例です。
接合面の電場・電流で候補を選ぶ機能を使う場合は、その機能を含む版を指定してください。[結合 API](coupling.md#combined-matching)

```toml
[dependencies]
sheath-model = { git = "https://github.com/Nkzono99/sheath-model.git", tag = "v0.2.1" }
```

ローカルのソースを使う場合は相対パスを指定します。

```toml
[dependencies]
sheath-model = { path = "../sheath-model" }
```

### Make・CMake からリンクする

利用側と同じコンパイラ・同じ版を指定して、ライブラリと `.mod` をインストールします。以下は GNU Fortran を使う例です。

```bash
fpm install --profile release --compiler gfortran --prefix /path/to/sheath-model-install
gfortran -I/path/to/sheath-model-install/include -c host.f90
gfortran host.o /path/to/sheath-model-install/lib/libsheath-model.a -o host
```

入力型と求解メソッドは [Fortran API](fortran-api.md) にあります。

## Python

Python 3.10 以上が必要です。インストール時に NumPy と SciPy が入ります。
リポジトリ直下で次を実行すると、Fortran の例と同じ条件を解きます。[fixed_entry.py](../examples/fixed_entry.py)

```bash
python -m pip install -e .
python examples/fixed_entry.py
```

Python は NumPy/SciPy による独立した実装です。描画用の Matplotlib は `python -m pip install -e '.[plot]'` で追加できます。
太陽高度から条件を与える場合は [ZhaoParams](python-api.md#zhaoparams) を使います。
遠方の電子分布を固定する場合は [固定入射モデルの計算例](python-api.md#fixed-reservoir) を参照してください。

## テストと実行例

```bash
fpm test
python -m unittest discover -s tests -v
```

[単位・座標・符号](conventions.md) · [検証内容とコンパイラの記録](numerics.md#検証) · [実行例の一覧](examples.md#実行例の一覧)
