# sheath-model

電子を放出する平面上の一次元・定常シースを解く Fortran ライブラリと Python パッケージです。
境界の電位、電位・電場のプロファイル、粒子流束を求めます。

## モデルを選ぶ

| 与える条件 | 求めるもの | 実装 |
| --- | --- | --- |
| 準中性の上流へ接続する | 背景電子の規格化と Type A/B/C のシース。[Zhao モデル](docs/model.md) | Fortran と Python で零電流。Fortran は境界電場の指定と静的評価にも対応 |
| 遠方の電子分布を固定する | 谷の診断、上端 0 V の有限箱、準中性の外側領域へつなぐ解。[固定入射モデル](docs/model.md#fixed-reservoir) | Python |
| 粒子シミュレータの流出を使う | 接合面の電位・電場・流束。零電流での更新や、内外の電場差と電流残差での候補選択。[シミュレータとの結合](docs/coupling.md) | Fortran |

## 使い始める

リポジトリ直下で、零電流の Type A を解く例を実行できます。

```bash
fpm run --example fixed_entry
```

```bash
python -m pip install -e .
python examples/fixed_entry.py
```

必要な環境と、Fortran プロジェクトに依存を追加する手順は [導入と最初の計算](docs/getting-started.md) にあります。

## 計算例

太陽高度を変えたときの Type A/B/C の電位と電場です。

[![Type A は内部に電位の極小を持ち、Type B/C は表面から上流へ単調に変化する。上段は電位、下段は電場。](docs/figures/sheath_profiles.png)](docs/figures/sheath_profiles.pdf)

[プロファイルの計算条件と CSV](docs/examples.md#profiles) · [解の Type マップ](docs/examples.md#maps) · [実行例の一覧](docs/examples.md#実行例の一覧)

## リファレンス

[Fortran API](docs/fortran-api.md) · [Python API](docs/python-api.md) · [単位・座標・符号](docs/conventions.md) · [数値解法と検証](docs/numerics.md)

## ライセンス

Python 実装と新規の公開窓口は [MIT](LICENSE)、BEACH 由来の数値実装は [Apache-2.0](LICENSES/Apache-2.0.txt) です。出典は [NOTICE](NOTICE) を参照してください。
