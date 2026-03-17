# zhao-sheath

Zhao et al. の月面光電子シース半解析モデル（Type A / B / C）を Python で解くための最小リポジトリです。

## 含まれるもの

- `src/zhao_sheath/solver.py`
  - `ZhaoParams`
  - `ZhaoSheathSolver`
- `examples/plot_profiles.py`
  - 基本的なプロファイル描画例
- `pyproject.toml`
  - 最小パッケージ設定

## 前提

- Python 3.10+
- NumPy
- SciPy
- Matplotlib（`examples/` の描画時のみ）

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

## 使い方

### Python から使う

```python
from zhao_sheath import ZhaoParams, ZhaoSheathSolver

params = ZhaoParams(alpha_deg=60.0, zmax_hat=120.0)
solver = ZhaoSheathSolver(params)
out = solver.solve_profile("A")

print(out["phi0_V"], out["phi_m_V"], out["n_swe_inf_m3"])
```

### モジュールとして実行する

```bash
python -m zhao_sheath --branch A --alpha 60 --zmax-hat 120
```

### 例を描画する

```bash
python examples/plot_profiles.py
```

## 典型的な構成方針

- **Type A**: 非単調ポテンシャル。未知数 `(phi0, phi_m, n_swe_inf)` を解いた後、一次積分ベースでプロファイルを再構成。
- **Type B**: 単調正電位枝。未知数 `(phi0, n_swe_inf)` を解いて BVP を解く。
- **Type C**: 単調負電位枝。未知数 `(phi0, n_swe_inf)` を解いて BVP を解く。

## 今後足しやすいもの

- `tests/` に数値回帰テスト追加
- `docs/` に式と論文対応表を追加
- GitHub Actions で lint / test を追加
- Release を切って版管理

## 推奨する GitHub 初期運用

1. この構成でリポジトリ作成
2. `v0.1.0` を tag / Release
3. 数式説明や図を README / docs に追加
4. 安定したら PyPI 公開を検討
