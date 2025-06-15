# 追加した入力ファイルとツール（監査対応版）

本文に記述がありながら入力ファイルが同梱されていなかった計算について、
入力ファイル・実行スクリプト・不足していたツールを作成したもの。
本ツリーには既に取り込み済み。

    make && make hod_dump.out fwd_only_timing.out
    python3 tools/repro_check.py       # 追加分も含めた存在チェック

すべての入力は本環境（gfortran 13.3, 1 コア）で短い epoch 数の smoke test を
通してあり、トレーナに受理されて走ることを確認済み。**数値が論文値を再現するか
は別問題**であり、監査報告書（`AUDIT_REPORT.md`）の指摘はそのまま残る。

## 新規ファイル

### 5.4 節（KdV のバッチ・学習率走査）
| ファイル | 内容 |
|---|---|
| `bench/kdv/input_batch{20,60,135,270}.dat` | 「バッチ 20/60/135/270 で床が単調低下」 |
| `bench/kdv/input_lr{1d-4,3d-4,1d-3,3d-3,1d-2,3d-2}.dat` | 「学習率を変えても床は 2 倍以内」（既定は 3e-3） |
| `bench/kdv/run_scans.sh` | 両走査を `scan_*/` に実行し、床（最後の 20% epoch の中央値）を印字 |

### 5.4 節（2 次元 Poisson での最適化器の順序）
| ファイル | 内容 |
|---|---|
| `zwork/pinn_poisson2d/input_sgd.dat` | 最急降下（既存の `input_nn.dat` が Adam） |
| `zwork/pinn_poisson2d/input_ngd.dat` | 自然勾配 |
| `zwork/pinn_poisson2d_kalman/input_decoupled.dat` | node-wise decoupled 版フィルタ |

### 5.6 節（Kovasznay）
| ファイル | 内容 |
|---|---|
| `zwork/pinn_kovasznay/input_no_boundary.dat` | 境界データを除いた配点のみの run（`Num_batch` は訓練分割 2800 に合わせてある） |

### 5.7 節（5 場 EHD の最適化器研究）
| ファイル | 内容 |
|---|---|
| `input_cold_m8.dat` | 曲率対 8（既存 `input_cold.dat` は 40） |
| `input_adam_2d-3.dat`, `input_adam_2d-4.dat` | fit からの Adam 2 種（`fitted_weight.dat` を `nn_weight.dat` にコピーして実行） |
| `input_cold_ngd_b40.dat` | dual 自然勾配・バッチ 40・6000 epoch |
| `input_cold_ngd_primal.dat` | 同じ手法の primal 解法（`Ngd_dual` なし）。epoch コスト比較用 |
| `input_kalman_colloc.dat` | per-pattern Kalman（既定の忘却係数） |
| `input_kalman_colloc_lambda1.dat` | 忘却係数を 1 に固定 |
| `input_kalman_colloc_lambda1_decoupled.dat` | 上に decoupled 共分散を併用 |

注: 同梱の `input_kalman_divergence.dat` は現行トレーナが受理しない
（`GD_method KALMAN` は単一出力観測で DATA 項を扱うため、5 出力ネットでは停止する）。
上記フィルタ用入力は配点項のみで構成してこの制約を回避している。

### 6 節
| ファイル | 内容 |
|---|---|
| `zwork/hod_4d_k7_order7/input_tanh_lr_x{0.1,1,10,100}.dat` | tanh の 100 倍学習率走査 |
| `zwork/hod_4d_k7_order7/run_lr_scan.sh` | 上を `lr_x*/` で実行し loss を印字 |
| `tools/hod_dump.f90` | **学習済みネットの全高階微分をダンプする新ツール**（配点 run はこれを出力しないため、7 階微分の主張を検証する手段が存在しなかった） |
| `bench/post/zk7_seventh.py` | そのダンプを厳密ソリトンと比較し、比を 3 通り（残差スロット / 全 120 スロット / 最悪スロット）で印字 |

### 4.1 節・5.1 節
| ファイル | 内容 |
|---|---|
| `tools/fwd_only_timing.f90` | 前進のみ・点あたりの計測（ADOL-C `tensor_eval` との比較用。既存ツールは勾配のみ） |
| `tools/check_exact_coeffs.py` | 階層の係数・平面波置換・Kovasznay・Lax・式(19) を SymPy で厳密検算 |

## 変更した既存ファイル
- `Makefile`: `hod_dump.out` と `fwd_only_timing.out` のルールを追加（既存ターゲットの記法に合わせた）。`make`, `make f2003check`, `make negtests`（33/33）が通ることを確認済み。
- `docs/REPRODUCING.md`: 「図表以外に本文が引用する run」の表を追加。
- `tools/repro_check.py`: 追加分 7 項目を登録。あわせて「ZK7 の 7 階微分」の項目が `tools/alpha_order.py`（列順を出すだけで微分値は出さない）を指していたのを、実際に検証できる `hod_dump.f90` + `zk7_seventh.py` に修正。

## 再生成方法
入力は既存ファイルの編集として生成されているので、配布物を更新したときは
同梱の生成スクリプトを走らせ直せばよい。

    ./make_inputs.sh <DNNF90 の展開先>     # 入力を生成
    python3 add_headers.py <DNNF90 の展開先>  # 各ファイル冒頭の由来コメントを付与

## コード修正: dual Gauss-Newton の Gram 行列形成

`app/optimizer_module.f90`（差分は `optimizer_gram.patch`）。
`K = JJᵀ/N` の形成が `jr(nrow, n_w)` の**行**方向の内積で行われており、
要素ごとに別キャッシュラインを触っていた。行を転置した複製 `jrT(n_w, nrow)`
を 1 回作り、そちらで内積を取る（対称性も利用して下三角のみ計算）。

| ミニバッチ | 行数 | 修正前 | 修正後 | 倍率 |
|---|---|---|---|---|
| 40 | 200 | 0.162 s/epoch | 0.062 s/epoch | 2.6x |
| 60 | 300 | 0.365 | 0.133 | 2.7x |
| 80 | 400 | 1.047 | 0.246 | 4.3x |
| 120 | 600 | 7.909 | 0.563 | 14.0x |

数値は不変であることを確認済み:
- EHD 120 点・520 epoch の完走結果が出荷 `ngd_reference` と全桁一致
  （train 1.1182e-6 / val 9.8474e-7）。所要 293 s（修正前は約 68 分）。
- `bench/opt_ngd` の 50 epoch run が修正前後で**ビット一致**。
- `make negtests` 33/33、`make f2003check` clean。
