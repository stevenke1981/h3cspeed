# stable-diffusion.cpp 部分對齊說明

本次改進只對齊 **MiniMax-H3 模型 metadata 偵測、幾何規格與 fail-fast 預檢**，不把 `stable-diffusion.cpp` 的 GGML/GGUF runtime、通用 backend 或影像模型管理器直接搬入 `h3cspeed`。

參考基準：

- `leejet/stable-diffusion.cpp@de298c225bed97c3f9026b73cd7b71e7879bd41b`
- `src/model/diffusion/minimax_h3.hpp` 的 `Config::detect_from_weights`
- `docs/minimax_h3.md` 的畫布與影格限制
- 本次修改基準：`stevenke1981/h3cspeed@8c9e1e3356fd50959cd5eef7a0a8a9ffde081e96`

## 已對齊範圍

| 能力 | stable-diffusion.cpp | h3cspeed 本次實作 |
|---|---|---|
| 從權重 shape 偵測架構 | `Config::detect_from_weights` | C detector 與 Python inspector |
| DiT / refiner block 數 | 解析 tensor name | 解析並要求 block index 從 0 連續 |
| Attention / FFN 維度 | 從 Q norm、QKV、FC1 推導 | 同等推導並驗證 QKV、SwiGLU 形狀 |
| 模型變體 | time embedder / AdaLN curve table | 兩者皆辨識；目前 CUDA 僅接受 time embedder |
| Header-only 檢查 | tensor storage metadata | 僅讀 safetensors header，不映射大型 payload |
| 畫布規格 | 寬高為 32 倍數 | 提供向上對齊報告，不偷偷修改 CLI 參數 |
| 影格規格 | 最少 5，符合 `17k + 5` | 與 `h3_align_frame_count()` 相同的建議值 |
| Fail-closed | 模型建立時失敗 | 缺漏、重複、稀疏 block、錯誤 rank/shape 提前失敗 |

## 模型預檢工具

檢查單一 transformer component：

```bash
python3 scripts/h3_model_info.py /models/MiniMax-H3/fl2va_transformer
```

掃描完整模型根目錄內的 H3 transformer components：

```bash
python3 scripts/h3_model_info.py /models/MiniMax-H3
```

嚴格驗證目前 `h3cspeed` CUDA kernels 是否能直接執行：

```bash
python3 scripts/h3_model_info.py /models/MiniMax-H3 \
  --strict-h3cspeed
```

輸出 JSON：

```bash
python3 scripts/h3_model_info.py /models/MiniMax-H3 \
  --strict-h3cspeed --json > model-info.json
```

若 checkpoint 使用 namespace：

```bash
python3 scripts/h3_model_info.py model.safetensors \
  --prefix model.diffusion_model
```

Exit code：

- `0`：metadata 完整，且目前 CUDA contract 相容。
- `2`：路徑、safetensors header 或模型 metadata 錯誤。
- `3`：能辨識為一致的 H3 架構，但目前固定 CUDA kernels 不支援。

## 可嵌入的 C detector

```c
#include "h3_model_config.h"

h3cspeed_h3_model_config config;
char error[256];

if (!h3cspeed_h3_model_config_detect(
        tensors, tensor_count, prefix,
        &config, error, sizeof(error))) {
    /* metadata malformed or incomplete */
}

h3cspeed_h3_compatibility incompatibility =
    h3cspeed_h3_model_compatibility(&config);
```

輸入 shape 採 safetensors / PyTorch 順序；Linear weight 是
`[out_features, in_features]`。模組不依賴 CUDA、JSON parser 或上游 overlay，因此可先在 portable CI 驗證，再接到 pinned upstream loader。

## 目前 CUDA contract

目前嚴格模式要求：

- variant：`time-embedder`
- hidden size：`5376`
- DiT blocks：`50`
- token refiner blocks：`2`
- attention heads / head dim：`56 / 128`
- FFN hidden：`14336`
- video / audio latent channels：`24 / 32`
- text dim：`5120`
- timestep input：`256`
- time hidden / output：`5376 / 2688`
- RoPE inverse-frequency length：`16`

AdaLN curve-table checkpoint 會被正確辨識，但會以不相容結束，避免在固定 time-embedder 路徑中晚期崩潰或錯誤執行。

## RTX 3070 Ti 8GB wrapper

`scripts/run-3070ti-8gb.sh` 在找到 `-d/--model-dir` 時，會於配置 CUDA 記憶體前執行 header-only 預檢。

```bash
./scripts/run-3070ti-8gb.sh -d /models/MiniMax-H3 -p "prompt"
```

控制方式：

```bash
H3_MODEL_PREFLIGHT=required ./scripts/run-3070ti-8gb.sh ...
H3_MODEL_PREFLIGHT=off      ./scripts/run-3070ti-8gb.sh ...
```

預設為 `auto`：有 Python 與 inspector 時執行；封裝環境缺少其中一項時保留既有啟動行為。`required` 則會把工具缺失視為錯誤。

## 驗證

```bash
./scripts/test-model-compat.sh
bash -n scripts/run-3070ti-8gb.sh
```

Portable suite 包含 Python safetensors inspector 測試與 C11 detector 測試，使用 `-Wall -Wextra -Wpedantic -Werror`。

本階段沒有修改 CUDA kernel、tensor layout、offload API、量化格式或數值路徑；GPU 數值、峰值 VRAM 與 sanitizer 驗證仍應在 NVIDIA 主機執行。

## 後續對齊順序

1. 將 C detector 接到 pinned upstream `h3_weight_store_open()` 後的 header inventory，並在 `--info` 顯示 variant 與 compatibility。
2. 導入 phase-aware weight lease，明確管理 token refiner、DiT block、final head 與 VAE 的權重生命週期，同時保留現有 CUDA ready/last-use event 安全規則。
3. 以實際 graph shape 推導 runtime headroom，逐步取代部分固定 safety margin。
4. 另開功能旗標實作 AdaLN curve-table 執行路徑，不改寫既有 time-embedder 預設行為。
5. 在 RTX 3070 Ti 8GB、RTX 3090 24GB 與另一張不同世代 NVIDIA GPU 驗證數值、VRAM、吞吐與 Compute Sanitizer。
