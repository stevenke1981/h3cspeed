# h3cspeed 0.2.0 中文說明

`h3cspeed` 是 `antirez/h3.c` 的 NVIDIA CUDA 移植工程。它保留 MiniMax-H3
原本的模型目錄、safetensors、C API、CLI、影音與多模態流程，只把 Apple
Metal／MPSGraph 後端換成 CUDA。架構參考 `llama.cpp`／GGML 的裝置與
buffer 隔離方式，但不依賴 `llama.cpp`，也沒有把 H3 改成 GGUF／LLM。

## 這一版解決什麼問題

v0.2.0 專門加入 **RTX 3070 Ti 8GB + 系統記憶體 offload**。執行時採三層
記憶體，而不是把所有權重塞進 8GB 顯存：

```text
第一層：RTX 3070 Ti VRAM 熱權重 LRU 快取
             ↓ 上傳／事件同步
第二層：系統 RAM 權重快取
             ↓ 只有可重新讀取的權重才會被淘汰
第三層：NVMe 上的 safetensors／作業系統檔案快取
```

核心行為：

- 10GB 以下顯卡會自動啟用 low-VRAM 模式；
- 顯存預算同時約束權重、activation 與 scratch，不只限制模型權重；
- BF16／FP32 原始權重可在 VRAM、RAM、safetensors 三層之間移動；
- 啟動時動態產生的 INT8 權重與 scale 會先完整保存到系統 RAM，再允許
  從 VRAM 淘汰，不會因為沒有原始檔案而遺失；
- 每個可卸載 tensor 都有 upload-ready 與 last-use CUDA event，核心尚在
  使用時不能被 LRU 釋放；
- RAM 也有 LRU。RAM 不足時，先刪除可以從 safetensors 重讀的副本，保留
  無法重建的 INT8 衍生權重；
- pinned RAM 預設只用 128MiB，另外使用 64MiB staging 分段搬運，避免在
  WSL2 大量鎖頁；
- 每次 submit 後可以釋放大型 scratch，降低階段切換時的顯存峰值。

這不是 `cudaMallocManaged`。Windows／WSL2 對完整 Unified Memory
oversubscription 的支援不像原生 Linux，因此本版改用可控的 RAM 快取、
檔案 fallback 與顯式 CUDA copy。

## 建議硬體

- RTX 3070 Ti 8GB；
- 系統 RAM 最低 64GB，**建議 96GB**；
- 模型放 NVMe SSD；
- Windows 11 原生、Windows 11 + WSL2 Ubuntu，或原生 Ubuntu 22.04／24.04；
- CUDA Toolkit、CMake 3.25+、Ninja、ICU、FFmpeg。

若你的主機是 96GB RAM，WSL2 建議不要使用預設的小容量。建立或修改：

```text
%UserProfile%\.wslconfig
```

內容例如：

```ini
[wsl2]
memory=80GB
swap=16GB
```

PowerShell 重新啟動 WSL：

```powershell
wsl --shutdown
```

## 1. 安裝依賴

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake ninja-build pkg-config python3 git \
  libicu-dev ffmpeg

nvidia-smi
nvcc --version
```

## 2. 針對 RTX 3070 Ti 編譯

RTX 3070 Ti 是 Ampere `sm_86`：

```bash
cd h3cspeed
H3CSPEED_CUDA_ARCHITECTURES=86 ./scripts/build.sh
```

原生 Windows PowerShell（會偵測 Visual Studio Build Tools、CUDA，並驗證／取得
ICU 76.1 runtime）：

```powershell
cd <repo>
.\scripts\build-native.ps1 -BuildDirectory build-native -CudaArchitectures 86
```

原生 Windows 產物位於 `build-native\h3cspeed.exe` 與
`build-native\h3cspeed-cuda-info.exe`；必要 ICU DLL 會自動複製到同一目錄。

產生：

```text
build/h3cspeed
build/h3cspeed-cuda-info
build/libh3cspeed.a
```

先查看顯卡與 offload 預算：

```bash
./build/h3cspeed-cuda-info
```

8GB 顯卡正常應顯示：

```text
offload mode: system-RAM + file fallback (automatic)
low-VRAM profile: yes
CUDA allocation budget: 約 5.7～6.2 GiB
resident GPU weight cache: 約 1.5 GiB
```

選用的 MiniMax-H3 FL2VA 下載器固定在以下不可變版本，模型權重不會放進此
原始碼儲存庫：

```text
MiniMaxAI/MiniMax-H3
939557dc319dd91227e30195a763f272ba7f8765
```

本專案不宣稱此模型快照的授權；下載或使用前請查閱上游發佈條款。

## 3. 最安全的第一個測試

```bash
./scripts/smoke-3070ti-8gb.sh ./MiniMax-H3
```

這個 smoke test 是 diagnostic smoke，不是品質 PASS；實測可辨識動物，
但 4 steps 仍可能出現彩噪。它使用：

- 256×256 正方形輸入／輸出畫布；
- H3 可生成的最小 22-frame decoder chunk；
- 4 steps；
- 50 個 DiT blocks；
- `--reuse 1 --core-reuse 1`，每一步重新計算；
- SSD streaming；
- 不使用 token reduction；
- 不開啟 `--show`。

也可以自己指定提示詞和輸出：

```bash
./scripts/smoke-3070ti-8gb.sh \
  ./MiniMax-H3 \
  "一隻紅狐狸在雪地緩慢行走，固定鏡頭，自然光，環境風聲。" \
  ./outputs/fox-smoke.mp4
```

## 4. 3070 Ti 通用低顯存包裝器

```bash
./scripts/run-3070ti-8gb.sh \
  -d ./MiniMax-H3 \
  -p "一隻紅狐狸在覆雪森林行走，中景穩定跟拍。" \
  -o outputs/fox-balanced.mp4 \
  --width 576 --height 320 \
  --render-width 288 --render-height 160 \
  --frames 22 --steps 20 \
  --layers 50 --reuse 1 --core-reuse 1
```

包裝器在你沒有自行指定時會補上：

```text
--ssd-streaming
--frames 22
--width 256 --height 256  (未指定任何尺寸時)
```

包裝器不會自動加入 token reduction、layers 或任何 reuse 參數，因此保留
upstream 品質預設：`--steps 20 --layers 50 --reuse 1 --core-reuse 1`。
若你明確指定 `--reuse N`，包裝器就不會再補 `--core-reuse`。若兩者都明確
指定且都大於 1，會在啟動二進位檔前清楚拒絕。

正式品質 baseline 使用 256×256、22 frames、20 steps、50 層、
`--reuse 1 --core-reuse 1`，不要用 diagnostic smoke 的 4-step 輸出判定
影片品質。第一輪不要使用 `--show`。

```bash
./scripts/run-3070ti-8gb.sh \
  -d ./MiniMax-H3 \
  -p "一隻紅狐狸在雪地緩慢行走，固定鏡頭，自然光。" \
  -o outputs/fox-quality-baseline.mp4 \
  --width 256 --height 256 \
  --frames 22 --steps 20 \
  --layers 50 --reuse 1 --core-reuse 1
```

若完全沒有指定輸出或 render 尺寸，包裝器會同時補上
`--width 256 --height 256`，避免方形 render 與上游 864×480 輸出比例不符。
正式 baseline 也建議先固定 256×256，再逐步增加輸出尺寸。

若要在 8GB 顯卡上縮短正式品質測試時間，可使用固定的 fast-quality 預設：
輸出要求為 480p（864×480）、5 秒；H3 會對齊成 124 幀（約 5.17 秒）。
內部以 288×160 低解析度渲染，使用 20 steps
與完整 50 層 DiT，每 4 個 denoising steps 才刷新一次 persistent core
（`--core-reuse 4`）。denoiser reuse 固定為 1；預設不啟用 token reduction，
也不會把兩種大於 1 的 reuse 同時傳給 runtime：

```bash
./scripts/fast-quality-3070ti-8gb.sh \
  ./MiniMax-H3 \
  "一隻紅狐狸穿過覆雪松林，中景跟拍，自然冬日光線。" \
  outputs/fox-fast-quality.mp4
```

也可以使用 `H3_FAST_QUALITY_MODEL_DIR`、`H3_FAST_QUALITY_PROMPT`、
`H3_FAST_QUALITY_OUTPUT` 覆寫模型目錄、提示詞與輸出路徑。預設會交給共用
runner 處理低顯存 plumbing，包括 `ram+file` offload 與 SSD streaming。
480p 是輸出尺寸，內部低解析度是為了適配 8GB 顯存的速度／容量取捨。增加
`H3_CUDA_HOST_CACHE_MIB` 可以減少其他 file-backed 權重被淘汰，但不能避免
SSD stream slot 重讀：stream 只保留有限的作用中 layer window，每個
denoising pass 仍會重讀下一個 slot。

## 5. Offload 設定

預設設定已保存在：

```text
profiles/rtx3070ti-8gb.env
```

主要環境變數：

| 變數 | 3070 Ti 預設 | 用途 |
|---|---:|---|
| `H3_CUDA_OFFLOAD` | `ram+file` | `auto`、`ram+file` 或 `off` |
| `H3_CUDA_VRAM_BUDGET_MIB` | `5888` | 所有 CUDA allocation 的總預算 |
| `H3_CUDA_WEIGHT_CACHE_MIB` | `1536` | VRAM 內可卸載熱權重上限 |
| `H3_CUDA_HOST_CACHE_MIB` | 自動 | 預設目前可用系統 RAM 的 60%，最高 64GiB |
| `H3_CUDA_PINNED_HOST_MIB` | `128` | pinned RAM 上限，不含 staging |
| `H3_CUDA_STAGING_MIB` | `64` | RAM／SSD 搬到 GPU 的分段緩衝區 |
| `H3_CUDA_RELEASE_SCRATCH` | `1` | submit 後釋放 GPU scratch |

96GB 主機可明確配置約 56GB RAM 快取：

```bash
export H3_CUDA_HOST_CACHE_MIB=57344
./scripts/run-3070ti-8gb.sh -d ./MiniMax-H3 -p "..." -o output.mp4
```

RAM 快取滿時，原始 safetensors 權重會退回檔案層；動態產生的 INT8 權重
無法從檔案直接重建，因此必須保留在 RAM。若 RAM 配置不足，程式會指出
需要增加 `H3_CUDA_HOST_CACHE_MIB` 或 WSL2 記憶體，不會無聲 OOM。

## 6. 驗證與監控

無 NVIDIA GPU 的靜態／可攜測試：

```bash
python3 scripts/validate_local.py
```

RTX 3070 Ti 實機：

```bash
ctest --test-dir build --output-on-failure
compute-sanitizer ./build/h3cspeed-cuda-info
watch -n 0.5 nvidia-smi
/usr/bin/time -v ./scripts/smoke-3070ti-8gb.sh ./MiniMax-H3
```

關閉程式時會輸出：

- peak device-live；
- resident weight peak；
- host-cache peak；
- VRAM eviction 次數與容量；
- RAM eviction 次數與容量；
- file fallback 讀取量；
- linear、conv、attention dispatch 次數。

## 目前限制

- 已在原生 Windows、CUDA 13.2、RTX 3070 Ti sm_86 使用固定版 FL2VA 模型
  完成編譯、GPU 探測、完整 text-to-video pipeline，以及 FFmpeg
  encode/probe/decode 驗收；固定 seed 的 256x256、22 frames 紅狐雪地松林提示
  已同時通過 4-step 語意 diagnostic 與 20-step 正式品質驗收，20-step 輸出含
  22 幀 H.264、AAC 非靜音音訊，且首／中／尾幀人工檢查符合提示；
- offload 會大量使用 PCIe 與 RAM，速度一定比完整 VRAM 常駐慢；
- attention 仍是 bounded-memory 參考核心，VAE convolution 尚未改用 cuDNN；
- 尚未支援多 GPU與 CPU 計算 fallback；
- 8GB 模式先用 256×256／22 frames 的 diagnostic smoke 確認可辨識輸出；
  正式品質判定則使用 256×256／22 frames／20 steps／50 層，兩種 reuse 都為 1。

下載封裝名稱：`h3cspeed-v0.2.0.zip`。
