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

### 不含模型的跨平台執行檔

runtime packager 支援兩種執行檔壓縮包：

```text
h3cspeed-v0.2.0-windows-x86_64-cuda13.2-sm86.zip
h3cspeed-v0.2.0-linux-x86_64-cuda13.2-sm86.tar.gz
```

Windows 版本由本機 RTX 3070 Ti 驗收機建置與測試；固定版本的 `binary builds`
GitHub Actions workflow 則在不可變 CUDA container 產生 Linux 版本。壓縮包
包含 CLI、CUDA-info、啟動設定、SHA-256、授權文件，以及可再散布的
CUDA／ICU runtime；不包含模型、`.h3c` conditioning sidecar 或輸出影片。
主機仍需相容的 NVIDIA 驅動與 FFmpeg／FFprobe。Windows 另外需要 Microsoft
Visual C++ 2015-2022 x64 Redistributable；Linux 採 Ubuntu 22.04 glibc 基線。

解壓後可先測試：

```powershell
.\bin\h3cspeed.exe --help
.\bin\h3cspeed-cuda-info.exe
```

```bash
./bin/h3cspeed --help
./bin/h3cspeed-cuda-info
```

Windows 壓縮包會在 RTX 3070 Ti 驗收機實跑；GitHub 的 Linux hosted runner
沒有相容 GPU，因此它只證明 CUDA 編譯／連結、ELF 啟動、依賴閉包與內嵌
`sm_86` 機器碼，不代表 Linux GPU 推理已 PASS。

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

## 準備本機 ComfyUI 量化 T2V 模型包

若本機 ComfyUI 已有四個 H3 檔案，可先用 header-only preparer 驗證並建立
可攜式模型根目錄。它會檢查 FL2VA ConvRot INT8 marker（group size 256）、
Qwen3-VL NVFP4 scale 與 `pre_quant_scale` alias、F16 video VAE、F32 audio
VAE，以及每個 safetensors offset／size；不會讀取大型 payload。

```powershell
python scripts/prepare_h3_quantized_model.py --validate-only
python scripts/prepare_h3_quantized_model.py
```

預設來源是 `E:\minimax-h3\ComfyUI\models`，小型 config／tokenizer 取自
`E:\models\MiniMax-H3`，輸出到
`E:\minimax-h3\ComfyUI\models\h3_t2v_quantized`。需要時可覆寫路徑：

```powershell
python scripts/prepare_h3_quantized_model.py `
  --models-root E:\minimax-h3\ComfyUI\models `
  --base-root E:\models\MiniMax-H3 `
  --output-root E:\minimax-h3\ComfyUI\models\h3_t2v_quantized
```

四個大型 safetensors 只建立 hardlink，不會複製 payload。只有 allow-list
內的小型 config／tokenizer 會複製到 `base/`，並在 `manifest.json` 記錄來源、
大小、header hash、dtype 與 schema coverage。驗證失敗或輸出根目錄已存在時
會 fail-closed；原生模型根目錄是 `h3_t2v_quantized/base`，保留
`FL2VA/transformer`、`FL2VA/text_encoder`、`FL2VA/video_vae/source` 與
`FL2VA/audio_vae`。請檢查 manifest 後再刪除或改用新的輸出目錄。

四個 hardlink payload 合計 42,470,585,471 bytes（39.55 GiB）。原生 CUDA
路徑會直接使用 FL2VA DiT 的 INT8 ConvRot 權重，對 activation 套用必要的
online Hadamard rotation；Qwen 則依 blocked scale 與 activation-side
`pre_quant_scale` 解碼 NVFP4/AWQ 權重，F16 video VAE 會在載入時轉換。
Ampere（`sm_86`）上的 Qwen NVFP4 是 correctness／capacity 路徑，會
materialize BF16 權重，並非原生 NVFP4 Tensor Core 執行。這個四檔模型根
只包含 T2V／FL2VA，不含另一個 Ref2VA transformer。

### ComfyUI CUDA conditioning bridge（量化模型建議路徑）

目前原生 text encoder 會以 BF16 執行 Qwen。直接在 native runtime 解碼
Qwen NVFP4/AWQ 雖然可以載入與執行，但仍標記為 experimental，不能當作此
量化包的語意品質 gate。正式可用的路徑是讓本機 ComfyUI CUDA runtime 先
編碼同一個 prompt，再把 prompt 綁定的 BF16 conditioning sidecar 交給
native INT8 DiT／VAE。這個 bridge 是明確、GPU-only，不會靜默改用 CPU。
Sidecar v2 另支援 FL2VA first／last keyframe I2V，但仍拒絕 Ref2VA 參考圖。
Helper 會先依 native 規則產生 canonical PNG（first=stretch、last=cover），
把完全相同的影像送入 Comfy Qwen 與 native keyframe 路徑，並在 Qwen 前後
驗證原始檔與 canonical 影像 SHA-256；sidecar 同時綁定 render geometry、
Picture／vision-pad token sequence 與 canonical digest。

Helper 會以 Comfy 的 tokenizer／Qwen CUDA 路徑產生 atomic sidecar，內容含
prompt UTF-8、token ID／tag、BF16 conditioning，以及整個 Qwen 模型的
SHA-256。直接呼叫時要使用提供的 ComfyUI virtual environment Python；
wrapper 會自動尋找這個 Python：

```powershell
<ComfyUI-root>\.venv\Scripts\python.exe scripts/encode_h3_quantized_prompt.py `
  --comfyui <ComfyUI-root> `
  --text-encoder <Qwen-NVFP4-or-AWQ-safetensors> `
  --output <cache-sidecar.h3c> `
  --prompt "A red fox walks through fresh snow in a pine forest." `
  --device cuda:0
```

Windows 一鍵 wrapper 會從提供的 ComfyUI root 或其上一層尋找 `.venv`／`venv`
Python，也可用 `-ComfyPython` 與 `-BinaryPath` 明確指定。它先執行 helper 建立
sidecar，再驗證 helper stdout 的 `model_sha256=` 與 `Get-FileHash` 一致，
只對子程序設定 `H3CSPEED_TEXT_EMBEDDING`、
`H3CSPEED_TEXT_ENCODER_SHA256`，最後還原呼叫端環境：

```powershell
.\scripts\run-h3-quantized.ps1 `
  -ModelRoot <prepared-root> `
  -ComfyUIRoot <ComfyUI-root> `
  -TextEncoder <Qwen-NVFP4-or-AWQ-safetensors> `
  -Prompt "A red fox walks through fresh snow in a pine forest." `
  -Output <output.mp4> `
  -Steps 20 -Width 256 -Height 256 -Frames 22
```

預設為 `20` steps、`256x256`、`22` frames；缺少路徑、非 CUDA device、
helper／sidecar／SHA 驗證失敗、尺寸錯誤或 reuse 設定衝突，都會在 native
程式啟動前 fail-closed。Sidecar 路徑已完成實機 4-step 與 20-step、exit 0、
完整 decode 且可辨識狐狸的 smoke；20-step 的 H.264/AAC artifact 與量測記錄
收錄於 `VALIDATION_RESULTS.md`，直接 native Qwen 仍維持 experimental。

FL2VA I2V 可傳入第一張、最後一張或兩者。Wrapper 會建立 canonical PNG 與
v2 sidecar，再把 canonical PNG 傳給 native keyframe conditioning；digest 或
geometry 不一致會 fail-closed：

```powershell
.\scripts\run-h3-quantized.ps1 `
  -ModelRoot <prepared-root> -ComfyUIRoot <ComfyUI-root> `
  -TextEncoder <Qwen-NVFP4-or-AWQ-safetensors> `
  -Prompt "A red fox walks through fresh snow in a pine forest." `
  -FirstFrame <first.png> -LastFrame <last.png> `
  -Output <i2v-output.mp4> -Steps 20 -Width 864 -Height 480 `
  -Frames 124
```

準備完成後，可先執行短版原生 diagnostic（experimental Qwen 路徑，不使用
sidecar）：

```powershell
.\build\h3cspeed.exe `
  -d E:\minimax-h3\ComfyUI\models\h3_t2v_quantized\base `
  -p "A red fox walks through fresh snow in a pine forest." `
  --width 256 --height 256 --frames 22 --steps 4 --layers 50 `
  --reuse 1 --core-reuse 1 --ssd-streaming `
  -o outputs\quantized-smoke.mp4
```

正式品質基準請保持相同 layers／reuse，改用 20 steps；4-step 只驗證
管線與提示語意。

### Opt-in SageAttention

設定 `H3_CUDA_ATTENTION=sage` 可啟用原生 Sage-style BF16 attention：Q／K
逐 token INT8 量化、DP4A 計算 QK，softmax 保持 FP32 online 計算，V／輸出
維持 BF16。`sm_80` 以上且符合條件的 DiT／text attention 會使用 Sage；
F32 Video VAE 等不符合 dtype／shape 的路徑仍使用既有 CUDA attention，
不會落到 CPU。預設仍是 `native`：

```powershell
$env:H3_CUDA_ATTENTION = 'sage'
.\scripts\run-h3-quantized.ps1 <arguments>
```

實測主機為 Windows 10 22H2 build 19045、RTX 3070 Ti 8 GiB（`sm_86`）、
NVIDIA driver 596.36、CUDA 13.2／nvcc 13.2.78、Visual Studio Build Tools
18.6.0（MSVC 14.51）。Release build 命令為
`scripts/build-native.ps1 -BuildDirectory build-quant -CudaArchitectures 86
-BuildType Release`；以 `$env:H3_CUDA_ATTENTION='native';
.\build-quant\bench_cuda_attention.exe` 跑 B=1／H=56／N=800／D=128、warmup
2 次、正式 10 次，量得 native 104.806 ms、Sage 97.644 ms（1.073x），
MAE 5.75e-6、最大絕對誤差 0.000488281、cosine 0.999999。backend 回報
device allocation peak 55.03 MiB、host cache 0 MiB；benchmark 的五個 BF16
host arrays 合計 54.69 MiB。這是單一 attention shape 實測，不代表整支影片
必然有相同比例加速；目前只實測 `sm_86`，較新 NVIDIA 架構仍待驗證。

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

- BF16-Qwen baseline 已在原生 Windows、CUDA 13.2、RTX 3070 Ti sm_86
  完成固定 seed 的 256x256、22 frames、20-step 紅狐雪地松林提示，並通過
  H.264/AAC full decode、幀數、非靜音音訊與畫面檢查；量化四檔模型則以
  Comfy CUDA conditioning bridge + native INT8 DiT／VAE 作為可用路徑；
- offload 會大量使用 PCIe 與 RAM，速度一定比完整 VRAM 常駐慢；
- native attention 仍是預設；opt-in Sage 目前只涵蓋符合條件的 BF16 shape，
  VAE convolution 尚未改用 cuDNN；
- 尚未支援多 GPU與 CPU 計算 fallback；
- 8GB 模式先用 256×256／22 frames 的 diagnostic smoke 確認可辨識輸出；
  量化 sidecar 的 4-step 與 20-step 實機驗收均已通過；BF16 與量化正式
  判定皆使用 256×256／22 frames／20 steps／50 層。

下載封裝名稱：`h3cspeed-v0.2.0.zip`。
