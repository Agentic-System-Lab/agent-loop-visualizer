# Agent Loop 可視化

此為[Agent Atlas 技術與實戰交流](https://cyyeh.github.io/agent-atlas-learning/)的專案之一。

- 模型：側欄可以選 [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) 或 [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)，在本機用 Hugging Face `transformers` 推論
- MTP 推測解碼：Gemma 4 E2B 可以在側欄打開（預設關閉），用 [google/gemma-4-E2B-it-assistant](https://huggingface.co/google/gemma-4-E2B-it-assistant) 當草稿模型，一次 forward 產生好幾個 token
- 授權：[Apache License 2.0](LICENSE)

左邊是像 ChatGPT 的聊天介面，右邊一步一步畫出 agent 迴圈：模型讀進脈絡、提出動作、harness 去執行、結果接回脈絡，直到停止。模型跑在你自己的電腦上，所以每個 token 的機率、候選 token、特殊 token、prefill 和生成速度都看得到。兩個模型跑的是同一個 harness，換模型時只有它們的文字格式不同，正好可以對照。

上方的「運作說明」頁把一輪對話從送出到回覆的每一步對應到程式碼，附上檔名、行號和程式片段。

![上方可以切換到「運作說明」頁，側欄可以選模型；左邊是對話，右邊是迴圈圖與第 1 步的 token。滑鼠停在 <|tool_call> 這個特殊 token 上，浮窗說明它的用途，並顯示它的機率、排名與前 5 名候選](docs/screenshot.png)

## 畫面上看得到什麼

| 畫面 | 看得到的東西 | 對應 unit 01 導讀 |
|---|---|---|
| 迴圈圖 | 目前在脈絡、模型、harness、環境、停止哪一格，正在流動的箭頭 | §4 Agent 就是放在迴圈裡的模型 |
| 脈絡 | 每一步模型讀到的完整字串：system prompt、工具 schema、對話，以及上一步接回來的工具結果。滑過深色的特殊 token 會說明它的用途 | §1 脈絡有上限、§3 工具定義也是 token |
| 模型 | 每個生成的 token，顏色越深代表被選中的 token 機率越低。滑過 token 看機率、排名、前 5 名候選，以及 top-k / top-p 剃掉了哪些；特殊 token 另外說明它的用途。打開 MTP 時，草稿模型猜中的 token 有底線，浮窗寫出草稿猜了什麼、有沒有被收下，step 卡片顯示收下幾個、主模型少跑了幾次 forward | §1 LLM 是會抽樣的條件機率分布、解碼設定 |
| Harness | 從模型自己的格式（Gemma 4 或 Qwen3.5）解析出的工具呼叫、schema 驗證結果、重複呼叫警告、你的允許或拒絕 | §3 模型從不直接碰環境 |
| 環境 | 工具實際回傳的結果與耗時 | §2 環境與可觀察的狀態 |
| 停止 | 停止原因和停止 token。Gemma 回答完停在 `<turn\|>`（相當於 end_turn）、要工具時停在 `<\|tool_response>`（相當於 tool_use）；Qwen 兩種都停在 `<\|im_end\|>`。另外還有步數上限、token 上限、使用者中斷 | §5 什麼時候停 |
| 模型分頁 | 參數量拆解、層數與層的種類、attention heads、脈絡上限、MTP 草稿模型、解碼設定的說明、每一步的 prefill 時間與 tok/s（開 MTP 時加上草稿收下幾個、解碼 forward 幾次），以及 tokenizer 全部特殊 token 的 id 與用途（Gemma 24 個、Qwen 33 個） | §1、§7 Agent 是系統 |
| 運作說明頁 | 一輪對話的 15 個階段（包括 MTP 怎麼猜、怎麼驗證），每段附上做這件事的程式碼片段和行號；行號在打開頁面時從原始碼查出來，程式改了會跟著更新 | 全部 |

## 快速開始

需求：

- macOS（Apple Silicon，用 MPS）、有 CUDA 的 Linux，或只用 CPU（很慢）
- 磁碟空間放模型權重：Gemma 4 E2B 約 10 GB，Qwen3.5 0.8B 約 1.8 GB；要用 MTP 再加上 Gemma 的草稿模型約 160 MB
- 記憶體：Gemma 用 MPS 或 CUDA 時建議 16 GB 以上（bfloat16 權重約 10 GB），只用 CPU 時會以 float32 載入，約 20.5 GB，建議 24 GB 以上；Qwen3.5 0.8B 小得多，8 GB 的機器也跑得動。同一時間只會載入一個模型，換模型時舊的會被放掉
- Linux + CUDA：鎖定的 torch 是 CUDA 13 版本，`nvidia-smi` 右上角的 CUDA Version 要 13.0 以上。驅動太舊時 torch 會偵測不到 GPU，改用 CPU 執行；側欄上方會顯示目前用的裝置
- [uv](https://docs.astral.sh/uv/)（會自動裝好 Python 3.12）
- 網路：第一次下載模型，以及 Wikipedia 工具

```bash
uv sync
uv run hf download google/gemma-4-E2B-it   # 先下載權重（約 10 GB）；跳過的話，第一次選到這個模型時會自動下載
uv run hf download Qwen/Qwen3.5-0.8B       # 約 1.8 GB
uv run hf download google/gemma-4-E2B-it-assistant   # MTP 草稿模型，約 160 MB；跳過的話，第一次打開 MTP 時會自動下載
uv run streamlit run streamlit_app.py      # 預設先載入 Gemma；MODEL_ID=Qwen/Qwen3.5-0.8B 可以先載入 Qwen
```

瀏覽器會打開 http://localhost:8501 。第一次載入 Gemma 加暖機大約 10–20 秒，Qwen 幾秒；之後重新整理頁面不會再載入。側欄最上面的「模型」選單可以換模型，換了會開新對話，解碼設定也換成新模型的建議值，MTP 回到關閉。側欄的「MTP 推測解碼」第一次打開時才載入草稿模型（不到 1 秒），之後一直留在記憶體裡。

> 下載時出現 401，可是模型明明是公開的？多半是本機存的 Hugging Face token 過期了。重新登入（`uv run hf auth login`），或加上 `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` 不送 token。

## 可以試試看的實驗

1. **只是聊天機器人**：在側欄關掉「使用工具」，問「台北現在幾點？」。模型看不到任何工具，只能猜或承認不知道。打開工具再問一次，比較兩次的脈絡分頁。
2. **抽樣**：同一個問題，temperature 設 0 和 1.5 各跑一次，看 token 的顏色和浮窗裡「剩下的候選」怎麼變。
3. **停止條件**：步數上限設 1，問一個需要工具的問題，停止原因會變成「達到步數上限」。
4. **人的位置**：打開「執行工具前先問我」，然後按「拒絕」。擋下來的是 harness，模型只會在下一步的脈絡裡看到一則錯誤觀察。對照導讀 §5 郵件刪除事件：寫在 prompt 裡的「先確認」會被壓縮掉，寫在 harness 裡的不會。
5. **推理也是 token**：打開「思考模式」，斜體的思考 token 會出現在工具呼叫之前；脈絡分頁裡 system turn 的開頭會多一個 `<|think|>`。
6. **迴圈會打轉**：問「ReAct 這篇論文是誰提出的？用 Wikipedia 查。」。E2B 常常重複同一個搜尋，harness 會標出「跟之前一模一樣的呼叫」，對照導讀 §4 ReAct 的錯誤分析。
7. **換模型，harness 不變**：同一個問題分別問 Gemma 和 Qwen，打開脈絡分頁比較：回合標記（`<|turn>` 對 `<|im_start|>`）、工具 schema 放的位置、工具呼叫的寫法（`call:名稱{…}` 對 XML）、工具結果接回的位置都不一樣，但 step 卡片裡的 harness、環境、停止判斷完全相同。
8. **MTP：猜得中才會快**：在 Gemma 打開「MTP 推測解碼」，分別按「台北現在幾點？紐約呢？」和「Write a Python function to test for primes.」，再問「用三段話介紹台北。」。比較每一步的 tok/s 和底線：工具呼叫幾乎每個 token 都被草稿猜中，英文程式碼約一半，中文長文只有一成左右。把 temperature 設成 0 再開關 MTP 各跑一次，輸出會一模一樣：草稿模型只決定要跑幾次 forward，不決定寫出什麼。

## 架構

```
streamlit_app.py   版面、session state、模型選單、背景 thread；執行中每 0.5 秒重畫一次
guide_page.py      「運作說明」頁；內容與行號查詢在 guide.py
render.py          把事件列表轉成 HTML：迴圈圖、step 卡片、token 熱度與浮窗、脈絡檢視
agent.py           harness：組 prompt → 生成 → 解析 → 驗證 →（先問你）→ 執行工具 → 接回脈絡 → 判斷要不要停
llm.py             本機推論，自己寫的解碼迴圈；MTP 的草稿與驗證也在這裡
families/          跟著模型走的東西：每個模型家族怎麼寫回合、思考、工具呼叫，哪些 token 會停、各是什麼意思
  __init__.py      Family 的定義、模型選單 MODELS、MTP 草稿模型 DRAFTERS、依 config.json 的 model_type 找家族
  gemma4.py        Gemma 4：輸出的 parser 與 24 個特殊 token 的說明
  qwen3_5.py       Qwen3.5：輸出的 parser 與 33 個特殊 token 的說明
tools.py           工具：schema（給模型看）加上 Python 函式（給 harness 執行）
tests/             pytest 測試，見下面「測試」
```

agent 在背景 thread 執行，每走到一個階段就 append 一個事件；UI 只負責把事件列表畫出來。所以按「停止」或「允許」時 Streamlit 重跑 script，背景的迴圈不會被打斷。

一次執行的事件依序是：`run_start`，然後每一步 `step_start` → `context` → `prefill` → `token` ×N → `model_end` → `parsed`，有工具呼叫時再加上 `tool_call` →（`approval_request` → `approval`）→ `tool_start` → `tool_result`，最後是 `run_end`。

### 為什麼自己寫解碼迴圈

`model.generate()` 不會在每一步把機率分布交出來，所以 `llm.py` 自己跑迴圈：每個 token 做一次 forward pass（帶 KV cache），依序套用 temperature、top-k、top-p，再抽樣。每個 token 因此都能回報：在模型原始分布裡的機率和排名、取樣時的機率、top-k / top-p 之後剩幾個候選，以及前 5 名候選。Greedy 模式下的輸出和 `generate()` 實測一致。

### MTP 推測解碼

解碼時每個 token 都要跑一次完整的 forward，時間主要花在把權重從記憶體讀出來；同一次 forward 多算幾個位置幾乎不會更慢（Gemma 4 E2B 在 Apple M5 上，1 個位置約 56 ms，7 個位置約 70 ms）。所以只要先便宜地猜好接下來幾個 token，主模型就能一次 forward 全部驗證。Google 為 Gemma 4 訓練了這樣的草稿模型（MTP，multi-token prediction），`transformers` 的 `generate(assistant_model=…)` 會用它，但同樣看不到每一步的分布，所以 `llm.py` 把它接進自己的解碼迴圈：

1. **猜**（`_draft`）：草稿模型 `gemma-4-E2B-it-assistant` 約 78M 參數、4 層。它沒有自己的 KV cache，直接讀主模型的 key / value（Gemma 4 E2B 後 20 層本來就共用前面兩層算好的那份），輸入是上一個 token 的 embedding 接上主模型最後一層的 hidden state。一次 greedy 猜 6 個 token（checkpoint 的 `num_assistant_tokens`）。
2. **驗證**（`_drafted`）：主模型對「剛確定的 token + 6 個猜測」做一次 forward，拿回 7 個位置的分布，依序交給 `_sample()` 檢查。收下的猜測不用另外 forward；第一個被退回的猜測，由主模型自己在那個位置選的 token 取代，後面全部作廢，KV cache 用 `crop()` 退回去；6 個都收下時，最後一個位置還能多給一個 token。
3. **輸出不變**：greedy 時，猜測剛好是第 1 名才收下，所以輸出和不開 MTP 一樣（`tests/test_real_model.py` 用真的模型比對）。抽樣時，猜測以「temperature、top-k、top-p 處理後它的機率」被收下，沒收下就從其他候選裡抽，每個 token 最後出現的機率完全不變（speculative sampling，`tests/test_llm.py` 用兩萬次抽樣檢查）。

收下的比例很看內容。在網頁裡用 Gemma 的預設解碼設定實測（Apple M5）：

| 這一步在寫什麼 | 不開 MTP | 開 MTP | 草稿收下 |
|---|---|---|---|
| 工具呼叫（「台北現在幾點？紐約呢？」第 1 步） | 12.6 tok/s | 40.9 tok/s | 34 / 35 |
| 根據工具結果回答（同一題第 2 步） | 13.0 tok/s | 22.6 tok/s | 42 / 106 |
| 英文程式碼，約 750 tokens | 8.8 tok/s | 11.6 tok/s（單獨跑 `llm.py`：8.1 → 29.0） | 51% |
| 中文長文（「用三段話介紹台北。」） | 10.3 tok/s | 10.3 tok/s | 7% |

### 支援更多模型

harness（`agent.py`）、解碼迴圈（`llm.py`）和畫面（`render.py`）都只寫一次，裡面沒有任何「如果是 Gemma」的分支。每個模型不一樣的地方集中在 `families/` 的一個 `Family` 物件裡：

- 讀輸出：`parse`（把輸出切成思考、工具呼叫、回覆）、`thought_prefix`、`channel_switch`（哪些 token 切到思考、工具呼叫、回覆）、`stop_tokens`
- 寫回對話：`reasoning_key`（思考存在哪個欄位，chat template 才讀得到）、`text_with_calls`（工具呼叫前的文字留不留）
- 畫面：`user_turn`、每個特殊 token 的說明、側欄與模型分頁的說明文字、沒有 generation_config.json 時的建議解碼設定

要加模型：

1. 同一個家族的另一個 checkpoint（例如更大的 Gemma 4 或 Qwen3.5）：在 `families/__init__.py` 的 `MODELS` 加一行。Gemma 4 的其他 checkpoint 也有對應的 MTP 草稿模型（`google/gemma-4-*-it-assistant`），在 `DRAFTERS` 加一行，側欄的 MTP 開關就能用。
2. 新的家族：照 `families/qwen3_5.py` 寫一個模組（輸出的 parser 加上特殊 token 的說明），把它的 `Family` 加進 `FAMILIES`。載入模型時會用 checkpoint `config.json` 裡的 `model_type` 自動找到它。`tests/test_chat_template.py` 會拿真的 tokenizer 和 chat template 檢查新家族：每個特殊 token 都有說明、模板寫進歷史的工具呼叫讀得回來、step 卡片標成「新加入」的剛好是工具結果。

### 兩個模型的格式

Gemma 4 的字串用 `<|"|>` 這個特殊 token 包起來，key 也不加引號，所以不是 JSON，`families/gemma4.py` 有自己的小 parser：

```
<|tool_call>call:get_current_time{timezone:<|"|>Asia/Taipei<|"|>}<tool_call|><|tool_response>
```

Gemma 寫完工具呼叫會停在 `<|tool_response>`，等 harness 接上結果；回答完則停在 `<turn|>`。這兩個停止 token 正好對應到 Anthropic Messages API 的 `stop_reason: tool_use` 和 `end_turn`。

Qwen3.5 用 XML，每個值都是純文字，型別要照工具的 schema 轉（和 vLLM 的 `qwen3_coder` parser 一樣）：

```
<tool_call>
<function=get_current_time>
<parameter=timezone>
Asia/Taipei
</parameter>
</function>
</tool_call><|im_end|>
```

Qwen 不管回答完或要用工具都停在 `<|im_end|>`，所以 harness 是看「解析出來有沒有工具呼叫」決定要不要結束，而不是看停止 token。工具結果會放進一則新的 `user` 訊息（`<tool_response>…</tool_response>`），而不是像 Gemma 接在同一個 model turn 裡。

## 測試

```bash
uv run pytest                                          # 約 15 秒，不載入模型權重、不連網路
RUN_MODEL_TESTS=1 uv run pytest tests/test_real_model.py   # 載入真的 Gemma（含 MTP 草稿模型）和 Qwen，約 40 秒；加 -k Qwen 只跑一個
```

大部分測試用 `tests/fakes.py` 的假模型：它照劇本一個 token 一個 token 吐出預先寫好的輸出，介面和 `LocalLLM` 一樣，所以 harness 和網頁可以從頭跑到尾。

| 檔案 | 測什麼 |
|---|---|
| `test_gemma4.py` | Gemma 4 的 parser：思考、工具呼叫、回覆的切分；`<\|"\|>` 字串、巢狀參數、各種格式錯誤 |
| `test_qwen3_5.py` | Qwen3.5 的 parser：XML 工具呼叫、照 schema 轉型別、思考、各種格式錯誤；用 Qwen 格式跑完整個 harness |
| `test_families.py` | 每個家族都說明了它用到的 token、`model_type` 找得到家族 |
| `test_tools.py` | schema 驗證與預設值、計算機的 AST 白名單（拒絕 `__import__` 之類）、時區、Wikipedia（HTTP 用假的） |
| `test_llm.py` | 自己寫的解碼迴圈：KV cache、停止 token、token 上限、中斷、channel 切換、被拆開的中文字元；temperature / top-k / top-p 的候選與機率；MTP 用照劇本猜的假草稿模型：猜測的收下與退回、全部收下時多一個 token、KV cache 退回多少、草稿模型看不到作廢的猜測，以及抽樣時檢查猜測不改變分布 |
| `test_agent.py` | harness：事件順序、工具結果接回脈絡、驗證錯誤、重複呼叫、結果截斷、每一種停止原因、允許 / 拒絕 / 等待時按停止 |
| `test_render.py` | 一次執行的每一個事件前綴都能畫出來（執行中每 0.5 秒重畫）、HTML 標籤成對、迴圈圖是合法 SVG、模型輸出一律跳脫、特殊 token 的說明浮窗與列表；MTP 的底線、浮窗、統計和草稿模型卡片 |
| `test_app.py` | 用 Streamlit `AppTest` 操作整個網頁：送出問題、範例按鈕、側欄設定傳到模型、多輪對話、脈絡與模型分頁、允許 / 拒絕 / 停止、新對話、還原設定、兩個分頁不能同時跑、換模型、運作說明頁、到說明頁再回來設定還在；MTP 開關預設關閉、打開才載入草稿模型、換到沒有草稿模型的 Qwen 會關掉並停用 |
| `test_chat_template.py` | 用每個模型真的 tokenizer 和 chat template（本機有快取才跑）：特殊 token 都是單一 token、每個特殊 token 都有說明、模板寫進歷史的工具呼叫能被 parser 讀回；用真的模板跑 harness，確認 step 卡片標成「新加入」的剛好是工具結果（兩個參數、思考模式、工具呼叫前有文字） |
| `test_guide.py` | 運作說明頁的每個程式片段和行號都找得到、事件表指向真的送出事件的那一行、HTML 標籤成對 |
| `test_real_model.py` | 預設跳過。每個模型：greedy 解碼和 `model.generate()` 一致，真的模型能呼叫計算機並回答；Gemma 開 MTP 時 greedy 輸出和不開一模一樣，而且草稿真的有被收下 |

## 已知限制

- 速度：Apple M5（32 GB）上 Gemma 大約 7–13 tok/s，Qwen3.5 0.8B 大約 20–25 tok/s。思考模式一步常有兩、三百個思考 token，要等 30 秒左右。打開 MTP 後，工具呼叫這類好猜的步驟大約快 3 倍，中文長文幾乎沒有差別（見上面「MTP 推測解碼」的表）。
- MTP 在網頁裡跑長回覆時，加速比單獨跑 `llm.py` 少很多（英文程式碼：單獨跑 8 → 29 tok/s，網頁裡 9 → 12 tok/s），原因還沒查清楚。可能是網頁每 0.5 秒重畫一次、瀏覽器和模型搶同一台機器，而 MTP 每一輪有好幾個很小的草稿 forward，比一般解碼更容易被這些干擾拖慢。
- MTP 只有 Gemma 4 E2B 能開：`transformers` 目前只支援 Gemma 4 的草稿模型讀主模型 KV cache 的這種做法，Qwen3.5 沒有對應的草稿模型。
- 開 MTP 時，主模型一次算 7 個位置，數值和一次算 1 個會有極小的差異；greedy 遇到兩個 token 機率幾乎一樣時，偶爾會選到不同的那個。
- Qwen3.5 的 24 層裡有 18 層是 Gated DeltaNet。transformers 有兩個加速 kernel：`flash-linear-attention`（Triton，只有 Linux wheel）和 `causal-conv1d`（CUDA），Mac 上都裝不了，所以改用 PyTorch 參考實作，結果一樣，只是比較慢；`llm.py` 在 MPS 和 CPU 上把 transformers 這個「改用參考實作」的提示關掉了。在有 CUDA 的 Linux 上，提示會照常出現，可以裝這兩個套件加速。
- 在 MPS 上，transformers 用多條 thread 載入 Qwen3.5 的權重會卡住，所以 `llm.py` 在 MPS 上改成單一 thread 載入（`HF_DEACTIVATE_ASYNC_LOAD=1`），兩個模型的載入時間都沒有變慢。
- Qwen3.5 0.8B 官方預設不開思考模式；開了容易一直想下去停不下來，可以把「每步 token 上限」調低。它沒附 generation_config.json，建議的解碼設定寫在 `families/qwen3_5.py`（官方建議的 presence penalty 這裡沒有實作）。
- 記憶體裡同時只放一個模型。兩個分頁選了不同的模型時，輪流在兩邊操作會一直重新載入，最好所有分頁用同一個模型。
- 只有一份模型，一次只能跑一個 agent，其他分頁要等。執行中重新整理或關掉分頁，agent 不會跟著停；其他分頁（包括重新整理後的同一頁）會顯示它還在跑，並提供「停止那個 agent」按鈕。
- 開著「執行工具前先問我」時，等確認超過 5 分鐘沒有回應，harness 會當作拒絕並停止迴圈，停止原因顯示「等待確認逾時」。
- token 浮窗要等那一步生成完才會出現，生成中只有熱度顏色（特殊 token 的用途說明除外），避免每次重畫都送出大量 HTML。浮窗是純 CSS，太靠近時間軸底部的 token，浮窗可能被切到。
- 特殊 token 的說明浮窗用 CSS anchor positioning 貼在 token 旁邊，不會被捲動框切到；不支援的瀏覽器會改成直接開在 token 下方，靠近框的底部時可能被切到。
- 兩個模型都很小：可能用簡體中文回答、重複同一個搜尋、只回答一半的問題，或在思考裡算錯。這些剛好是可以觀察的材料。
- Wikipedia 工具需要網路。
- `MODEL_ID` 環境變數決定一開始選哪個模型，也可以指定選單裡沒有的 checkpoint，只要它屬於已經支援的家族（Gemma 4 或 Qwen3.5）；其他家族要先照「支援更多模型」加一個 family。

## 授權

本專案的程式碼以 [Apache License 2.0](LICENSE) 授權。Gemma 4 模型權重和它的 MTP 草稿模型由 Google 以 Apache 2.0 釋出（見[模型卡](https://huggingface.co/google/gemma-4-E2B-it)、[草稿模型卡](https://huggingface.co/google/gemma-4-E2B-it-assistant)），Qwen3.5 模型權重由 Alibaba 的 Qwen 團隊以 Apache 2.0 釋出（見[模型卡](https://huggingface.co/Qwen/Qwen3.5-0.8B)），都不包含在本專案裡。

AI 使用說明：程式碼由 Claude Code 協助撰寫，並在本機用 gemma-4-E2B-it（含 MTP 草稿模型 gemma-4-E2B-it-assistant）和 Qwen3.5-0.8B 實際執行、用瀏覽器檢查過畫面。
