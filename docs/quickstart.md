# 使用方式

這是一個可實際運行的研究原型：瀏覽器呈現魔方轉動、虛擬果蠅移動及神經活動；Python 在本機執行 LIF 模型和局部突觸權重更新。可以設定每輪時間，保存結果並繼續訓練。目前實驗尚未證明模型學會任意魔方，實際結果見[實驗紀錄](experiment_results.md)。

## 在目前這台 Windows 電腦啟動

在專案目錄開啟 PowerShell：

```powershell
cd D:\connectome_rubiks_cube
.\scripts\run.cmd
```

啟動器可從 PowerShell、命令提示字元或檔案總管雙擊執行。它依序尋找專案 `.venv`、PATH 中的 Python，以及目前使用者的 Codex Python runtime，選用具 NumPy 的 Python 3.10+，並自動開啟瀏覽器。網址是 **http://127.0.0.1:8765/**。第一次產生腦部與回放資料需要幾秒；保留終端視窗讓伺服器運行。

`run.cmd` 只對當次 PowerShell 程序設定 `ExecutionPolicy Bypass`，不修改電腦的永久執行原則。也可直接用以下等價命令呼叫底層腳本：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run.ps1
```

## 第一次在其他電腦安裝

先安裝 Python 3.10+，下載或 clone 這個儲存庫，進入根目錄：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\scripts\run.cmd
```

macOS／Linux 可直接執行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m connectome_lab serve --open
```

一般操作僅需要 NumPy，資料子圖與腦部顯示素材已在儲存庫內。只有重新解析 MaleCNS 原始 Parquet 連線資料才需要 `python -m pip install -e ".[malecns]"`。請從這份 checkout 的根目錄執行；資料檔案目前由儲存庫提供。

## 設定時間、保存與接續

1. 在頁面訓練區選擇資料集；預設為 **MaleCNS v1.0**。選 2×2 或 3×3，以及打亂深度。深度指打亂轉動的次數，並非最短解距離。初次可用深度 1、60 秒觀察流程。
2. 輸入時間上限（秒），按開始／繼續訓練。沒有固定嘗試回合數；每次嘗試最多走 `max(6, 深度 × 4)` 步，失敗就重設同一個打亂狀態再嘗試。
3. 時間到、首次解出或按停止並保存時，系統保存神經權重、活動快照及本輪變化。時間在每個動作之間檢查，最後寫檔和製作回放可能稍晚完成；看到完成狀態後再關閉。
4. 再按開始會接續相同資料集、魔方大小與深度的 checkpoint。前一輪未解出時繼續挑戰同一個打亂狀態；前一輪已解出時換下一個狀態，同時保留權重。
5. 關閉服務後，下次用相同輸出目錄啟動，仍會從磁碟接續。結束 PowerShell 服務請按 `Ctrl+C`，讓正在執行的訓練先停止並保存。

每個 `(資料集, 大小, 深度)` 使用各自的訓練資料夾，例如：

```text
outputs/live/sessions/malecns_cube3_depth1/
  checkpoint.npz                  最新模型、神經狀態、亂數狀態與進度
  progress.json                   可直接閱讀的累積進度
  history/<run_id>.npz             各輪結束時的 checkpoint
  history/<run_id>-changes.npz     各突觸 before、after、delta 及端點 ID
  history/<run_id>.json            時間、停止原因、成功數及權重改變量
```

checkpoint 會定期保存（預設約 10 秒，於嘗試邊界）並在正常停止時再保存，採臨時檔完成後原子替換。接續會保留已學到的突觸權重、感覺／動作介面與亂數狀態；每個新嘗試會重設短期神經動態。保存的膜電位和 trace 可供分析，並非在中斷的魔方步驟逐毫秒繼續。更新的是既有突觸的權重，不是新增神經元或生長軸突。

時間訓練在解出時給 `+1`，走滿嘗試步數仍未解出時给 `−0.1`，中間步驟為 `0`。這與研究 benchmark 的純 solved reward 協定不同，不能直接把兩者成功率當成同一實驗比較。時間長短也不保證解出或泛化能力提升。

## 如何看三個視覺區域

魔方區依模型實際選擇的面轉動，右側神經區對應同一筆動作記錄。即時訓練可能在兩次畫面更新之間走多步，頁面顯示收到的精確最新狀態；訓練速度不被動畫播放速度限制。停止後可查看保存權重所產生的回放。

虛擬果蠅區是另一個二維覓食環境，模型選擇前進、左轉或右轉。它用來展示同一類神經控制介面如何驅動動作，並不是生物力學模擬，也不是果蠅正在用腿操作畫面中的魔方。果蠅回放與魔方訓練屬不同任務。

神經區可旋轉、縮放及選取神經元，查看放電率、膜電位和動作機率。放電率用 Hz；膜電位是模型的無量綱數值。全腦灰色解剖背景不代表全部神經元正在放電，活動顏色來自選定模擬子圖的真實模型輸出。

## 為甚麼現在有果蠅腦的形狀？用了完整大腦嗎？

之前的通用圖布局主要表達節點連接；解剖外形需要來源資料的三維位置或神經纖維骨架。現在 MaleCNS 顯示使用真實座標，並附上 12 個代表性神經元的 SWC 骨架作為靜態背景。因此可看到腦部與腹神經索的空間形狀；骨架只是小量真實樣本，不是整個腦所有纖維的重建畫面。

| 層次 | MaleCNS v1.0 的範圍 |
| --- | --- |
| 完整匯入節點集合 | 166,700 個有神經元分類註記的節點，含成年雄蠅腦與腹神經索 |
| 可顯示的真實定位點 | 140,638 個；包含 139,662 個 soma 位置和 976 個 soma attachment 位置 |
| 缺少來源定位點 | 26,062 個；不為它們捏造解剖座標 |
| 預設即時模擬 | 從真實連接體選取的 512 個神經元誘導子圖 |
| 完整匯入連線 | 25,582,837 個有向節點對，124,177,143 個突觸接觸；套用來源信心門檻與自連結處理 |

完整圖與預設 512 節點模擬是兩個不同層次。子圖讓目前 CPU 與瀏覽器互動可用；灰色全腦位置並沒有被冒充成完整全腦動態。採樣也會遺漏許多迴路，不能代表完整果蠅功能。MaleCNS 是公開重建版本所定義的集合，亦不等於生物體每個未重建片段都完整無缺。來源和處理見 [MaleCNS 資料說明](malecns.md)及[官方下載頁](https://male-cns.janelia.org/download/)。

可切换 **FlyWire v783** 比較成年雌蠅腦：全資料有 139,255 個節點，預設模擬同樣取 512 個。其點位是公開 community anchor，不是每個神經元的胞體或完整骨架；這份資料沒有 MaleCNS 的腹神經索。

若要自行重新下載並匯入完整 MaleCNS 圖：

```powershell
python -m pip install -e ".[malecns]"
python -m connectome_lab fetch-malecns --archive outputs/full_brain/malecns.npz
python -m connectome_lab simulate --graph outputs/full_brain/malecns.npz --steps 30 --output outputs/full_brain/simulation
```

原始連線 Parquet 約 1.05 GB，另有註記資料；完整圖需要比子圖更多記憶體和運算時間。NPZ 保留全圖的神經元、稀疏邊及來源資訊；之後讀取不需要 pyarrow。也可在匯入命令加 `--database outputs/full_brain/malecns.sqlite` 建立 SQLite。完整圖檔和原始下載放在被 Git 忽略的本機資料夾，不隨程式碼上傳。

**目前這台電腦已有完整圖，不必重新下載。** 若要讓完整 166,700 個節點參與限時魔方模擬，可直接指定全圖路徑：

```powershell
.\scripts\run.cmd train-timed --graph outputs/full_brain/malecns.npz --seconds 600 --size 3 --depth 1 --output outputs/sessions/full_malecns
```

再次執行相同命令便會接續全圖 checkpoint。這條命令使用完整圖；`--graph malecns` 則使用預設 512 節點子圖。全圖時間上限同樣在決策之間檢查，單次決策與壓縮存檔可能需要數秒至數十秒。本機 5 秒上限的全圖流程檢查完成 1 次決策，連同最後保存合計約 54.6 秒；它驗證完整模型能執行並保存，沒有完成嘗試或產生獎勵學習。一般觀察和多輪學習請先用網頁的 512 節點模式。

## 命令列與離線回放

所有 CLI 參數也能交給 Windows 啟動腳本：

```powershell
# 改用另一個本機埠
.\scripts\run.cmd serve --port 8766 --open

# 3×3、打亂深度 3，運行最多 10 分鐘
.\scripts\run.cmd train-timed --graph malecns --seconds 600 --size 3 --depth 3 --output outputs/sessions/my_cube

# 下次執行完全相同指令，從此資料夾保存的權重接續
.\scripts\run.cmd train-timed --graph malecns --seconds 600 --size 3 --depth 3 --output outputs/sessions/my_cube

# 將已訓練的模型匯出成可直接開啟的 HTML
.\scripts\run.cmd dashboard --graph malecns --size 3 --cube-checkpoint outputs/sessions/my_cube/checkpoint.npz --output outputs/my_cube.html

# 查看所有命令
.\scripts\run.cmd --help
```

`python -m connectome_lab` 可代替 `.\scripts\run.cmd`。同一 checkpoint 必須匹配圖指紋、魔方大小及深度；要獨立實驗就換 `--output` 資料夾。`train-timed --fresh` 明確開始新模型，並為舊 checkpoint 建立備份。CLI 的 `outputs/sessions/...` 和網頁的 `outputs/live/sessions/...` 各自保存，只有指定同一個資料夾才會共享權重；不要讓兩個程序同時寫入同一訓練資料夾。

離線 HTML 可播放與檢查已記錄動作，不會在瀏覽器背後執行 Python。要實際訓練請使用 `serve` 的本機頁面。Web 服務只監聽 `127.0.0.1`。

## 常見情況

| 情況 | 處理 |
| --- | --- |
| `No module named numpy` 或腳本找不到環境 | 在所選 Python 執行 `python -m pip install -e .`，或按上述方式建立 `.venv` |
| 8765 已被占用 | 先停止原服務，或用 `serve --port 8766 --open` |
| 想查看更早的歷史 | 頁面恢復最近 50 輪；較早的 checkpoint 與各輪 JSON 仍在對應資料夾的 `history/` 中 |
| 權重有變但很少解出 | 這是目前模型的可能結果，請比較凍結模型、隨機動作與未見過的狀態；不以權重改變量代替學習成功率 |
| 想看測試與研究結果 | 參閱[實驗紀錄與重現指令](experiment_results.md)及[模型假設](model.md) |
