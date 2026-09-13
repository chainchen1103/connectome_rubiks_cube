> 本專案使用 AI 進行輔助，AI 主要完成可視化的 UI 部分、部分如註解等文本內容與可行性評估，由底層內容留出接口後交由 AI 完成非主要的簡單任務。

# Connectome Rubik's Cube Lab

以《基於果蠅神經連接體的數位學習系統》為研究規格，逐步建立可重現的神經連接體、LIF 動力學、局部可塑性與環境回饋系統。

本專案的定位是 **connectome-constrained artificial neural system**。突觸數是初始相對權重的 proxy；模型參數與感覺／動作介面是工程假設，不能等同真實果蠅生理功能。

## 實作階段

1. 圖資料匯入、SQLite、腦區及上下游查詢、互動 3D 匯出。
2. 稀疏邊運算的 LIF、興奮／抑制作用、活動與穩定性量測。
3. Hebbian、STDP、reward-modulated eligibility trace、雙刺激學習。
4. 延遲 cue 的 T-maze、記憶與消融對照。
5. 2×2／3×3 魔方、由少量打亂開始的課程、狀態分離評估與拓樸對照。

## 執行環境

Python 3.10+，NumPy。安裝及實驗指令將隨各階段加入。原始 PDF 保留於專案根目錄。

## 目前可直接使用

在 Windows 的專案根目錄執行：

```powershell
.\scripts\run.cmd
```

會啟動 Python 並開啟 **http://127.0.0.1:8765/**。選 MaleCNS、魔方大小、打亂深度及時間上限，按開始／繼續；到時、解出或手動停止時自動保存權重，下一輪接續。第一次使用可選打亂深度 1、60 秒。新電腦先安裝 Python 3.10+，執行 `python -m pip install -e .`，再用 `python -m connectome_lab serve --open` 啟動。

畫面包含魔方轉動、虛擬果蠅動作、神經活動與解剖背景。預設採用 **MaleCNS v1.0 的 512 神經元子圖**進行即時模擬；完整資料含 166,700 個節點，背景呈現其中 140,638 個有真實座標的節點。完整腦形狀與完整全腦動態是不同層次，實際範圍在畫面與文件中分開標示。

- [安裝、操作、限時續訓與保存位置](docs/quickstart.md)
- [MaleCNS 資料來源、完整圖與骨架](docs/malecns.md)
- [已執行的實驗、失敗結果及重現指令](docs/experiment_results.md)
- [模型、學習規則與研究界限](docs/model.md)

目前已完成資料、模擬、學習／對照流程和互動保存系統；實驗尚未證明已學會通用魔方解法。神經權重有變化與偶然解出簡單打亂，均不能代替未見狀態的泛化測試。