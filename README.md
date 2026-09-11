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