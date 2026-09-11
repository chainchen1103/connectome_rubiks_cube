# Connectome Rubik's Cube Lab

以《基於果蠅神經連接體的數位學習系統》為研究規格，逐步建立可重現的神經連接體、LIF 動力學、局部可塑性與環境回饋系統。

本專案的定位是 **connectome-constrained artificial neural system**。突觸數是初始相對權重的 proxy；模型參數與感覺／動作介面是工程假設，不能等同真實果蠅生理功能。

## 實作階段

1. 圖資料匯入、SQLite、腦區及上下游查詢、互動 3D 匯出。
2. 稀疏邊運算的 LIF、興奮／抑制作用、活動與穩定性量測。
3. Hebbian、STDP、reward-modulated eligibility trace、雙刺激學習。
4. 延遲 cue 的 T-maze、記憶與消融對照。
5. 2×2／3×3 魔方、由少量打亂開始的課程、狀態分離評估與拓樸對照。

各階段分別提交並推送到現有 Git 遠端。實驗結果與未達成項目會明確記錄，不把可執行的訓練框架視為已驗證的科學結論。

## 執行環境

Python 3.10+，NumPy。安裝及實驗指令將隨各階段加入。原始 PDF 保留於專案根目錄。
