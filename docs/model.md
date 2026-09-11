# 模型、介面與研究界限

## LIF 與可塑性

`dynamics.py` 使用 dimensionless voltage/current、毫秒時間，對固定輸入做指數漏電更新；突觸電流指數衰減，放電有一個時間格的傳導延遲。放電後完整封鎖 `ceil(refractory_ms/dt_ms)` 個時間格。以邊陣列與 `bincount` 累積，不建立 N×N 矩陣。預設步長 1 ms、膜時間常數 10 ms、突觸時間常數 12 ms。依 [Gerstner 等人的 LIF 教材](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html) 實作簡化模型，參數並非由本資料測得。

GABA 預設為負，ACh 為正；其他／unknown 預設正值，只是模型假設。果蠅 glutamate 的作用需受體資訊，不能僅憑傳導物質名稱斷定。可用逐 neuron ID 的 `sign_overrides` 改寫。真實幼蟲範例沒有 NT 資料，因此預設全正，不能用它證明真實生理 E/I 平衡。

初始 magnitude = gain × contact_count / 該目標的總 incoming contact count。訓練保持邊集合與來源符號不變，限制 magnitude 範圍。對照重接保留的是原始 outgoing contact strength；初始化 normalization 後的有效強度不保證相同。

`plasticity.py` 提供 Hebbian、pair STDP、reward STDP 與完全凍結。STDP 在讀取舊 trace 後才加入本格 spike；孤立的同時 spike 不貢獻更新。時間以 emitted spikes 計，未另平移突觸到達延遲。Reward STDP 先累積 eligibility，再用標量 reward 乘上更新。這些規則可獨立研究；依據 [Legenstein 等人的 reward STDP 研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC2543108/)，本專案並不宣稱實作重現其全部實驗。

## 閉迴路學習

實際 task agent 使用另一個明確命名的 **motor-edge three-factor rule**，不是宣稱裸 pair STDP 已解決任務。三個因子為：

1. 神經元放電率的指數活動 trace。
2. 隨機動作訊號減去該動作的機率（centered postsynaptic signal）。
3. 環境提供的標量 reward。

輸入使用固定 encoder：小觀察向量分配至感覺神經元；魔方用固定、有種子的稀疏投影，無訓練好的表示。輸出取 motor 神經群的樹突電流（沿既有邊加權 presynaptic trace）後用 softmax 抽樣。可塑性只作用於既有 motor incoming edges；没有新增稠密 policy、反向傳播、教師動作、搜尋 solver 或打亂逆序。

此更新是固定 presynaptic activity 下的局部 policy semi-gradient，未對 recurrent dynamics 微分。活動 trace 是明確加入的短期記憶機制。T-maze 成功不能獨自證明記憶由 recurrent topology 湧現；`memory=False` 會清除每步的神經狀態與活動 trace，作為聯合記憶消融。可塑性 eligibility 仍跨步保留以接收延遲 reward，且不參與動作選擇。

輸出／輸入分群優先使用原始 role；缺少 annotation 時用固定、可記錄的神經元 ID 順序指定工程介面。在三種拓樸對照之間保留同一 neuron 順序、encoder seed、輸出分群、超參數、環境種子與預算。重接也會改變 role 間連通，不能把差異全部歸因某一類 biological motif。

## 評估

測試使用 agent 副本與獨立亂數生成器，凍結權重，也不改動訓練的 RNG 或神經狀態。紀錄成功率、Wilson 區間、互動次數、權重 L1 改變、spike 活動與每 neuron cue response。Wilson 區間僅是 Bernoulli 描述統計；反覆測試少量相同狀態不提供等量獨立泛化樣本。

魔方只給 solved reward，限制步數。課程按打亂序列長度，而非最短解距離。即使打亂深度不同，也可能得到同一 configuration；最終 held-out 起始狀態必須從所有訓練階段移除，並另外記錄訓練軌跡是否拜訪它。旋轉對稱等价狀態仍可能跨集合；不宣稱對称不變泛化。

此框架可驗證工程行為與探索研究問題。全腦功能重現、拓樸優勢、自然果蠅記憶與任意完整魔方的自主解法均須更多資料與研究才能判定。
