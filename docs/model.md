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

## 限時連續魔方與獎懲

`train-timed` 與網頁訓練使用同一個連續回合：魔方不因走滿步數而重設，只有解出、牆鐘時間上限或使用者停止才結束本次執行。未解出的 checkpoint 保存目前魔方、回合累積步數與獎懲、最近狀態、神經動態、可塑性 trace 和亂數狀態；接續從決策邊界恢復。解出後再次開始才建立新魔方回合，同時保留權重、重設短期神經動態。時間在動作之間檢查，單個決策與保存檔案可能使實際結束時間晚於設定值。

令每面有 `m = size²` 張貼紙，`n[f,c]` 為面 `f` 中顏色 `c` 的數量，定義配對分數：

```text
Φ(state) = Σ_f Σ_c n[f,c] × (n[f,c] − 1) / (6 × m × (m − 1))

reward = Φ(after) − Φ(before)
         − 0.002
         − 0.01 × immediately_inverse
         − 0.02 × revisits_recent_state
         + 1.0 × solved
```

`Φ` 衡量同一個面內同色貼紙的配對比例，取值介於 0 和 1，六面全同色時為 1。立即反轉指本步恰好撤銷上一個四分之一圈轉動；重訪指轉動後狀態出現在最近 256 個已拜訪狀態中。這些紀錄隨未完成的回合一起保存。各項相加後傳入既有局部權重更新，時間截止和手動停止本身不追加負獎勵。

配對改善沿路相減：若沒有完成解題，只繞一圈回到同狀態，其改善項總和為 0，還會累積步數與可能的重訪懲罰。這避免單靠反覆撤銷同一轉動取得持續正回饋，但不保證找到解法。配對分數不等於最短解距離，某些必要路徑需要暫時減少同色配對；因此這是待測試的工程 shaping 規則，不能聲稱已證明最優策略、拓樸優勢或生物可信度。模型不查詢 solver、教師動作或逆打亂序列。

差分回饋的理論背景見 [Ng、Harada 與 Russell（1999）](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)。本版本另加動作與歷史狀態懲罰，沒有宣稱套用該論文的最優策略不變保證。

先前限時版本只在解出給 `+1`、有限步數失敗後給 `−0.1` 並重設。本版本改用上述規則，歷史實驗分數不視為新協定的驗證。舊 checkpoint 若缺少目前魔方狀態，遷移時只保留模型權重，從其原始打亂狀態建立連續回合，並記錄遷移通知。

## 評估

測試使用 agent 副本與獨立亂數生成器，凍結權重，也不改動訓練的 RNG 或神經狀態。紀錄成功率、Wilson 區間、互動次數、權重 L1 改變、spike 活動與每 neuron cue response。Wilson 區間僅是 Bernoulli 描述統計；反覆測試少量相同狀態不提供等量獨立泛化樣本。

研究 benchmark／curriculum 的魔方任務仍只給 solved reward、限制步數，與上述即時連續訓練分開。課程按打亂序列長度，而非最短解距離。即使打亂深度不同，也可能得到同一 configuration；最終 held-out 起始狀態必須從所有訓練階段移除，並另外記錄訓練軌跡是否拜訪它。旋轉對稱等价狀態仍可能跨集合；不宣稱對称不變泛化。

此框架可驗證工程行為與探索研究問題。全腦功能重現、拓樸優勢、自然果蠅記憶與任意完整魔方的自主解法均須更多資料與研究才能判定。
