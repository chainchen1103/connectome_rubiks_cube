# 資料來源與可重現處理

本系統提供真正公開的果蠅**幼蟲**連接體子圖，以及獨立標示的合成測試網路。幼蟲資料不能視為成年果蠅 FlyWire 全腦，也不能用局部子圖的學習結果推論生物果蠅具有解魔術方塊能力。

## 真實資料

來源為 [Pedigo et al. (2023), eLife 12:e83739 的 Figure 1 source data 1](https://elifesciences.org/articles/83739/figures)，底層重建來自 [Winding et al. (2023), Science, doi:10.1126/science.add9330](https://doi.org/10.1126/science.add9330)。[論文在 PMC 的版權說明](https://pmc.ncbi.nlm.nih.gov/articles/PMC10115445/)標示 Creative Commons Attribution；衍生 CSV 保留來源與作者歸屬，資料依 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)提供，與本專案程式碼授權分開。

- 原始 ZIP：`https://cdn.elifesciences.org/articles/83739/elife-83739-fig1-data1-v2.zip`
- ZIP SHA-256：`c92177b4720d86b210ea44c10b9354fdf079c8cca88f0c0bce3b57815293c9cb`
- ZIP 內檔案：`elife/meta_data.csv`、`elife/G_edgelist.txt`。
- 實際匯入檔包含 3,591 筆神經元註記、119,010 列有向邊；包含輸入／附屬神經元，這不是論文「腦內神經元總數」。
- 控制圖不容許自連結，因此預設明確移除 577 列自連結；其餘方向與接觸數保留。
- 提供的 `data/larval_subset/` 為 192 個神經元、3,361 條邊的誘導子圖，並非完整生物迴路或具代表性的隨機樣本。

選樣演算法先依 skeleton ID 數值排序，再由第一個具連結的 sensory/input 神經元 `37365` 出發，在忽略方向的鄰接圖執行 BFS，鄰居依 ID 排序，取前 192 個節點；最後保留原始有向邊與原始接觸數。方向只在選點過程暫時忽略，資料與模擬仍是有向的。原始 ZIP 不納入 Git；衍生 CSV、SHA-256 與選樣參數納入 Git。

神經元 ID 一律以字串保留。`neuron_type` 取原始 `merge_class`；sensory/input 與 motor 註記決定對應 role，其他節點為 inter。`region` 是原資料的 hemisphere，並非 neuropil。此樣本含 13 個 sensory 與 179 個 inter，**沒有 motor 註記節點**；若實驗配置人工輸出讀出，須揭露為工程映射，不可新增虛構的解剖標籤。

ZIP 未提供神經傳遞物質或真實三維座標：所有 NT 保留 `unknown`，XYZ 為零值占位，不能拿來當解剖座標。可視化需要另做示意布局。模擬若將 unknown 指定為興奮性，屬模型假設，不能當作原始觀測。突觸接觸數也不是直接測量的突觸效能或完整動力學參數。

## 重現真實子圖

```python
from connectome_lab.data import fetch_larval, load_larval_archive, export_csv, sha256_file

archive = fetch_larval("data/downloads")  # 約 1.3 MB，不需帳號；先驗證固定 SHA-256
graph = load_larval_archive(archive, max_neurons=192)
neurons, synapses = export_csv(graph, "data/larval_subset")
graph.metadata["derived_files_sha256"] = {
    neurons.name: sha256_file(neurons), synapses.name: sha256_file(synapses)
}
export_csv(graph, "data/larval_subset")
```

`max_neurons=None` 載入整份來源。`remove_self_loops=False` 可保留自連結，但建構 loopless 控制圖前必須明確處理。來源雜湊不同會直接拒絕匯入，不會自動接受上游異動。CSV 匯入、SQLite 往返、來源標示與控制圖約束有離線單元測試。

## FlyWire／一般 CSV 匯入

標準 neuron CSV 欄位：`id,neuron_type,neurotransmitter,role,region,x,y,z`。
標準 edge CSV 欄位：`source,target,synapse_count`，方向為 presynaptic → postsynaptic。

支援 `root_id/pt_root_id`、`pre_root_id/pre_pt_root_id`、`post_root_id/post_pt_root_id`、`syn_count`、`nt_type`、`cell_type` 等別名；不把長整數 root ID 經過浮點數轉換。未知端點、重複 neuron ID、不合法索引、非正或非有限權重直接報錯。重複有向節點對會加總接觸數，並在 metadata 記錄加總列數。edge `neuropil` 記錄為來源各 neuropil 的接觸數統計；它不等於神經元 region，不能由此推得 soma 位置。`subgraph(region=...)` 只依神經元 region 選誘導子圖，不是按突觸 neuropil 篩選。

SQLite 使用 neuron 與 synapse 表、索引與外鍵，metadata 以 JSON 保存。寫入先完成臨時資料庫再原子替換；預設不覆蓋既有路徑，確定重建時傳 `overwrite=True`。網路鄰接查詢可選 upstream、downstream、both 與 hop 數，回傳包含起點的可達 ID。

## 合成資料與控制圖

`synthetic_graph(seed=0)` 提供 24 sensory、32 inter、12 motor 的 68 節點測試網路。所有連接、NT、座標都由程式生成；對每個輸入到 12 個輸出的直接邊使用相同初始接觸數，避免預先寫入輸入—動作對應。

`degree_preserving_shuffle` 以有向雙邊交換保留逐節點入度／出度；權重跟著來源邊，因此也保留逐來源輸出強度與全域權重集合，但不保留輸入強度或 motifs。metadata 記錄要求交換次數、成功次數、嘗試次數；狹小或剛性圖可能無法充分交換，且有限交換不保證均勻混合。

`random_control` 使用有向 G(n,m)，保留節點數、邊數與全域接觸數集合，不保留逐節點度數、強度或原始符號分布。兩類控制都不產生自連結或重複有向邊；它們的生物拓樸標示為 false。只有保持刺激、模型參數、訓練量、隨機種子集與評估規則一致的比較，才可用于評估本模型對連接拓樸的敏感度。
