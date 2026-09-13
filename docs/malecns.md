# MaleCNS：資料範圍、真實外形與完整圖

本專案已接入 **Male Adult Fly Brain and Nerve Cord，MaleCNS v1.0**。這是成年雄性果蠅的腦與腹神經索（VNC）資料，與 FlyWire v783 成年雌性腦是不同個體、不同重建資料。官方提供連接表、註記與神經元骨架，採 CC BY 4.0 授權；本專案固定 v1.0 檔案並校驗 SHA-256。[官方下載與授權](https://male-cns.janelia.org/download/)

## 目前到底用了多少神經元？

| 層次 | 本專案實際內容 |
| --- | --- |
| 完整分類神經元集合 | 166,700 個：來源註記中 `superclass` 非空的全部神經元，包含腦與 VNC |
| 完整匯入的有向連接 | 25,582,837 對，合計 124,177,143 個來源加權接觸；來源突觸信心門檻為 0.5，配對權重門檻為 1 |
| 解剖位置 | 140,638 點：139,662 個 `somaLocation`，加 976 個 `tosomaLocation` 附著位置 |
| 缺少上述位置 | 26,062 個：保留在完整連接圖，沒有在畫面中捏造位置 |
| 網頁預設即時模擬 | 512 個神經元、9,079 對真實有向連接的誘導子圖 |
| 骨架示例 | 12 個來源 SWC 神經元，每個最多顯示 2,500 個原有相鄰節點線段 |

完整圖保留所有分類神經元，即使神經元没有解剖座標或在指定權重門檻下沒有連接。來源完整配對表含 151,856,684 列；其中 126,273,746 列涉及未分類片段而未納入此分類神經元圖，另移除 101 列自連接。神經元對之間重複列先合併，再套用使用者指定的 `min_synapses`。因此，這裡的配對數不能直接與採用「每對至少 5 個突觸」的網站統計比較。

「完整」在此指**此發佈版本的分類中樞神經元集合**，不代表所有未分類片段、膠細胞、整個周邊神經系統或未重建的突觸都齊全。完整連接圖已匯入並實際跑過短時間 LIF 模擬；預設網頁操作和限時訓練使用 512 個神經元，以維持互動速度。

## 為什麼別人的畫面看起來像果蠅大腦？

連接表只記錄「誰連到誰、權重多少」，沒有每條神經纖維的形狀。用力導向圖排列這種資料，通常得到網狀團塊；使用 EM 空間中的真實位置、神經元骨架或 neuropil 表面，才會呈現解剖外形。

本專案的灰色背景使用來源真實位置，骨架示例使用官方 SWC 的原有線段，統一由 8 nm 體素單位換算成 nm。點雲有些點是神經纖維靠近 soma 的附著位置，不能一概稱為胞體。12 個骨架只做形態示例，沒有把它們當成全腦的完整纖維重建；灰色背景也不表示那些神經元正參與目前的 512 節點模擬。彩色活動、膜電位與放電率來自當前模擬迴路。

512 節點迴路由有真實位置的 24 個分散 sensory 節點與 12 個 motor-related 節點作為起點，以有向 BFS 選點，再保留選中節點之間的原始連接。此抽样**不是代表性樣本**：整份資料中僅 30 個 sensory 註記節點有上述 soma/tosoma 位置，因此此視覺化限制會排除許多周邊感覺輸入。來源、選點 ID、角色對應及這項偏差均寫入 `data/malecns/provenance.json`。

## 使用與重現

預設介面與 512 節點範例不需要下載原始資料，也不需要 PyArrow：

```powershell
python -m connectome_lab serve --dataset malecns --open
```

此工作區已產生完整圖 `outputs/full_brain/malecns.npz`，包含節點、真實字串 ID、連接陣列及來源 metadata，讀取僅需 NumPy。大型原始資料與完整圖未加入 Git；其他電腦可依固定來源重建：

```powershell
python -m pip install -e ".[malecns]"
python -m connectome_lab fetch-malecns --archive outputs/full_brain/malecns.npz
```

來源下載約 1.1 GB。已存在的完整圖需要明確加入 `--overwrite` 才覆寫；可加入 `--database outputs/full_brain/malecns.sqlite` 同時寫出 SQLite。PyArrow 只用於讀取原始 Feather 表，平常載入已建好的 NPZ 不需要它。

可直接執行完整圖的短模擬，而不輸出數千萬條邊的網頁：

```python
import numpy as np
from connectome_lab.malecns import load_malecns_archive
from connectome_lab.dynamics import LIFNetwork, activity_metrics

graph = load_malecns_archive("outputs/full_brain/malecns.npz")
network = LIFNetwork(graph)
current = np.zeros(graph.n_neurons)
sensory = [i for i, neuron in enumerate(graph.neurons)
           if neuron.role == "sensory"][:24]
current[sensory] = 3.5
spikes = np.array([network.step(current if t < 15 else np.zeros_like(current))
                   for t in range(30)])
print(graph.n_neurons, graph.n_edges)
print(activity_metrics(spikes))
```

本機實測這個 30 ms 全圖檢查用約 6 秒 CPU 時間；31 個神經元在刺激期間有放電，輸出數值有限。這是資料與動力學可運作的檢查，不能據此推論大腦活動已校準到生理狀態，也不代表完整圖已學會魔方。模型目前採用簡化 LIF、工程式感覺／動作介面，以及局部獎勵調整；傳遞物質、受體、神經調質與身體力學仍有簡化假設。

要重新產生小迴路與解剖資料，可使用：

```python
from connectome_lab.malecns import (
    load_malecns_archive, select_malecns_circuit, save_malecns_anatomy,
)
from connectome_lab.data import export_csv

graph = load_malecns_archive("outputs/full_brain/malecns.npz")
export_csv(select_malecns_circuit(graph, 512), "data/malecns")
save_malecns_anatomy("data/downloads/malecns", "data/malecns/anatomy.npz")
```

`data/malecns/assets.sha256.json` 記錄發佈的小迴路、位置與骨架資產雜湊。`skeletons.json` 逐個保留來源 SWC URL、SHA-256、來源線段數和採樣方式；未新增假連線。原始註記、傳遞物質與連接表的固定 URL 和 SHA-256 位於 `connectome_lab/malecns.py` 及資料 provenance 中。
