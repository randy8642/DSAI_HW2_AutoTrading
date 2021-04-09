# DSAI_HW2_AutoTrading
NCKU DSAI course homework

## 前置工作
### 作業說明
* 說明連結\
[[Dropbox paper](https://paper.dropbox.com/doc/DSAI-HW2-AutoTrading-z7Ke9N2AUZQnPf5NG3ZOt)]
* 目標\
以每日開開盤價、最高價、最低價與收盤價為基準，\
決定明日採取行動後，最大化收益。

### 環境
* python 3.6.4
* Win 10

### 使用方式
1. 進入專案資料夾\
`cd /d [path/to/this/project]`  

2. 安裝所需套件\
`pip install -r requirements.txt`  (**不含PyTorch**，需照以下步驟安裝。)


3. 安裝`PyTorch`\
    **詳見PyTorch官網，如有其他需求也請至該網站下載 [[LINK](https://pytorch.org/get-started/locally/)]**
    * Windows環境
      * 具有GPU (CUDA 11.1)\
        `pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html` 
      * 無GPU\
        `pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`
    * Linux環境
      * 具有GPU (CUDA 11.1)\
        `pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html` 
      * 無GPU\
        `pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`     
4. 執行\
`python main.py --training training.csv --testing testing.csv --output output.csv`

## 資料來源
IBM公司過去某時間段的每日股市開盤價、最高價、最低價與收盤價，合計四項數值。\
[[Stock History Reference](https://www.nasdaq.com/market-activity/stocks/ibm)]

## 分析
首先列出資料後180天的漲幅蠟燭圖與KD線，觀察其漲/跌情形。
![後180天蠟燭圖與KD線](https://i.imgur.com/qYVBZyN.png)

另外透過seasonal decompose分析其開盤價，\
並依序拆解為trend component、seasonal component以及residuals，\
希望藉此觀察是否有規律可循，然並無看見明顯規律。
![拆解trend](https://i.imgur.com/zlQKmNp.jpg)

最後畫出訓練資料與測試資料的趨勢，可發現兩者的確為連續資料，\
同時考量資料量的差異，應可直接使用訓練資料之平均值與標準差，\
對測試資料進行正規化。
![TrTs](https://i.imgur.com/rZjWCKF.jpg)


## 方法說明
### 概述
使用sliding windows的方式切割資料，以過去n天的四項指標預測隔天的開盤價。\
接著根據預測值與今日開盤價做比較，得出漲/跌趨勢，再根據該趨勢決定明日行動。\
(**各參數詳見`config.py`**)

### 讀取資料並正規化
```py
# Load data
Data = np.array(pd.read_csv(os.path.join(P, args.training), header=None))
Val = np.array(pd.read_csv(os.path.join(P, args.testing), header=None))
# Normalization
D_tra, mu, std = functions._nor(Data[:-1, :])
D_tes = functions._tsnor(mu, std, Val)
```
此處為避免使用到未來的測試資料，根據訓練資料與測試資料為**同一連續資料**的原則，\
故使用訓練資料的**平均值**與**標準差**進行正規化。

### Sliding Windows
```py
D_tra_T, L_tra_T = functions._pack(D_tra, config.tap), functions._pack(L_tra, config.tap)
D_tes_T, L_tes_T = functions._pack(D_tes, config.tap), functions._pack(L_tes, config.tap)
```
由於訓練資料與測試資料開頭前n-1天不足window size所需長度(n)，\
故兩段資料的前n-1天皆做**zero padding**補齊長度。

### 將資料包成dataset形式
```py
train_data = torch.from_numpy(D_tra_T).type(torch.FloatTensor)
train_label = torch.from_numpy(L_tra_T).type(torch.FloatTensor)
train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=32, shuffle=True)

test_data = torch.from_numpy(D_tes_T).type(torch.FloatTensor)
test_label = torch.from_numpy(L_tes_T).type(torch.FloatTensor)
test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=1, shuffle=False)
```
為確保測試時，模型能一次只輸入一天的資料，並輸出隔天的預測開盤價，\
因此在`test_dataloader`處**將`batch_size`設為1**；**並把`shuffle`關閉**。

### 測試時輸出隔天行動
```py
# Initialize parameters
out = Val[0,0]
hold = 0
act_tot = []
hold_tot = []
for n_ts, (Data_ts, Label_ts) in enumerate (test_dataloader):
    # Record today open price
    rec = out

    data = Data_ts
    data = data.to(device)
    # Ouput the prediction of tomorrow open price
    out, _ = single_model(data)
    out = out.cpu().data.numpy()
    # Compare the open price above, and decide action
    trend = functions._comp(rec, out)
    act, hold = functions._stock(trend, hold)
```
每次輸入模型之前記錄當天(預測)開盤價，並比較隔天預測值輸出漲/跌趨勢，\
之後根據該趨勢採取行動。採取行動邏輯及模型架構詳見下述。

#### 判斷行動依準
| 持有數量 | 預測明天相對今天 | 採取動作 |
|----------|------------------|----------|
| 1        | 漲 (1)           | 賣 (-1)  |
| 1        | 跌 (-1)          | 無 (0)   |
| 0        | 漲 (1)           | 賣 (-1)  |
| 0        | 跌 (-1)          | 買 (1)   |
| -1       | 漲 (1)           | 無 (0)   |
| -1       | 跌 (-1)          | 買 (1)   |

#### 模型架構
![model](https://i.imgur.com/ROotUbG.png)

## 結果
將預測值與實際開盤價比較化成趨勢圖，如下所示：\
![trend](https://i.imgur.com/2QuAqGm.png)
由該處可發現，雖大致上趨勢皆有符合實際情形，但仍有1天左右的偏移，\
推測此為預測失準的原因所致。
\
訓練時loss的變化圖如下所示：
![loss](https://i.imgur.com/25IFSwV.png)
此處使用Adam作為optimizer；L1 loss作為loss function，**參數詳見`config.py`**。
