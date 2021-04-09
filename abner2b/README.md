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
`cd DSAI_HW2_AutoTrading`

2. 安裝所需套件\
`pip install -r requirements.txt`

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
`python main.py --training training.csv -- testing testing.csv --output output.csv`

5. 驗證最終收益 (**該步驟可略**)
`python profit_calculator.py --stock testing.csv --action output.csv`

## 資料來源
IBM公司過去特定時間段的每日股市開盤價、最高價、最低價與收盤價，合計四項數值。\
[[Stock History Reference](https://www.nasdaq.com/market-activity/stocks/ibm)]

## 分析
![最後365天股市圖](https://i.imgur.com/EZzK8Dl.png)
![拆解trend](https://i.imgur.com/zlQKmNp.jpg)

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
D_tes = functions._tsnor(mu, std, Val_tot)
```
此處為避免使用到未來的測試資料，根據訓練資料與測試資料為**同一連續資料**的原則，\
故使用訓練資料的**平均值**與**標準差**進行正規化。

### Sliding Windows
```py
# Fill the empty of testing data
Val_tot = np.concatenate((Data[((config.tap-1)*-1):, :], Val))
# Sliding windows
D_tra_T, L_tra_T = functions._pack(D_tra, config.tap), functions._pack(L_tra, config.tap)
D_tes_T, L_tes_T = functions._pack(D_tes, config.tap), functions._pack(L_tes, config.tap)
# Remove redundant data
D_tes_T = D_tes_T[(config.tap-1):,:,:]
L_tes_T = L_tes_T[(config.tap-1):,:,:]
```
由於訓練資料為整段連續資料的開頭，
## 結果