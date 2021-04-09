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
![最後365天股市圖](/img/candlestick_last365.png)
![拆解trend](/img/decompose.jpeg)

## 方法說明

## 結果