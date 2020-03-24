# Txdownload
It can download the Tx information for the day via command line in terminal below:
```
python Txdownload.py 
      [--url]: is Tx download url(default is http://www.taifex.com.tw/cht/3/futPrevious30DaysSalesData)
```
Or you can use command line in terminal below download the Tx information :
```
python Txdownload.py --url=http://www.taifex.com.tw/cht/3/futPrevious30DaysSalesData
```

# Txtransfer
```
python Txtransfer.py 
      [--dataname]: csv file name(ex. Daily_2020_02_11)(default is '')[Need]
      [--exp_mon]: Expiration month(default is 到期月份(週別))
      [--cls_time]: Closing time(default is 成交時間)
      [--recent]: recent prices value(default is 近月價格)
      [--far]: far month prices value(default is 遠月價格)
      [--open]: open auction value(default is 開盤集合競價)
      [--Commodity]: Commodity code to filt columns(default is 商品代號)
      [--fil]: field to select(default is TX)
      [--engine]: read csv engine(default is python)
      [--encode]: read csv encoding(default is utf-8)[Need]
```
It can transfer the Tx information to open、high、low、close and volume features via command line in terminal below:
```
python Txtransfer.py --dataname=Daily_2020_02_11 --encode=big5
```
That the execution time was be below:
```
Use time:  1.847367525100708 s (ex. Daily_2020_02_11)
```
and the result stored in folder that named data
