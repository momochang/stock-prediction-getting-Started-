#import jieba
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
#from wordcloud import WordCloud
from bs4 import BeautifulSoup

q = '台指期'
count = 1
start = 0
lenhe_url = ('https://udn.com/search/result/2/')
year = 2018
month = 12
test = str(month) + '-' + input("請輸入日期: ")

file = open('D:/Python/專題/資料/聯合新聞網搜尋結果(' + str(year) + '-' + test +').txt', 'a', encoding = 'utf-8')
file2 = open('D:/Python/專題/資料/聯合新聞網搜尋結果(' + str(year) + '-' + test + ').notime.txt', 'a', encoding = 'utf-8')

for i in range(1,2):
    if i == 1:
        lenhe_url = lenhe_url + q
        r = requests.get(lenhe_url)
    else:
        reg = '/' + str(i)
        r = requests.get(lenhe_url + q + reg)
        
    if r.status_code == requests.codes.ok:
        #print("OK!")
        soup = BeautifulSoup(r.text, 'html.parser')
        stories = soup.find_all('div', id="search_content")

        for s in stories:
            #當頁資料的筆數共20筆
            for pr in range(0,20):
                print("第",count,"筆資料--------------------")
                print("標題: " + s.select('h2')[pr].text)#輸出新聞標題
                print("發布來源、時間: " + s.select('span')[pr].text)#輸出發布地點、時間
                #抓取每筆新聞標題的網址
                time_url = s.select('dt')[pr].find('a')['href']
                
                r2 = requests.get(time_url)
                if r2.status_code == requests.codes.ok:
                    soupr2 = BeautifulSoup(r2.text, 'html.parser')
                    #找尋符合story_bady_info_author的資料，也就是詳細的時間
                    storiesr2 = soupr2.find_all(class_='story_bady_info_author')
                    for sr2 in storiesr2:
                        #輸出詳細時間
                        print("發布時間(詳細): " + sr2.select('span')[0].text)
                        #如果輸入的日期符合詳細時間，就寫到文檔中
                        if  test in sr2.select('span')[0].text:
                            file.write("第"+ str(count) +"筆資料: " + s.select('h2')[pr].text + "\n")
                            file.write("發布來源、時間: " + s.select('span')[pr].text + "\n")
                            file.write("發布時間(詳細): " + sr2.select('span')[0].text + "\n")
                            file2.write(s.select('h2')[pr].text + "。")
                            
                            
                count += 1#資料筆數+1
        time.sleep(5)#休息5秒鐘

file.close()
file2.close()
