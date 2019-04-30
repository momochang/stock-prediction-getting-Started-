import requests
import urllib.request
import zipfile
import numpy as np
from bs4 import BeautifulSoup

def download():
    r = requests.get('http://www.taifex.com.tw/cht/3/futPrevious30DaysSalesData')
    
    if r.status_code == requests.codes.ok:
        
        soup = BeautifulSoup(r.text, 'html.parser')
        stories = soup.find_all(class_='table_c')
        #value = soup.find('input', {'id': 'button7'}).get('onclick')
        for i in stories:
            #print(i)
            value = i.select('td')[3].find('input', {'id': 'button7'}).get('onclick')
            v = value.split("'")
            print(v[1])
        
        v1 = v[1].split("/")
        urllib.request.urlretrieve(v[1],v1[7])
        #f=zipfile.ZipFile('D:/Python/專題/下載台指期每日行情/'+  v1[7])
        f=zipfile.ZipFile(v1[7])
        
    
    text_name = v1[7].split(".")
    file_name = text_name[0]
    zip_name = v1[7]
    with zipfile.ZipFile(zip_name, 'r') as myzip:
        myzip.extract(file_name + '.csv')
    return file_name
#download()

    