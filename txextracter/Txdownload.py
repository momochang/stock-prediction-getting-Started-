import requests
import urllib.request
import zipfile
import sys
import tensorflow as tf
from bs4 import BeautifulSoup

class Txdownload:
    def __init__(self):
        pass
    def download_data(self):
        '''
        r: Tx download url
        soup: catch Tx url html type
        stories: find all conforming to class table_c of html type
        value: used select to catch download button
        '''
        try:
            r = requests.get(FLAGS.url)
        except:
            print("Unexpected error:", sys.exc_info()[0])
        else:
            try:
                if r.status_code == requests.codes.ok:
                    soup = BeautifulSoup(r.text, 'html.parser')
                    stories = soup.find_all(class_='table_c')
                for i in stories:
                    value = i.select('td')[3].find('input', {'id': 'button7'}).get('onclick')
                    v = value.split("'")
                    print(v[1])
                    
                v1 = v[1].split("/")
                urllib.request.urlretrieve(v[1],v1[7])
                f = zipfile.ZipFile(v1[7])
            except:
                print("Unexpected error:", sys.exc_info()[0])
            else:
                text_name = v1[7].split(".")
                file_name = text_name[0]
                zip_name = v1[7]
                try:
                    with zipfile.ZipFile(zip_name, 'r') as myzip:
                        myzip.extract(file_name + '.csv')
                    return file_name
                except:
                    print("open csv error")

# TEST #
if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('url', 'http://www.taifex.com.tw/cht/3/futPrevious30DaysSalesData', 'Tx download url')
    data = Txdownload()
    data.download_data()