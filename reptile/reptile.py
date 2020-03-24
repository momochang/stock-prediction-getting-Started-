import re
import sys
import requests
import tensorflow as tf
from time import time, sleep
from bs4 import BeautifulSoup
from datetime import datetime   
from selenium import webdriver
from collections import Counter

class Date:
    def __init__(self):
        self.year = int(datetime.now().year)
        self.month = int(datetime.now().month)
        self.day = int(datetime.now().day)        

class Crawler(Date):
    def __init__(self, driver, name, url, path, mode, encode):
        self.driver = driver
        self.name = name
        self.url = url + '/search/word/2/' + self.name
        self.path = path
        self.mode = mode
        self.encode = encode
        self.title = []
        self.href = []

    def connect(self):
        try:
            self.driver.get(self.url)
            self.driver.maximize_window()
            print('Chrome Web Connect Sucess')
        except Exception as e:
            print('error ', sys.exc_info()[0])

    ### Via BeautifulSoup to Load relative information from HTML
    def parser(self):
        try:
            soap = self.driver.page_source.encode('utf-8').strip()
            self.soap = BeautifulSoup(soap, 'html.parser')
            print('HTML Parser Sucess')
        except Exception as e:
            print('error ', sys.exc_info()[0])


    ### write content into txt file
    def write(self, title, date, href, content):
        with open('Crawler.txt', 'a', encoding = 'utf8') as f:
            f.write(title + '\n')
            f.write(date + '\n')
            f.write(href + '\n')
            f.write(content + '\n')


    ### get the next 10 pages href
    def get_page(self):
        block = self.soap.find_all("a", {"class" : ""})
        for page in block:
            try:
                if page.string != None:
                    ### use regular expression to filted \n 、\t
                    test = re.split(r"(\W+)", str(page.string))[2]
                    if test.isdigit():
                        if test in self.title:
                            pass
                        else:
                            self.title.append(test)
                            self.href.append(page['href'])
            except:
                pass       

    ### get all off results in current page 
    def result(self):
        block = self.soap.find_all("div", {"class" : "news"})
        for i in block:
            title = i.find('a').text.split('.')[-1]
            href = i.find('a')['href']
            content = i.find('p').text
            date = i.find('span').text.split('．')[0]
            print(title)
            print(date)
            print(href)
            print(content)

            self.write(title, date, href, content)
            

    ### open the new window and jump it
    def open_(self):
        for href in self.href:
            self.driver.execute_script("window.open(arguments[0])", href)
            windows = self.driver.window_handles
            self.driver.switch_to.window(windows[-1])

            try:
                sp = self.driver.page_source.encode('utf-8').strip()
                self.sp = BeautifulSoup(sp, 'html.parser')
                print('==> Connect Sucess')
            except Exception as e:
                print('error ', sys.exc_info()[0])

            self.result()
            #self.driver.close()
            sleep(5)

    ### find the advanced search result
    def search(self):
        advance = self.soap.find_all("div", {"class" : "search-more"})
        for i in advance:
            self.driver.get(i.find('a')['href'])
            soap = self.driver.page_source.encode('utf-8').strip()
            self.soap = BeautifulSoup(soap, 'html.parser')
            self.result()
            self.get_page()
            self.open_()
        self.driver.close()
    



def main():
    chromedriver = "C:/Users/ML/Desktop/culture/chromedriver_win32/chromedriver.exe"
    driver = webdriver.Chrome(chromedriver)
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('name', '台指期', 'the reptiled name')
    tf.app.flags.DEFINE_string('url', 'https://udn.com', 'United News Network web url')
    tf.app.flags.DEFINE_string('path', 'C:/Users/ML/Desktop/culture/reptile/data', 'stored reptil result path')
    tf.app.flags.DEFINE_string('mode', 'w', 'open file that use something mode')
    tf.app.flags.DEFINE_string('encode', 'utf-8', 'file encode')
    crawler = Crawler(
                    driver,
                    FLAGS.name, 
                    FLAGS.url, 
                    FLAGS.path, 
                    FLAGS.mode, 
                    FLAGS.encode,
                )
    crawler.connect()
    crawler.parser()
    crawler.search()

if __name__ == '__main__':
    main()
