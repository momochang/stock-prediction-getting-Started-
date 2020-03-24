from bs4 import BeautifulSoup
from selenium import webdriver
from time import sleep
import tensorflow as tf
import sys
import requests
import re
import os


def count_(func):
	def wrapper(*args):
		res = func(*args)
		print(res[0], res[1])
	return wrapper

def check_(filename, mode):
	try:
		file = open(filename, mode)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

class Crawler:
	def __init__(self, url, driver, epoch):
		self.url = url
		self.driver = driver
		self.epoch = epoch

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

	### let mouse scroll to web page bottom and made it load the new data
	@count_
	def load_(self):
		for i in range(1, self.epoch):
			self.driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
			sleep(i)

		self.soap = BeautifulSoup(self.driver.page_source, 'html.parser')
		self.search()
		return ('Your search results totally has ', len(self.hist))


	### Crawler search results from the current page
	#@count_
	def search(self):
		self.hist = self.soap.find_all("div", {"class" : "story-list__text"})
		#return ('Your search results totally has ', len(self.hist))

	### Use regulat expression to filted HTML tags
	def regular(self, content):
		reg = re.compile('<[^>]*>')
		content = reg.sub('', str(content)).replace('\n', '')
		content = content[1:-1]
		return content

	### Open the new window and jump it
	@count_
	def open_(self, result):
		self.driver.execute_script("window.open(arguments[0])", result)
		windows = self.driver.window_handles
		self.driver.switch_to.window(windows[-1])

		try:
			sp = self.driver.page_source.encode('utf-8').strip()
			self.sp = BeautifulSoup(sp, 'html.parser')
			print('==> Connect Sucess')
		except Exception as e:
			print('error ', sys.exc_info()[0])

		### Find the current page that has article position
		advance = self.sp.find_all("section", {"class" : "article-content__editor"})
		self.content = self.regular(advance)
		self.driver.close()
		self.driver.switch_to.window(windows[0])
		return ('The number of Character has ', len(self.content))

	def write_(self, name, mode, encode, title, content):
		with open(name, mode, encoding = encode) as f:
			try:
				f.write(title)
				f.write(content)
			except:
				pass
	

	def run(self, **kwargs):
		leng = len(self.hist)
		for index in self.hist:
			if index.find('a').string != None:
				title = index.find('a').string
			else:
				title = index.find('a').text
			
			if index.find('a')['href'] != None:
				href = index.find('a')['href']
				if len(href) != None:
					print(title)
					print(href, end = ' ')

					self.open_(href)
					self.write_(
						name = kwargs['name'],
						mode = kwargs['mode'],
						encode = kwargs['encode'],
						title = title,
						content = self.content,
						 )
				sleep(10)

		self.driver.close()


def main():
	chromedriver = "C:/Users/ML/Desktop/culture/chromedriver_win32/chromedriver.exe"
	driver = webdriver.Chrome(chromedriver)
	FLAGS = tf.app.flags.FLAGS
	tf.app.flags.DEFINE_string('sourceurl', 'https://udn.com/search/word/2/', 'the News website homepage URL')
	tf.app.flags.DEFINE_string('keyword', '身分', 'search result of keyword')
	tf.app.flags.DEFINE_string('filename', 'article.txt', 'store result in txt file')
	tf.app.flags.DEFINE_string('mode', 'a', 'use mode to open the txt file')
	tf.app.flags.DEFINE_string('encode', 'utf-8', 'use character encode to store result in txt file')
	tf.app.flags.DEFINE_integer('epoch', 10, 'the number of mouse scroll')

	check_(FLAGS.filename, FLAGS.mode)
	craw = Crawler(FLAGS.sourceurl + FLAGS.keyword, driver, FLAGS.epoch)
	craw.connect()
	craw.parser()
	craw.load_()
	craw.run(
		name = FLAGS.filename,
		mode = FLAGS.mode,
		encode = FLAGS.encode,
		)
	sleep(5)

if __name__ == '__main__':
	main()

### find the advanced search result
# advance = soap.find_all("div", { "class" : "context-box__content"})
# for i in advance:
# 	if i.find('a'):
# 		driver.get(i.find('a')['href'])
# 		soap = driver.page_source.encode('utf-8').strip()
# 		soap = BeautifulSoup(soap, 'html.parser')

### find the advanced search result
# for x in mydivs:
# 	course = x.find_all("div",{"class":"story-list__text"})
# 	if len(course) != 0:
# 		#for i in course:
# 		result = course.select('h2')
# 		print(result)
		# for i in range(len(course)):
		# 	print(course[i].text)
		# 	print(course[i])
		# try:
		# 	print(driver.find_elements_by_partial_link_text(course[i].text))
		# except Exception as e:
		# 	driver.close()
		# 	print('error ', sys.exc_info()[0])

	#print('test')
		# print(course[0])
		# print(course[0].text)
		# print(len(course))
		# driver.find_element_by_link_text(course[0].text).click()
		# time.sleep(3)
		# driver.get(sourceurl)








