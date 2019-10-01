# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-download-web-images/
Created on Wed Oct 31 11:03:21 2018

@author: Akitaka
"""
import urllib.error
import urllib.request

def download_image(url, dst_path):
    try:
        data = urllib.request.urlopen(url).read()
        with open(dst_path, mode="wb") as f:
            f.write(data)
    except urllib.error.URLError as e:
        print(e)

url = 'https://upload.wikimedia.org/wikipedia/en/2/24/Lenna.png'
dst_path = 'data/src/lena_square.png'
# dst_dir = 'data/src'
# dst_path = os.path.join(dst_dir, os.path.basename(url))
download_image(url, dst_path)

error_url = 'https://upload.wikimedia.org/wikipedia/en/2/24/Lenna_xxx.png'
download_image(error_url, dst_path)
# HTTP Error 404: Not Found

url_list = []
for i in range(5):
    url = 'http://example.com/basedir/base_{:03}.jpg'.format(i)
    url_list.append(url)

from bs4 import BeautifulSoup

url = 'https://news.yahoo.co.jp/photo/'
ua = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) '\
     'AppleWebKit/537.36 (KHTML, like Gecko) '\
     'Chrome/55.0.2883.95 Safari/537.36 '

req = urllib.request.Request(url, headers={'User-Agent': ua})
html = urllib.request.urlopen(req)

soup = BeautifulSoup(html, "html.parser")

img_list = soup.find(class_='headline').find_all('img')
url_list = []
for img in img_list:
    url_list.append(img.get('src'))

download_dir = 'temp/dir'
sleep_time_sec = 1

for url in url_list:
    filename = os.path.basename(url)
    dst_path = os.path.join(download_dir, filename)
    time.sleep(sleep_time_sec)
    print(url, dst_path)
    # download_image(url, dst_path)


