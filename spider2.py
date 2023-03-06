import requests
import os
import re
import random
import time
import csv


HEADERS = {'Host': 'api.taptapdada.com',
           'Connection': 'Keep-Alive',
           'Accept-Encoding': 'gzip',
           'User-Agent': 'okhttp/3.10.0'}

BASE_URL = 'https://api.taptapdada.com/review/v1/by-app?sort=new&app_id={}' \
            '&X-UA=V%3D1%26PN%3DTapTap%26VN_CODE%3D593%26LOC%3DCN%26LANG%3Dzh_CN%26CH%3Ddefault' \
            '%26UID%3D8a5b2b39-ad33-40f3-8634-eef5dcba01e4%26VID%3D7595643&from={}'



class TapSpiderByRequests:
    def __init__(self, csv_save_path, game_id):
        self.start_from = 0
        self.reviews = []
        self.spider(csv_save_path, game_id)

    def spider(self, csv_save_path, game_id):
        end_from = self.start_from + 8600
        for i in range(self.start_from, end_from+1, 10):
            url = BASE_URL.format(game_id, i)
            try:
                resp = requests.get(url, headers=HEADERS).json()
                resp = resp.get('data').get('list')
                self.parse_info(resp)
                print('=============已爬取第 %d 页=============' % int(i/10))
                if i != end_from:
                    print('爬虫等待中...')
                    pause = random.uniform(0, 0.5)
                    time.sleep(pause)
                    print('等待完成，准备翻页。')
            except Exception as error:
                print('爬取第%i页出现异常，异常信息如下：' % int(i/10))
                raise error
                exit()

        self.write_csv(csv_save_path, self.reviews)

    def parse_info(self, resp):
        for r in resp:
            review = {}
            review['id'] = r.get('id')
            review['author'] = r.get('author').get('name').encode('gbk', 'ignore').decode('gbk')
            review['updated_time'] = r.get('updated_time')
            review['device'] = r.get('device').encode('gbk', 'ignore').decode('gbk')
            review['spent'] = r.get('spent')
            review['stars'] = r.get('score')
            content = r.get('contents').get('text').strip()
            review['contents'] = re.sub('<br />|&nbsp', '', content).encode('gbk', 'ignore').decode('gbk')
            review['ups'] = r.get('ups')
            review['downs'] = r.get('downs')
            self.reviews.append(review)

    def write_csv(self, full_path, reviews):
        title = reviews[0].keys()
        path, file_name = os.path.split(full_path)
        if os.path.exists(full_path):
            with open(full_path, 'a+', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, title)
                writer.writerows(reviews)
        else:
            try:
                os.mkdir(path)
            except Exception:
                print('路径已存在。')
            with open(full_path, 'a+', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, title)
                writer.writeheader()
                writer.writerows(reviews)


if __name__ == '__main__':
    csv_save_path = r'D:\360MoveData\Users\10539\Desktop\data.csv'
    game_id = [168332, 2301, 62448, 43639, 58885, 192675, 177635, 165287, 31597, 70056]
    for id in game_id:
            TapSpiderByRequests(csv_save_path, id)