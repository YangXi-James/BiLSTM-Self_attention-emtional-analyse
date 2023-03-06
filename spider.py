import requests
import os
import re
import random
import time
import csv
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent


HEADERS = {'Host': 'api.taptapdada.com',
           'Connection': 'Keep-Alive',
           'Accept-Encoding': 'gzip',
           'User-Agent': 'Mozilla/5.0'}
STOP_POINT_FILE = 'stop_point.txt'
path = r'D:\360MoveData\Users\10539\Desktop\data.csv'
total_id = [168332, 2301, 62448, 43639, 58885, 192675, 177635, 165287, 31597, 70056]
id=2301
name = ['原神', '王者荣耀', '光遇', '我的世界',  '月圆之夜', '幻塔', '哈利波特：魔法觉醒', 'Phigros', '碧蓝航线', '和平精英']



def spider():
    end_from = 200
    for i in range(1, end_from,10):
        url = 'https://api.taptapdada.com/review/v1/by-app?sort=new&app_id={}' \
            '&X-UA=V%3D1%26PN%3DTapTap%26VN_CODE%3D593%26LOC%3DCN%26LANG%3Dzh_CN%26CH%3Ddefault' \
            '%26UID%3D8a5b2b39-ad33-40f3-8634-eef5dcba01e4%26VID%3D7595643&from={}'.format(2301,i)
        try:
            inform = requests.get(url, headers=HEADERS).json()
            result = inform.get('data').get('list')
            for r in result:
                review = {}
                # 游戏名称
                # id
                review['id'] = r.get('id')
                # 昵称
                review['author'] = r.get('author').get('name').encode('gbk', 'ignore').decode('gbk')
                # 评论时间
                review['updated_time'] = r.get('updated_time')
                # 游玩时长（分钟）
                review['spent'] = r.get('spent')
                # 打分
                review['stars'] = r.get('score')
                # 评论内容
                content = r.get('contents').get('text').strip()
                review['contents'] = re.sub('<br />|&nbsp', '', content)
                # 支持度
                review['ups'] = r.get('ups')
                # 不支持度
                review['downs'] = r.get('downs')
                reviews.append(review)
            print('已爬取第 %d 页' % i)
            # if i != end_from:
            print('爬虫等待中...')
            pause = random.uniform(0, 0.5)
            time.sleep(pause)
            print('等待完成，准备翻页。')
            # else:
            #     with open(STOP_POINT_FILE, 'w') as f:
            #         f.write(str(i+10))

        except Exception as error:
            with open(STOP_POINT_FILE, 'w') as f:
                f.write(str(i))
            print('爬取第%i页出现异常，断点已保存，异常信息如下：' % int(i/10))
            raise error
            exit()
        write_csv(path, reviews)


# def resume():
#     start_from = 0
#     if os.path.exists(STOP_POINT_FILE):
#         with open(STOP_POINT_FILE, 'r') as f:
#             start_from = int(f.readline())
#     return start_from


def write_csv( full_path, reviews):
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
            print('error')
        with open(full_path, 'a+',  encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, title)
            writer.writeheader()
            writer.writerows(reviews)


# start_from = resume()
reviews = []
spider()
