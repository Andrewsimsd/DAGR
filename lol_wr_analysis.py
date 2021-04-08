# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 22:54:31 2020

@author: andre
"""

import bs4
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup

url = r'https://na.op.gg/summoner/userName=Kr%C3%BDll'
name = url.split('Name=')[-1]
save_file_dir = r''
uClient = uReq(url)
page_html = uClient.read()
uClient.close()
#html parser
page_soup = soup(page_html, 'html.parser')
users = page_soup.findAll('div', {'class': 'SummonerName'})
win_ratio = int(str(page_soup.findAll('span', {'class': 'winratio'})[0]).split('%')[0].split(' ')[-1])
teams = []
for i, user in enumerate(users):
    if name in str(user):
        if (i%10 < 5):
            teams.append(1)
        else:
            teams.append(2)
            
enemy_win_ratios = []
friendly_win_ratios = []

for i, user in enumerate(users):
    user_url = r'https:' + user.a['href']
    user_uClient = uReq(user_url)
    user_page_html = user_uClient.read()
    user_uClient.close()
    user_page_soup = soup(user_page_html, 'html.parser')
    try:
        user_win_ratio = int(str(user_page_soup.findAll('span', {'class': 'winratio'})[0]).split('%')[0].split(' ')[-1])
    except:
        user_win_ratio = 0
    if name in user_url:
        continue
    if (i%10 < 5) & (teams[i//10] == 1):   
        friendly_win_ratios.append(user_win_ratio)
    elif (i%10 >= 5) & (teams[i//10] == 2):
        friendly_win_ratios.append(user_win_ratio)
    else:
        enemy_win_ratios.append(user_win_ratio)

with open()