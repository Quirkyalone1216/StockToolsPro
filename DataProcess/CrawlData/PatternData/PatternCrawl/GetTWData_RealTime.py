import datetime
from PatternCrawl_CalAllPattern import CalAllPattern
from PatternCrawl_SortResult import *
import discord
from dotenv import load_dotenv
import os
import re
import shutil
import requests


def LineSetting():
    load_dotenv('Token\\LineToken_StockBot.env')

    # To get Line StockBot Chatroom's Token
    Line_TOKEN = os.getenv('Line_TOKEN')

    LineStockMessage(Line_TOKEN)


def LineStockMessage(LineToken):
    url = 'https://notify-api.line.me/api/notify'
    token = LineToken
    headers = {
        'Authorization': 'Bearer ' + token  # 設定TOKEN
    }
    data = {
        'message': '測試一下！'  # 設定要發送的訊息
    }
    data = requests.post(url, headers=headers, data=data)  # 使用 POST 方法
    print(data)


def DiscordSetting():
    load_dotenv('Token\\DiscordToken.env')

    # Discord bot 的 Token
    DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')

    # intents
    intents = discord.Intents.default()
    intents.message_content = True  # NEW 新增要求讀取訊息權限
    # client
    client = discord.Client(intents=intents)

    DiscordExecute(client, DISCORD_TOKEN)


def DiscordExecute(client, DISCORD_TOKEN):
    # 事件處理：當 bot 成功登入時
    @client.event
    async def on_ready():
        print(f'Logged in as {client.user}')

    # 事件處理：監聽訊息
    @client.event
    async def on_message(message):
        # 如果訊息來自 bot 自身，不做任何事
        if message.author == client.user:
            return

        if re.match(r"\d{4}\.TW", message.content):
            print(f"Received message from {message.author}: {message.content}")
            content = GetTW_RealTime_Data(message.content)
            try:
                # 由於 Discord 訊息有字元限制，可能需要分段發送或截斷
                await message.channel.send(content[:2000])  # 發送前 2000 個字符
            except Exception as e:
                await message.channel.send(f'Error reading file: {e}')

    # 使用 Token 啟動 bot
    client.run(DISCORD_TOKEN)


def removeDir(directory):
    for DirName in directory:
        for filename in os.listdir(DirName):
            file_path = os.path.join(DirName, filename)
            try:
                # 檢查是否為文件，然後刪除
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # 如果是目錄，刪除整個目錄
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


def ReadRealTimeData(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        content = file.read()

    return content


def GetTW_RealTime_Data(message_content):
    rmDirList = ['sort_test', 'test']
    removeDir(rmDirList)

    # 獲取今天的日期
    today = datetime.date.today()

    # 使用timedelta增加一天
    time_end = (today + datetime.timedelta(days=1)).strftime("%Y%m%d")
    time_start = (today - datetime.timedelta(days=6)).strftime("%Y%m%d")

    SDate = time_start
    EData = time_end
    Prod = message_content
    Kind = 'Stock'
    Cycle = '30m'

    os.makedirs('test', exist_ok=True)
    CalAllPattern(SDate, EData, Prod, Kind, Cycle, 'test')

    result_path = 'test'
    output_dir = 'sort_test'
    outputFileName = 'Pattern_TW_Test.txt'
    mode = 'Highlight'
    region = 'TW'
    SortOutput(result_path, output_dir, outputFileName, mode, region)

    filePath = "Pattern_TW_Test.txt"
    content = ReadRealTimeData(filePath)
    return content


if __name__ == '__main__':
    DiscordSetting()  # Discord
    # LineSetting()
