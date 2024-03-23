# 導入必要的庫
import discord
from dotenv import load_dotenv
import os

# 載入環境變數
load_dotenv('DiscordToken.env')

# 獲取 Discord 機器人的 Token
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')

# 意圖
intents = discord.Intents.default()
intents.message_content = True  # 新增要求讀取訊息權限
# 客戶端
client = discord.Client(intents=intents)

# 事件處理：當機器人成功登錄時
@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

# 事件處理：監聽訊息
@client.event
async def on_message(message):
    # 如果訊息來自於機器人本身，不做任何事
    if message.author == client.user:
        return

    # 列印接收到的訊息
    print(f"Received message from {message.author}: {message.content}")

    # 如果收到的訊息是 "hello"
    if message.content == 'hello':
        # 回覆 "Hello!"
        await message.channel.send('Hello!')

# 使用 Token 啟動機器人
client.run(DISCORD_TOKEN)
