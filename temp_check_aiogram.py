import inspect
from aiogram import Bot
print('has download_file:', hasattr(Bot, 'download_file'))
print('signature:', inspect.signature(Bot.download_file))
