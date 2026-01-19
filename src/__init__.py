import getpass
import os

import dotenv

dotenv.load_dotenv()

# 1. 检查并设置 API Key
if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter your DeepSeek API key: ")
