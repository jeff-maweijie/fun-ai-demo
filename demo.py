import os
from dotenv import load_dotenv
import json
import sqlite3
from openai import OpenAI, APIConnectionError, APITimeoutError

# ==========================================
# 1. 数据库持久化层 (SQLite)
# ==========================================
DB_FILE = "chat_memory.db"

def init_db():
    """初始化数据库"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # 存储持久化记忆，user_id 为主键
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_memory (
            user_id TEXT PRIMARY KEY,
            facts TEXT
        )
    ''')
    conn.commit()
    conn.close()

def get_memory(user_id):
    """读取用户持久记忆"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT facts FROM user_memory WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return {}

def update_memory(user_id, new_facts):
    """更新用户持久记忆（覆盖旧值）"""
    if not new_facts:
        return
    current_facts = get_memory(user_id)
    current_facts.update(new_facts)  # 字典 update 实现覆盖
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_memory (user_id, facts) VALUES (?, ?)
        ON CONFLICT(user_id) DO UPDATE SET facts=excluded.facts
    ''', (user_id, json.dumps(current_facts, ensure_ascii=False)))
    conn.commit()
    conn.close()

# ==========================================
# 2. 大模型调用与核心逻辑
# ==========================================
class ChatBot:
    def __init__(self):
        # 从环境变量获取 API Key，如果没有则进入 Mock 模式
        # self.api_key = os.getenv("DEEPSEEK_API_KEY", "your_api_key_here")

        # 改为读取.env文件中的API key，方便文件修改演示
        load_dotenv()  # 加载 .env 文件中的环境变量
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "your_api_key_here")

        self.is_mock = self.api_key == "your_api_key_here"
        
        if not self.is_mock:
            # 这里以 DeepSeek 为例，如果是其他模型请修改 base_url
            self.client = OpenAI(
                api_key=self.api_key, 
                base_url="https://api.deepseek.com/v1"
            )
        # 短期上下文记忆 (字典结构: {user_id: [history_msgs]})
        self.context_window = {} 

    def get_recent_context(self, user_id):
        """获取最近6轮上下文"""
        return self.context_window.get(user_id, [])

    def append_context(self, user_id, role, content):
        """追加上下文，保留最近6轮(即12条消息)"""
        if user_id not in self.context_window:
            self.context_window[user_id] = []
        self.context_window[user_id].append({"role": role, "content": content})
        
        # 裁剪历史：保留最近 12 条消息 (6个用户+6个AI)
        if len(self.context_window[user_id]) > 12:
            self.context_window[user_id] = self.context_window[user_id][-12:]

    def chat(self, user_id, user_input):
        # 1. 读取持久记忆和上下文
        persistent_facts = get_memory(user_id)
        recent_context = self.get_recent_context(user_id)

        # 2. 构建 System Prompt
        system_prompt = f"""
        你是一个有记忆的AI助手。
        当前用户的已知事实(记忆)：{json.dumps(persistent_facts, ensure_ascii=False)}
        请分析用户的输入，并严格返回 JSON 格式，不要包含其他多余字符。
        格式如下：
        {{
            "emotion": "情绪标签(必须在: 开心, 低落, 焦虑, 平静, 无法判断 中选择一个)",
            "reply": "你的回复，需适应用户当前情绪",
            "memory_update": {{"键": "值"}} 
            // 如果用户提到了新的称呼、喜好，请放入这里。如果没有，请返回空字典 {{}}
        }}
        """

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(recent_context)
        messages.append({"role": "user", "content": user_input})

        # 3. 调用大模型 (加入异常处理)
        try:
            if self.is_mock:
                # Mock 测试逻辑：模拟不同场景
                if "小林" in user_input:
                    res_json = {"emotion": "平静", "reply": "好的小林，记下了你喜欢喝拿铁。", "memory_update": {"name": "小林", "drink": "拿铁"}}
                elif "批评" in user_input:
                    res_json = {"emotion": "低落", "reply": "抱歉听到这个，开会挨批确实不好受。", "memory_update": {}}
                elif "怎么办" in user_input:
                    res_json = {"emotion": "低落", "reply": "关于开会挨批的事，建议你先复盘一下问题。", "memory_update": {}}
                elif "茶" in user_input:
                    res_json = {"emotion": "平静", "reply": "好的，改成喝茶了。", "memory_update": {"drink": "茶"}}
                elif "喝什么" in user_input:
                    drink = persistent_facts.get("drink", "不知道")
                    res_json = {"emotion": "平静", "reply": f"你之前说喜欢喝{drink}。", "memory_update": {}}
                else:
                    res_json = {"emotion": "平静", "reply": "收到。", "memory_update": {}}
                return res_json
            else:
                # 真实 API 调用
                response = self.client.chat.completions.create(
                    model="deepseek-chat", # 或 gpt-4o-mini 等
                    messages=messages,
                    response_format={"type": "json_object"},
                    timeout=10.0 # 设置超时时间
                )
                return json.loads(response.choices[0].message.content)

        except (APITimeoutError, APIConnectionError) as e:
            # 捕获网络超时或连接异常
            print(f"\n[系统提示] 模型调用失败: 网络超时或连接错误。")
            return None
        except Exception as e:
            print(f"\n[系统提示] 模型调用失败: {str(e)}")
            return None

# ==========================================
# 3. 命令行交互界面
# ==========================================
def main():
    init_db()
    bot = ChatBot()
    
    print("="*40)
    print("AI 对话 Demo 启动！(输入 'quit' 退出)")
    print("请先选择测试用户: 输入 1 (用户A) 或 2 (用户B)")
    print("="*40)
    
    current_user = ""
    while current_user not in ["1", "2"]:
        current_user = input("请选择当前用户 (1/2): ").strip()
    
    user_id = f"User_{current_user}"
    print(f"当前登录: {user_id}")

    while True:
        user_input = input("\n你: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        if not user_input.strip():
            continue

        # 调用大模型获取结果
        result = bot.chat(user_id, user_input)

        # 核心可靠性要求：失败时给出提示，且不写入记忆
        if result is None:
            print("AI: [调用失败，请重试。本次对话未保存]")
            continue

        # 提取结果
        emotion = result.get("emotion", "无法判断")
        reply = result.get("reply", "")
        memory_update = result.get("memory_update", {})

        # 打印回复和情绪标签
        print(f"AI [{emotion}]: {reply}")

        # 写入记忆与上下文
        if memory_update:
            update_memory(user_id, memory_update)
            # 调试用：可打印当前记忆
            # print(f"  (后台日志: 记忆已更新 -> {get_memory(user_id)})")

        # 保存到短期上下文
        bot.append_context(user_id, "user", user_input)
        bot.append_context(user_id, "assistant", reply)

if __name__ == "__main__":
    main()