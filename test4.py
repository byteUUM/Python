# 练习准备
# 先建一个 config.json
# {
#   "model": "gpt-4.1",
#   "temperature": 0.7,
#   "max_tokens": 500
# }
# 再建一个 prompt.txt
# Explain {topic} in simple terms.

# 练习题
# 题 1：读取 JSON
# 读 config.json，打印 model
import json
with open("./config.json","r") as fr:
  conf = json.load(fr)
print(conf)

# 题 2：修改配置
# 把 temperature 改成 0.2
conf["temperature"] = 0.2

# 题 3：保存新文件
# 把修改后的内容写入 config_new.json
with open("./config.json","w") as fw:
  json.dump(conf,fw)

# 题 4：读取文本
# 读 prompt.txt，打印内容
with open("./prompt.txt","r") as fp:
  p_text = fp.read()
print(p_text)

# # 题 5：字符串处理
# # 用户输入 " LangChain "，去掉前后空格并转小写后打印

n = " LangChain "
n = n.strip().lower()
print(n)