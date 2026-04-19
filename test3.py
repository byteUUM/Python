# 用这份数据：
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."}
]

# 题 1：打印所有角色
# 输出每条消息的 role
for x in messages:
    print(x["role"])
# for x in messages:
#     print(list(x.keys())[0])
    


# 题 2：打印最后一条内容
# 拿到最后一条消息的 content
print(messages[-1])
print(messages[-1]["content"])

# 题 3：新增一条消息
# 往列表里追加：
# {"role": "user", "content": "Explain it simply."}
messages.append({"role": "user", "content": "Explain it simply."})
print(messages)

# 题 4：筛选 user 消息
# 把所有 role == "user" 的消息找出来并打印
for x in messages:
    if x["role"] == "user":
        print(x,end=" ")
print()

# 题 5：统计消息数
# 输出当前总共有多少条消息
print(len(messages))