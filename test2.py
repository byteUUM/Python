# 题 1：欢迎函数
# 写函数 welcome(name)，返回一句欢迎语，不是打印，是返回。
def welcome(name):
    print("Welcome",name)

# 题 2：价格计算函数
# 写函数 calc_cost(tokens, price_per_1k)，按
# tokens / 1000 * price_per_1k
# 返回总价。
def calc_cost(tokens, price_per_1k):
    tokens/1000*price_per_1k
    
# 题 3：遍历列表
# 有列表：
# nums = [3, 5, 8, 13]
# 用 for 循环把每个数字打印出来。
nums = [3,5,8,13]
for i in nums:
    print(i,end=' ')
print()
for i in range(len(nums)):
    print(nums[i],end=' ')
print()
    
# 题 4：找最大值
# 写函数 find_max(numbers)，返回列表最大值。
def find_max(numbers):
    ret = float('-inf')
    for x in numbers:
        if ret<x: ret = x
    return ret

result = find_max([10,56,22,99,2,11,34])
print(result)