import random
text = "一二三四五六七八九十"
temp = ""
for i in range(1000):
    temp += text[random.randint(0,9)]

print(temp)