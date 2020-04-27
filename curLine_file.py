# -*- coding: utf-8 -*-
import sys
def curLine():
    file_path = sys._getframe().f_back.f_code.co_filename  # 获取调用函数的路径
    file_name=file_path[file_path.rfind("/") + 1:] # 获取调用函数所在的文件名
    lineno=sys._getframe().f_back.f_lineno#当前行号
    str="[%s:%s] "%(file_name,lineno)
    return str

stop_words = set(('/n', '/r', '/t', "阿", "啊", "了", "哈", "吗", "呀", "呢", "吧", "诶", "的", "是")) # , "么"
stop_mark = set((",", ".", "?", "!", ":"))  # 已经全角转半角，故替换英文标点即可
#     """全角转半角"""
def quanjiao2banjiao(query_lower):
    rstring = list()
    for uchar in query_lower:
        if uchar in stop_words:
            continue
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        uchar = chr(inside_code)
        if uchar not in stop_mark:
            rstring.append(uchar)
    rerturn_str = "".join(rstring)
    return rerturn_str
# 大写变小写，　全角变半角，　去标点　注意训练语料和模板也要改
def normal_transformer(query):
    query_lower = query.strip("’,?!，。？！� \s").lower()
    rstring = quanjiao2banjiao(query_lower).strip().replace("您","你").replace("她","他")
    return rstring