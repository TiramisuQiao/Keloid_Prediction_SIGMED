import pandas as pd
import pickle

# 读取数据并且将数据保存为pickle文件类型
path = r"C:\Users\wang_\OneDrive\Desktop\datas.xlsx"   # 文件路径
data = pd.read_excel(path)
# print(data.head())    #调试
# print(data.columns)   #调试

need_columns_1 = data[['3.瘢痕发生的部位[多选](一处部位发生多处瘢痕疙瘩)',
       '3 (一处部位发生一处瘢痕疙瘩)', '3 (多处部位,每处部位发生多处瘢痕疙瘩)', '3 (多处部位,每处发生一处瘢痕疙瘩)',
       '4.\t瘢痕疙瘩产生的原因', '5.\t原瘢痕疙瘩处是否瘙痒', '6.\t原瘢痕疙瘩处是否疼痛', '7.\t原瘢痕疙瘩处是否感染',
       '8.\t您的饮食喜欢和爱好(偏好甜食)', '8 (偏好辛辣)', '8 (偏好偏咸)', '8 (平常吸烟)', '8 (平常喝酒)',
       '9.\t您皮肤的性质', '10.\t是否曾有以下疾病', '11.\t您的家族近亲是否有人也有瘢痕疙瘩',
       '12.\t您接受放疗的方式是哪种', '13.\t您的放疗次数', '14.\t放疗后，您采取了以下哪些防疤的措施？[多选](疤痕贴)',
       '14(疤痕药膏)', '14(减张器)', '14(其他)',
       '15.\t疤痕疙瘩是否复发[多选](原手术切除部位长出新的瘢痕(复发标准1))', '15(瘢痕切除后长大变厚变宽(复发标准2))',
       '15(切口处有瘢痕疙瘩迹象(复发标准3))', '15(手术后伤口出现持续瘙痒,疼痛,或肤色变深(复发标准4))', '15(无上述症状)']]

need_columns_2 = data[['16.\t术后多长时间出现复发']]
# 将数据转换为 1 或 0
# 如果值大于 0，则转换为 1；否则转换为 0
binary_list = [1 if x > 0 else 0 for x in need_columns_2.iloc[:, 0]]

# print(need_columns.head())       #调试
nested_list1 = need_columns_1.values.tolist()
# nested_list2 =  need_columns_2.values.flatten().tolist()

# print("转换后的嵌套列表：")
# print(nested_list_str)

# need_columns.to_pickle('new_code/need_data.pkl')
# with open('new_code/data.pkl', 'wb') as file:
#     pickle.dump(nested_list, file)



with open('new_code/data_1.pickle', 'wb') as file:
    pickle.dump(nested_list1, file)

print("success1")

with open('new_code/data_2.pickle', 'wb') as file:
    pickle.dump(binary_list, file)
print("success2")
