import pandas as pd
import json
import os

json_file_address = './ceshiJsonToCsv/0.001-feder0.005.json'
# 使用os.path.basename获取路径的最后一部分，即文件名
file_name_with_extension = os.path.basename(json_file_address)

# 使用os.path.splitext去掉文件扩展名
file_name_without_extension, _ = os.path.splitext(file_name_with_extension)
# 加上后缀.xlsx
file_name_with_xlsx = file_name_without_extension +'.xlsx'

print(file_name_with_xlsx)
print(file_name_without_extension)
with open(json_file_address) as f:
    data = json.loads(f.read())
# df = pd.json_normalize(
#     data,
#     record_path=['train_average_loss_history',
#                  'train_average_acc_history',
#                  'val_average_loss_history',
#                  'val_average_acc_history',
#                  'test_average_acc_history']
# )
df = pd.json_normalize(data, record_path='test_average_acc_history', )
df.to_excel(file_name_with_xlsx)