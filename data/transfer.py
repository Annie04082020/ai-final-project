import os
import shutil

# 設定源資料夾路径
source_folder = 'tw'
# 设定目标資料夾路径
destination_folder = '../additional_data/tw'
# 確保目标資料夾存在，如果不存在就創建一个
os.makedirs(destination_folder, exist_ok=True)
# 設定你想要在源資料夾中保留的檔案數量
num_files_to_keep = 1000

# 獲取源資料夾內所有檔案的列表
files = os.listdir(source_folder)
# 確保檔案列表按日期排序或其他任何標準
# 例如: 使用檔案修改時間來排序
files.sort(key=lambda x: os.path.getmtime(os.path.join(source_folder, x)))

# 如果檔案數量超過了需要保留的數量，則將多餘的檔案移到目标資料夾
if len(files) > num_files_to_keep:
    files_to_move = files[num_files_to_keep:]
    for file in files_to_move:
        shutil.move(os.path.join(source_folder, file), destination_folder)
        print(f'Moved {file} to {destination_folder}')

print('Finished moving files.')

