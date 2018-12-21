import pandas  as pd

print_list=pd.read_csv(r'C:\Users\synapse\Desktop\12_21_2018_data_paths\print_list.csv')

for index, row in print_list.iterrows():
    print(row['disk_path'], row['final_path'])