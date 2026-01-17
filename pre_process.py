import pandas as pd
import os
import glob
import openpyxl

name_in = ["1hour_Commercial", "1hour_Office", "1hour_Public", "1hour_Residential",
           "5min_Commercial", "5min_Office", "5min_Public", "5min_Residential",
           "30min_Residential", "30min_Commercial", "30min_Office", "30min_Public",
]

time_keys = ["1hour", "5min", "30min"]

def simple_merge_excel(folder_path, output_file):
    """简单合并Excel文件，自动处理缺失值和异常值"""

    # 获取所有Excel文件
    files = glob.glob(os.path.join(folder_path, "*.xlsx")) + \
            glob.glob(os.path.join(folder_path, "*.xls"))

    if not files:
        print("未找到Excel文件")
        return

    # 读取所有文件
    dfs = []
    for file in files:
        df = pd.read_excel(file)
        dfs.append(df)

    # 找到所有文件的共同列
    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols = common_cols.intersection(set(df.columns))

    common_cols = list(common_cols)

    # 合并共同列
    merged_df = pd.concat([df[common_cols] for df in dfs], ignore_index=True)

    # 处理缺失值 - 数值列用中位数填充，其他列用前向填充
    '''
    for col in merged_df.columns:
        if merged_df[col].dtype in ['float64', 'int64']:
            # 数值列：先用前向填充，再用后向填充
            merged_df = merged_df.dropna()
            #merged_df[col] = merged_df[col].fillna(method='ffill').fillna(method='bfill')
            # 如果还有缺失值，用中位数填充
            if merged_df[col].isnull().any():
                merged_df = merged_df.dropna()
                #merged_df[col] = merged_df[col].fillna(merged_df[col].median())
        else:
            # 非数值列：用前向填充
            merged_df = merged_df.dropna()
            #merged_df[col] = merged_df[col].fillna(method='ffill').fillna(method='bfill')
    '''

    # 处理异常值 - 使用IQR方法
    numeric_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        Q1 = merged_df[col].quantile(0.25)
        Q3 = merged_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 将异常值限制在边界内
        merged_df[col] = merged_df[col].clip(lower=lower_bound, upper=upper_bound)

    # 保存结果
    merged_df.to_excel(output_file, index=False, engine = 'openpyxl')
    print(f"数据已合并保存到: {output_file}")
    print(f"合并了 {len(files)} 个文件")
    print(f"最终数据形状: {merged_df.shape}")

    return merged_df

# 使用示例
#if __name__ == "__main__":
    # 直接调用
time_folder_name = ["1_hour", "5_minutes", "30_minutes"]
time_folder_name_in = {
    "1_hour" : "1hour",
    "5_minutes" : "5min",
    "30_minutes" : "30min"
}
type_folder_name = ["Commercial", "Office", "Public", "Residential"]
year_folder_name = ["2016", "2017", "2018"]
#year_folder_name = ["2016"]


for year in year_folder_name:
    for time in time_folder_name:
        for type1 in type_folder_name:
            simple_merge_excel(
                folder_path = "E:/专业课/MCM/XJTU_C/电力负荷数据/" + year + "/" + time + "/" + year + "_" + time_folder_name_in[time] + "_" + type1,  # 你的Excel文件所在文件夹
                output_file = "./merged_results/" + year + "_merged_result_" + time_folder_name_in[time] + "_" + type1 + ".xlsx", # 输出文件路径
            )

'''
for key1 in name_in:
    simple_merge_excel(
        folder_path = "./merged_results",
        output_file = "./merged_result_" + key1 + ".xlsx", # 输出文件路径
        key = key1
    )
'''
