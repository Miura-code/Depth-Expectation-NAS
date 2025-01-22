# Contact: https://github.com/Miura-code
# tensorboardの学習データから各条件のエポック毎の平均を計算する

import pandas as pd
import glob
from utils.parser import get_parser, parse_gpus, BaseConfig
import os

class Config(BaseConfig):
    def build_parser(self):
        parser = get_parser("")
        # ======== model settings ============
        parser.add_argument('--path', default="", help='path')
      
        return parser
    
    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))


def main(config):
    # ディレクトリ内のすべてのCSVファイルのパスを取得
    csv_files = glob.glob(f"{config.path}/*.csv")

    # 各CSVを読み込んで1つのデータフレームに統合
    dataframes = []

    columns = ['column1', 'column2', 'column3']  # 必要な列名を指定
    df = pd.DataFrame(columns=[])

    for file in csv_files:
        file_df = pd.read_csv(file)
        filtered_df = file_df[file_df['Step'] < 150]
        filtered_df = filtered_df['Value']
        df['{}'.format(file)] = filtered_df

    # 各列の平均を計算（NaNを除外）
    averages = df.mean()

    # 計算結果を新しい行として追加
    df['row_average'] = df.mean(axis=1)
    # # 結果を確認
    column_lengths = df.apply(len)

    print(column_lengths)

    df.to_csv(os.path.join(config.path, "stat.csv"), index=False)

    



if __name__ == "__main__":
    config = Config()
    main(config)
