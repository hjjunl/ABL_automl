from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sqlalchemy import text
import pandas as pd
from pydantic import BaseSettings
import numpy as np
import os
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pymysql

pymysql.install_as_MySQLdb()
today = datetime.today().strftime("%Y%m%d")


class Settings(BaseSettings):
    USERNAME: str
    PASSWORD: str
    HOST: str
    PORT: str
    DBNAME: str
    QUERY: str

    class Config:
        env_file = 'db_config.env'


settings = Settings()

# data 받고 train, test로 split
def connect_db(data):
    print("======= Start connectDB_dict =======")
    engine = create_engine('mysql+mysqldb://%s:%s@%s:%s/%s' % (settings.USERNAME, settings.PASSWORD, settings.HOST, \
                                                               settings.PORT, settings.DBNAME))
    sql = text(open(os.path.join(f'{os.getcwd()}/query/', settings.QUERY)).read())
    result = engine.execute(sql)
    dataset = result.fetchall()
    df = pd.DataFrame(dataset)
    df.columns = result.keys()

    # 기본적으로 train, test 0.2 비율로 split
    train_data, test_data = train_test_split(data, test_size=float(0.2),
                                             random_state=42)
    train_data.to_csv(f'{os.getcwd()}/data/raw/{today}/train.csv', index=False)
    test_data.to_csv(f'{os.getcwd()}/data/raw/{today}/test.csv', index=False)


def main(data):
    connect_db(data)


def parse_args():
    parser = argparse.ArgumentParser(description="Save dataframe from DB")
    parser.add_argument('-data', '--data', type=str, help='Data set (defult: data.csv)', default='train.csv',
                        required=False)

    args = parser.parse_args()
    return args.data

# 재학습 시 새 파일 생성
def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


if __name__ == '__main__':

    # 처음 실행시 미리 파일 생성
    directory_path_raw = f'data/raw/' + today
    directory_path_feature1 = f'outputs/' + today
    directory_path_preprocess = f'./data/preprocessed/' + today
    directory_path_feature2 = f'outputs/' + today + '/feature_output'

    createDirectory(directory_path_raw)
    createDirectory(directory_path_feature1)
    createDirectory(directory_path_preprocess)
    createDirectory(directory_path_feature2)
    main(parse_args())