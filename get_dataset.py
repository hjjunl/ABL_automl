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


def connect_db(data):
    print("======= Start connectDB_dict =======")
    engine = create_engine('mysql+mysqldb://%s:%s@%s:%s/%s'%(settings.USERNAME, settings.PASSWORD, settings.HOST,\
                                                                                    settings.PORT,settings.DBNAME))
    sql = text(open(os.path.join(f'{os.getcwd()}/query/', settings.QUERY)).read())
    result = engine.execute(sql)
    dataset = result.fetchall()
    df = pd.DataFrame(dataset)
    df.columns = result.keys()

    df.to_csv(f'{os.getcwd()}/origin_data/{data}', index=False)


def main(data):
    connect_db(data)

def parse_args():
    parser = argparse.ArgumentParser(description="Save dataframe from DB")
    parser.add_argument('-data', '--data', type=str, help='Data set (defult: train.csv)', default = 'train.csv', required=False)

    args = parser.parse_args()
    return args.data

if __name__ == '__main__':
    main(parse_args())