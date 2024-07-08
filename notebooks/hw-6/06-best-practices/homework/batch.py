#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os

ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', None)

def prepare_data(df: pd.DataFrame, categorical) -> pd.DataFrame:
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def read_data(filename: str) -> pd.DataFrame:
    # if ENDPOINT_URL is None:
    #     options = {}
    # else:
    #     options = {
    #         'client_kwargs': {
    #             'endpoint_url': ENDPOINT_URL
    #         }
    #     }

    return pd.read_parquet(filename)


def save_data(df: pd.DataFrame, filename: str):
    if ENDPOINT_URL is None:
        options = {}
    else:
        options = {
            'client_kwargs': {
                'endpoint_url': ENDPOINT_URL
            }
        }

    df.to_parquet(
        filename,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )




def get_input_path(year: int, month: int) -> str:
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    return input_file.format(year=year, month=month)


def get_output_path(year: int, month: int) -> str:
    output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    return output_file.format(year=year, month=month)


def main(year: int, month: int):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    categorical = ['PUlocationID', 'DOlocationID']

    df = read_data(input_file)
    df = prepare_data(df, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print('predicted mean duration:', y_pred.mean())
    print('sum of predicted durations:', y_pred.sum())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    save_data(df_result, output_file)





if __name__ == '__main__':
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    main(year=year, month=month)
