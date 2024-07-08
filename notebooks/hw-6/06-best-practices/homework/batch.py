#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os
import boto3
import io
from urllib.parse import urlparse

ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', None)

def prepare_data(df: pd.DataFrame, categorical) -> pd.DataFrame:
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

# def read_data(filename: str) -> pd.DataFrame:
#     if ENDPOINT_URL is None:
#         options = {}
#     else:
#         options = {
#             'client_kwargs': {
#                 'endpoint_url': ENDPOINT_URL
#             }
#         }

#     return pd.read_parquet(filename)

def read_data(filename: str) -> pd.DataFrame:
    print(f"Reading data from {filename}")
    
    s3_client = boto3.client(
        's3',
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1'
    )
    
    # Parse the S3 URI
    parts = filename.replace("s3://", "").split("/")
    bucket_name = parts[0]
    key = "/".join(parts[1:])
    
    print(f"Accessing bucket: {bucket_name}, key: {key}")  # Debugging line
    
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        parquet_data = response['Body'].read()
        return pd.read_parquet(io.BytesIO(parquet_data))
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        raise


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
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    input_pattern.format(year=year, month=month)
    print(f"Input file pattern: {input_pattern.format(year=year, month=month)}")  # Debugging line
    print("--------------------------------")  # Debugging line

    return input_pattern.format(year=year, month=month)


def get_output_path(year: int, month: int) -> str:
    default_output_pattern = 's3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def main(year: int, month: int):
    input_file = f's3://nyc-duration/in/{year:04d}-{month:02d}.parquet'
    output_file = f's3://nyc-duration/out/{year:04d}-{month:02d}.parquet'

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    categorical = ['PULocationID', 'DOLocationID']

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
