#!/usr/bin/env bash

cd "$(dirname "$0")"

export INPUT_FILE_PATTERN="s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
export OUTPUT_FILE_PATTERN="s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
export S3_ENDPOINT_URL="http://localhost:4566"

docker compose up -d

sleep 3

aws --endpoint-url=${S3_ENDPOINT_URL} \
    s3 mb s3://nyc-duration
pip run python integration_test.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker compose logs
    docker compose down
    exit ${ERROR_CODE}
fi

cd ../
pip run python batch.py 2023 1

aws --endpoint-url=${S3_ENDPOINT_URL} \
    s3 ls --recursive s3://nyc-duration

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker compose logs
    docker compose down
    exit ${ERROR_CODE}
fi

cd - > /dev/null
docker compose down
