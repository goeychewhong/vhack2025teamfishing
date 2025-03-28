import time
import pytz
import requests
import pandas as pd
from datetime import datetime

endpoints = [
    # "cryptoquant|1h|btc/market-data/coinbase-premium-index?window=hour", 
    # "cryptoquant|1h|btc/exchange-flows/inflow?exchange=binance&window=hour&",
    # "cryptoquant|1h|btc/exchange-flows/outflow?exchange=binance&window=hour&", 
    # "cryptoquant|1h|btc/market-data/taker-buy-sell-stats?window=hour&exchange=binance",
    # "cryptoquant|1h|btc/market-data/price-ohlcv?window=hour",
    # "cryptoquant|1h|btc/flow-indicator/exchange-whale-ratio?exchange=binance&window=hour"
]

API_URL = "https://api.datasource.cybotrade.rs/"

start_time = int(
    datetime(year=2021, month=1, day=1, tzinfo=pytz.timezone("UTC")).timestamp() * 1000
)
current_quota = 0
reset_time = 0
all_quota = 10000
for topic in endpoints:
    count = 1
    try:
        print(
            f"all_quota: {all_quota}, current_quota : {current_quota}, reset_time: {reset_time}"
        )
        if all_quota - current_quota <= 0:
            time.sleep(reset_time / 1000)
            print(f"Sleep for {reset_time}")
        provider = topic.split("|")[0]
        endpoint = topic.split("|")[-1]
        url = f"{API_URL}/{provider}/{endpoint}&start_time={start_time}&limit=50000"
        print(f"--------------------------------")
        print(f"{url}")
        response = requests.get(
            url,
            headers={"X-API-KEY": "..."},
        )
        print(response.reason)
        print(response.status_code)
        print(response.text)
        all_quota = int(response.headers["X-Api-Limit"])
        current_quota = int(response.headers["X-Api-Limit-Remaining"])
        reset_time = int(response.headers["X-Api-Limit-Reset-Timestamp"])
        print(
            f"all_quota: {all_quota}, current_quota : {current_quota}, reset_time: {reset_time}"
        )
        data = response.json()["data"]
        df = pd.DataFrame(data)
        print(f"Done fetch {topic}")
        print(df)
        #Save the result to a CSV file
        df.to_csv(f"{count}_{provider}.csv", index=False)
        count += 1
    except Exception as e:
        print(response.status_code)
        print(f"Failed to fetch {topic} : {e}")
