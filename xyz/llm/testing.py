import pandas as pd
df = pd.DataFrame({
    'timestamp': [1717000000, 1717003600, 1717007200],
    'close': [100, 101, 102]
})
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df.set_index('datetime', inplace=True)
print(df.resample('h').agg({'close': 'last'}))
