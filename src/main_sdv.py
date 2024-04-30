from sdv.datasets.local import load_csvs
from sdv.lite import SingleTablePreset
from sdv.metadata import SingleTableMetadata
from datetime import datetime


timestamp = datetime.now().isoformat().replace(' ', 'H').replace(':','-')


# Load your own csv
datasets = load_csvs(folder_name='./training_data',
                     read_csv_parameters={
                         'skipinitialspace': True,
                         'encoding': 'utf_8'
                     })

data = datasets['portfolio_rebalancing_data']

# Auto detect metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)

# train the model
synthesizer = SingleTablePreset(metadata, name='FAST_ML')
synthesizer.fit(data=data)

# Generate 1000 new rows of data
synthetic_data = synthesizer.sample(num_rows=1000)

# Ensure cumulative sum per ticker is positive
tickers = synthetic_data['Ticker'].unique()
for ticker in tickers:
    mask = synthetic_data['Ticker'] == ticker
    synthetic_data.loc[mask, 'NetTradeAmount'] = synthetic_data.loc[mask, 'NetTradeAmount'].clip(lower=0)
    synthetic_data.loc[mask, 'NetTradeAmount'] += abs(synthetic_data[mask]['NetTradeAmount'].min())


# Save the synthetic data to a new CSV file
print(f'Saving file as synthetic_data_{timestamp}.csv')
synthetic_data.to_csv(f'synthetic_data_{timestamp}.csv', index=False)
print(synthetic_data)