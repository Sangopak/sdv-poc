from sdv.datasets.local import load_csvs
from sdv.lite import SingleTablePreset
from sdv.metadata import SingleTableMetadata


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

# Save the synthetic data to a new CSV file
synthetic_data.to_csv('synthetic_data.csv', index=False)
print(synthetic_data)