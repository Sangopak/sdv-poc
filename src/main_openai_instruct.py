import json
from openai import OpenAI
import pandas as pd

# Set up OpenAI API key
my_assigned_team = 'team_13'
keys = json.load(open('C:/shared/content/config/api-keys/hackathon_openai_keys.json'))
my_key = keys[my_assigned_team]

client = OpenAI(api_key=my_key)

# # Example stock transaction history as a Pandas DataFrame
# transactions_data = {
#     'Date': ['2023-01-01', '2023-01-05', '2023-02-01', '2023-02-15'],
#     'Action': ['Buy', 'Sell', 'Buy', 'Sell'],
#     'Symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
#     'Quantity': [100, 50, 75, 25],
#     'Price': [150, 160, 170, 180]
# }

# Convert the data to a Pandas DataFrame
transactions_df = pd.read_csv('portfolio_rebalancing_data.csv')

# Convert DataFrame to a formatted string
transactions_str = transactions_df.to_string(index=False)
allocation_goal = '60% Stock, 30% Bond and 10% Cash'

prompt = "Given the portfolio allocation goal: "+ allocation_goal +" and transaction history:\n" + transactions_str + "\nAnalyze the trends for portfolio rebalancing from this data."

response = client.completions.create(
  model="gpt-3.5-turbo-instruct",
  prompt=prompt,
  max_tokens=1024,
  temperature=0.9
)

print(response.choices[0].text)