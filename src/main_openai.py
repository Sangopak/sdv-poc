import json
from openai import OpenAI
import pandas as pd

# Set up OpenAI API key
my_assigned_team = 'team_13'
keys = json.load(open('C:/shared/content/config/api-keys/hackathon_openai_keys.json'))
my_key = keys[my_assigned_team]

client = OpenAI(api_key=my_key)

# Convert the data to a Pandas DataFrame
transactions_df = pd.read_csv('portfolio_rebalancing_data.csv')

# Convert DataFrame to a formatted string
transactions_str = transactions_df.to_string(index=False)

prompt = "Given the portfolio transaction history:\n" + transactions_str + "\nDertmine the target portfolio allocation and analyze the rebalance trends from this data."

response = client.completions.create(
  model="gpt-3.5-turbo-instruct",
  prompt=prompt,
  max_tokens=1024,
  temperature=0.9
)

print(response.choices[0].text)