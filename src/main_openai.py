import json
import pandas as pd
from openai import OpenAI

my_assigned_team = 'team_13'
keys = json.load(open('C:/shared/content/config/api-keys/hackathon_openai_keys.json'))
my_key = keys[my_assigned_team]


client = OpenAI(api_key=my_key)

# Example stock transaction history as a Pandas DataFrame
transactions_data = {
    'Date': ['2023-01-01', '2023-01-05', '2023-02-01', '2023-02-15'],
    'Action': ['Buy', 'Sell', 'Buy', 'Sell'],
    'Symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
    'Quantity': [100, 50, 75, 25],
    'Price': [150, 160, 170, 180]
}

# Convert the data to a Pandas DataFrame
transactions_df = pd.DataFrame(transactions_data)

# Convert DataFrame to a formatted string
transactions_str = transactions_df.to_string(index=False)

prompt = "The following are stock transaction history:\n" + transactions_str + "\n\nAnalyze the trends and insights from this data."
stream = client.chat.completions.create(
  model="gpt-3.5-turbo-instruct",
  messages=[
    {"role": "user", "content": prompt}
  ],
  stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")