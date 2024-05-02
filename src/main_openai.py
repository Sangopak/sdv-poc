import json
from openai import OpenAI
import pandas as pd
from pandas import DataFrame

TEAM_KEY='team_13'
CSV_DATA='portfolio_rebalancing_data.csv'
MODEL='gpt-3.5-turbo-instruct'

def setup_openai_client(my_assigned_team: str): 
  # Set up OpenAI API key
  #my_assigned_team = 'team_13'
  keys = json.load(open('C:/shared/content/config/api-keys/hackathon_openai_keys.json'))
  my_key = keys[my_assigned_team]
  client = OpenAI(api_key=my_key)
  return client

def read_training_data(csv_name: str):
  # Convert the data to a Pandas DataFrame
  transactions_df = pd.read_csv(csv_name)
  return transactions_df

def convert_dataframe_to_string(transactions_df: DataFrame):
  # Convert DataFrame to a formatted string
  transactions_str = transactions_df.to_string(index=False)
  return transactions_str

def create_prompt(transactions_str: str):
 return "Given the portfolio transaction history:\n" + transactions_str + "\nDertmine the target portfolio allocation and analyze the rebalance trends from this data."

def call_openei_and_send_response(client, model, prompt):
  # OpenAI with given prompt and model
  response = client.completions.create(
    model=model,
    prompt=prompt,
    max_tokens=1024,
    temperature=0.9
  )
  return response.choices[0].text

def handler(team, csv_data, model):
  # Create the openai client
  openai_client = setup_openai_client(team)
  # Read CSV data
  transaction_dataframe = read_training_data(csv_data)
  # Convert dataframe to string
  transactions_str = convert_dataframe_to_string(transaction_dataframe)
  # Create prompt
  prompt = create_prompt(transactions_str)
  # Call OpenAi to get analysis
  print(f"Calling OpenAI for analysis for input {csv_data} with model {model}")
  print(call_openei_and_send_response(openai_client, model, prompt))

# Calling Handler
handler(TEAM_KEY, CSV_DATA, MODEL)