import json
from openai import OpenAI
import pandas as pd
from pandas import DataFrame

TEAM_KEY='team_13'
CSV_DATA='portfolio_rebalancing_data.csv'
MODEL='gpt-3.5-turbo-instruct'

def setup_openai_client(my_assigned_team: str): 
    """
    Set up the OpenAI API client.

    Parameters:
    - my_assigned_team (str): The team key assigned for using the OpenAI API.

    Returns:
    - client (OpenAI): An instance of the OpenAI client.
    """
    keys = json.load(open('C:/shared/content/config/api-keys/hackathon_openai_keys.json'))
    my_key = keys[my_assigned_team]
    client = OpenAI(api_key=my_key)
    return client

def read_training_data(csv_name: str):
    """
    Read training data from a CSV file and convert it into a DataFrame.

    Parameters:
    - csv_name (str): The name of the CSV file containing the training data.

    Returns:
    - transactions_df (DataFrame): A DataFrame containing the training data.
    """
    transactions_df = pd.read_csv(csv_name)
    return transactions_df

def convert_dataframe_to_string(transactions_df: DataFrame):
    """
    Convert a DataFrame to a formatted string.

    Parameters:
    - transactions_df (DataFrame): The DataFrame to be converted.

    Returns:
    - transactions_str (str): A formatted string representation of the DataFrame.
    """
    transactions_str = transactions_df.to_string(index=False)
    return transactions_str

def create_prompt(transactions_str: str):
    """
    Create a prompt for analyzing portfolio rebalancing trends.

    Parameters:
    - transactions_str (str): A formatted string representing the portfolio transaction history.

    Returns:
    - prompt (str): The prompt for analysis.
    """
    return "Given the portfolio transaction history:\n" + transactions_str + "\nDetermine the target portfolio allocation and analyze the rebalance trends from this data."

def call_openei_and_send_response(client, model, prompt):
    """
    Call the OpenAI API with a given prompt and model.

    Parameters:
    - client (OpenAI): An instance of the OpenAI client.
    - model (str): The name of the model to use for text generation.
    - prompt (str): The prompt for text generation.

    Returns:
    - response_text (str): The generated text response from the OpenAI API.
    """
    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=1024,
        temperature=0.9
    )
    return response.choices[0].text

def handler(team, csv_data, model):
    """
    Main handler function for processing portfolio rebalancing data.

    Parameters:
    - team (str): The team key assigned for using the OpenAI API.
    - csv_data (str): The name of the CSV file containing the training data.
    - model (str): The name of the model to use for text generation.

    Returns:
    - None
    """
    # Create the OpenAI client
    openai_client = setup_openai_client(team)
    # Read CSV data
    transaction_dataframe = read_training_data(csv_data)
    # Convert dataframe to string
    transactions_str = convert_dataframe_to_string(transaction_dataframe)
    # Create prompt
    prompt = create_prompt(transactions_str)
    # Call OpenAI to get analysis
    print(f"Calling OpenAI for analysis for input {csv_data} with model {model}")
    print(call_openei_and_send_response(openai_client, model, prompt))

# Calling Handler
handler(TEAM_KEY, CSV_DATA, MODEL)
