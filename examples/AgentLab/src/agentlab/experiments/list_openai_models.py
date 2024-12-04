import pandas as pd
from openai import OpenAI

if __name__ == "__main__":
    models = OpenAI().models.list()
    df = pd.DataFrame([dict(model) for model in models.data])

    # Filter GPT models or o1 models
    df = df[df["id"].str.contains("gpt") | df["id"].str.contains("o1")]

    # Convert Unix timestamps to dates (YYYY-MM-DD) and remove time
    df["created"] = pd.to_datetime(df["created"], unit="s").dt.date
    df.sort_values(by="created", inplace=True)
    # Print all entries
    print(df)
