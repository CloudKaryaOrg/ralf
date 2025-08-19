from ralf import Ralf
import pandas as pd

print("RALF instance getting initialized.")

hfToken = input("Please enter HuggingFace Token: ")

my_ralf = Ralf(HF_TOKEN=hfToken, GEMINI_API_KEY='abc456')
print("RALF instance created successfully.")

data = [
    ["If there are 3 apples and you take away 1, how many apples are left?", "4"],
    ["What has no voice but can still speak to you?", "4"],
    ["question3", "1"],
    ["question4", "1"],
    ["question6", "1"],
    ["question7", "1"],
    ["question8", "4"],
    ["question9", "4"],
    ["questionA", "4"],
    ["questionB", "4"],
    ["questionC", "1"],
    ["questionD", "1"],
    ["questionE", "1"],
    ["questionF", "1"],
    ["questionG", "4"],
    ["questionH", "4"],
    ["questionI", "4"],
    ["questionJ", "4"],
]
df = pd.DataFrame(data, columns=["source", "target"])

my_ralf.load_and_process_data(df, text_column='source',
                            label_column='target',
                            model_name='bert-base-uncased')
print("RALF load and process data successfully.")

my_ralf.load_and_configure_model()
print("RALF load and configure model successfully.")

print("End Test")
