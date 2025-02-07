from doi2bib.crossref import get_bib
import re
from bs4 import BeautifulSoup
from habanero import Crossref
import pandas as pd
import ast

def process_battery_data(csv_path):
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Convert string representation of lists to actual lists
    df['Extracted_name'] = df['Extracted_name'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Extract unique elements
    unique_elements = {element for entry in df['Extracted_name'] for sub_dict in entry for element in sub_dict.keys()}
    unique_elements = list(unique_elements)  # Convert to list for indexing

    # Function to convert chemical composition to binary labels
    def get_binary_labels(compound):
        labels = [1 if element in {e for sub_dict in compound for e in sub_dict.keys()} else 0 for element in unique_elements]
        return labels

    # Apply function to create multi-labels
    df['Chemical_labels'] = df['Extracted_name'].apply(get_binary_labels)

    # Convert labels to DataFrame
    df_binary = pd.DataFrame(df['Chemical_labels'].tolist(), columns=unique_elements)

    # Filter columns where sum is greater than 1000
    filtered_columns = df_binary.columns[df_binary.sum(axis=0) > 1000]

    # Concatenate DOI column with the filtered binary labels
    df_final = pd.concat([df[['DOI']], df_binary[filtered_columns]], axis=1)

    return df_final

csv_path = './Dataset/battery.csv'
df_processed = process_battery_data(csv_path)

print(df_processed.head(10))

def get_abstract_from_doi(doi):
    cr = Crossref()
    result = cr.works(ids=doi)
    if "abstract" in result["message"]:
        abstract = result["message"]["abstract"]
        soup = BeautifulSoup(abstract, "html.parser")
        return soup.get_text()
    else:
        return None

def get_title_from_doi(doi):
    _,bibtex_entry = get_bib(doi)
    match = re.search(r"title=\{(.+?)\}", bibtex_entry, re.IGNORECASE)
    return match.group(1) if match else None

# df_processed = df_processed.head(100)
# df_processed['text'] = None
#
# for doi in df_processed['DOI']:
#     title = get_title_from_doi(doi)
#     abstract = get_abstract_from_doi(doi)
#     if abstract:
#         print(title, ' ', abstract)


def enrich_and_filter_text(df):
    df['text'] = None  # Initialize the column

    for idx, doi in enumerate(df['DOI']):
        title = get_title_from_doi(doi)
        abstract = get_abstract_from_doi(doi)

        if abstract:  # Append only if abstract exists
            df.at[idx, 'text'] = f"{title} {abstract}"

    # Remove rows where text is None and reset index
    df = df.dropna(subset=['text']).reset_index(drop=True)

    return df


# Usage:
df_processed = df_processed.head(100)
df_processed = enrich_and_filter_text(df_processed)

df_processed.to_csv('df_processed.csv',index=False)



