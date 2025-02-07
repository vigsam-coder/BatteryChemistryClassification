import argparse
import ast
import pandas as pd
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
from habanero import Crossref
from doi2bib.crossref import get_bib

class BatteryDataProcessor:
    def __init__(self, csv_path, debug=False):
        self.csv_path = csv_path
        self.debug = debug
        self.df = None
        self.unique_elements = None

    def load_and_process_data(self):
        """Load CSV and process battery data into a structured format."""
        print("[INFO] Loading dataset...")
        self.df = pd.read_csv(self.csv_path)

        # Convert string representation of lists to actual lists
        self.df['Extracted_name'] = self.df['Extracted_name'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        # Extract unique elements from chemical composition
        self.unique_elements = {
            element for entry in self.df['Extracted_name']
            for sub_dict in entry
            for element in sub_dict.keys()
        }
        self.unique_elements = list(self.unique_elements)  # Convert to list for indexing

        # Apply function to create multi-labels
        self.df['Chemical_labels'] = self.df['Extracted_name'].apply(self.get_binary_labels)

        # Convert labels to DataFrame
        df_binary = pd.DataFrame(self.df['Chemical_labels'].tolist(), columns=self.unique_elements)

        # Filter columns where sum is greater than 1000
        filtered_columns = df_binary.columns[df_binary.sum(axis=0) > 1000]

        # Concatenate DOI column with the filtered binary labels
        self.df = pd.concat([self.df[['DOI']], df_binary[filtered_columns]], axis=1)
        print("[INFO] Data processing completed.")

    def get_binary_labels(self, compound):
        """Convert chemical composition to binary labels."""
        return [1 if element in {e for sub_dict in compound for e in sub_dict.keys()} else 0
                for element in self.unique_elements]

    @staticmethod
    def get_abstract_from_doi(doi):
        """Fetch abstract from DOI using Crossref."""
        cr = Crossref()
        result = cr.works(ids=doi)
        if "abstract" in result["message"]:
            abstract = result["message"]["abstract"]
            soup = BeautifulSoup(abstract, "html.parser")
            return soup.get_text()
        return None

    @staticmethod
    def get_title_from_doi(doi):
        """Fetch title from DOI using doi2bib."""
        _, bibtex_entry = get_bib(doi)
        match = re.search(r"title=\{(.+?)\}", bibtex_entry, re.IGNORECASE)
        return match.group(1) if match else None

    def enrich_and_filter_text(self):
        """Fetch title and abstract for each DOI and filter out missing data."""
        print("[INFO] Enriching dataset with titles and abstracts...")

        if self.debug:
            print("[DEBUG] Running in debug mode. Processing only the first 100 entries.")
            self.df = self.df.head(1000)

        self.df['text'] = None  # Initialize the column

        for idx, doi in tqdm(enumerate(self.df['DOI']), total=len(self.df), desc="Processing DOIs"):
            title = self.get_title_from_doi(doi)
            try:
                abstract = self.get_abstract_from_doi(doi)
            except:
                abstract = None

            if abstract:  # Append only if abstract exists
                self.df.at[idx, 'text'] = f"{title} {abstract}"

        # Remove rows where text is None and reset index
        self.df = self.df.dropna(subset=['text']).reset_index(drop=True)
        print("[INFO] Enrichment completed.")

    def save_to_csv(self, output_path):
        """Save the processed DataFrame to a CSV file."""
        self.df.to_csv(output_path, index=False)
        print(f"[INFO] Processed data saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Battery Data Processing Script")
    parser.add_argument("csv_path", help="Path to the input CSV file")
    parser.add_argument("output_path", help="Path to save the processed CSV file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (process first 100 rows only)")

    args = parser.parse_args()

    processor = BatteryDataProcessor(args.csv_path, debug=args.debug)
    processor.load_and_process_data()
    processor.enrich_and_filter_text()
    processor.save_to_csv(args.output_path)

if __name__ == "__main__":
    main()


