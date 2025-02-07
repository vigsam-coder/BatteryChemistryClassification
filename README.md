# 🔋 BatteryChemistryClassification  

**BatteryChemistryClassification** is an AI-powered project that classifies battery research papers into different chemistries using **Large Language Models (LLMs)**. By automating this process, researchers can efficiently extract valuable insights from the vast amount of literature in battery technology.  

## 📌 Table of Contents  
- [Introduction](#introduction)  
- [Features](#features)  
- [Models Used](#models-used)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Dataset](#dataset)  
- [Results](#results)  
- [Future Work](#future-work)  
- [Contributing](#contributing)  
- [License](#license)  
- [Acknowledgments](#acknowledgments)  

## 🚀 Introduction  
BatteryChemistryClassification employs **state-of-the-art transformer models** to categorize research papers based on battery chemistries. It enhances literature analysis by providing an **automated and scalable** approach to text classification.  

## ✨ Features  
- **LLM-based Text Classification**: Categorizes battery research papers into different chemistries.  
- **Pretrained Transformer Models**: Fine-tuned **BERT, DeBERTa, RoBERTa, LongFormer** models.  
- **Scalable & Customizable**: Can be further fine-tuned for related downstream tasks.  
- **Automated Literature Processing**: Assists in quickly identifying relevant studies.  

## 🏗️ Models Used  
The following transformer models have been trained and evaluated for text classification:  
- **BERT** (Bidirectional Encoder Representations from Transformers)  
- **DeBERTa** (Decoding-enhanced BERT with Disentangled Attention)  
- **RoBERTa** (Robustly Optimized BERT Pretraining Approach)  
- **LongFormer** (Efficient Transformer for Long Documents)  

## 🛠️ Installation  
1. Create a virtual environment (optional but recommended):
    ```bash
   python3 -m venv env  
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   
2. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/BatteryChemistryClassification.git
   cd BatteryChemistryClassification
   pip install .


3. Download and scrap scientific data
    ```bash
   python BatteryChemistryClassification/battery_data.py Dataset/battery.csv Dataset/df_processed.csv --debug

4. Change the model configurations in
    ```bash
    BatteryChemistryClassification/config/config.yaml
   
5. Train the LLM model
   ```bash
    python BatteryChemistryClassification/training.py

