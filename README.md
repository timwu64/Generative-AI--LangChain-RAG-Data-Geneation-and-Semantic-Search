# HomeMatch Project Overview

## Introduction

"HomeMatch" is an innovative application aimed at revolutionizing the real estate industry by offering personalized property search experiences to potential buyers. This application utilizes Large Language Models (LLMs) and vector databases to convert standard real estate listings into personalized narratives, catering to the unique preferences and needs of each buyer.

## Project Goals

The primary goal of "HomeMatch" is to create a seamless, engaging, and tailored property search experience by:

- **Understanding Buyer Preferences**: Capturing detailed buyer preferences in natural language, including location, budget, property type, amenities, and lifestyle choices.
- **Integrating with Vector Databases**: Utilizing vector embeddings to match properties with buyer preferences based on various criteria.
- **Personalizing Listing Descriptions**: Generating tailored property descriptions that highlight aspects relevant to the buyer’s preferences without altering factual information.
- **Presenting Listings**: Showcasing personalized listings to potential buyers in a compelling and informative manner.

## Core Components

- **LLM for Narrative Generation**: Generate property descriptions to individual buyer preferences.
- **Vector Database Integration**: Efficiently matches listings with buyer preferences using semantic embeddings.
- **User Preference Interface**: Captures and processes buyer preferences in a structured manner.

## Functionality

The core functionality of "HomeMatch" includes:

- **Understanding Buyer Preferences**: Interpreting buyers' requirements and preferences in natural language, including to location, property type, budget, and amenities.
- **Vector Database Integration**: Matching properties with buyer preferences by connecting with a vector database where all available property listings are stored.
- **Personalized Listing Descriptions**: Generating customized property descriptions that highlight features relevant to the buyer's preferences, ensuring all information remains factual.
- **Presentation of Listings**: Displaying personalized listings in an engaging and informative manner.

## Setup Instructions

### Step 1: Python Project Initialization

1. **Create a new Python project**, setting up a virtual environment.
2. **Install necessary packages**. Essential packages include LangChain, an LLM library such as OpenAI's GPT, and a vector database package (e.g., ChromaDB).

Installation commands:
```bash
pip install langchain==0.0.305
pip install openai==0.28.1
pip install chromadb==0.4.15
pip install tiktoken
pip install langchain_core
```

### Step 2: Project File Structure

If working within Google Colab, ensure you mount your Google Drive for easy access to project files:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Define your project file path accordingly to maintain an organized workspace.

### Step 3: Prepare the API Key File

Ensure you have an API key from OpenAI. If you do not have one, you can obtain it by creating an account on the OpenAI platform and accessing the API section. Create a text file named api_key.txt in your project directory. Open the file and paste your OpenAI API key inside it. Save and close the file then run the code below.

```python
# Path to the file containing the API key
api_key_file = 'api_key.txt'

# Function to read the API key from the file
def get_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()  # .strip() removes any leading/trailing whitespace

# Retrieve the API key from the file
openai.api_key = get_api_key(api_key_file)
# display(openai.api_key)
```

## Running Instructions

1. **Import Required Libraries**: Ensure all necessary libraries are imported at the beginning of your script. Key imports include `openai`, `langchain`, and `pandas`.

2. **Initialize Components**: Set up the LLMChain, vector store, and document loaders. Example initialization:

   ```python
   import openai
   from langchain import LLMChain
   from langchain.llms import OpenAI
   from langchain_core.prompts import ChatPromptTemplate
   from langchain.prompts import PromptTemplate
   from langchain.docstore.document import Document
   from langchain.document_loaders.csv_loader import CSVLoader
   from langchain.document_loaders import DataFrameLoader
   from langchain.embeddings.openai import OpenAIEmbeddings
   from langchain.text_splitter import CharacterTextSplitter
   from langchain.vectorstores import Chroma
   from langchain.chains import RetrievalQA
   from langchain.chains.question_answering import load_qa_chain
   from langchain.output_parsers import StructuredOutputParser, ResponseSchema
   import pandas as pd
   from tqdm import tqdm
   import random

   print(openai.__version__)
   ```

3. **Generate Real Estate Listings**: Use the LLM to create property listings that will be stored in the vector database for matching with buyer preferences.
   ```python
   Generating Listings: 100%|██████████| 100/100 [16:10<00:00,  9.71s/it]
   ```
5. **Implement Buyer Preference Interface**: Collect and process buyer preferences to structure these for querying against the vector database.
   
6. **Store Listings in Vector Database**: Initialize and populate your vector database with the generated listings and their embeddings.
   ```python
    loader = DataFrameLoader(data_sort, page_content_column="Langchain_page_content")
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    split_docs = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    db = Chroma.from_documents(split_docs, embeddings)
    retriever=db.as_retriever()
    ```

7. **Perform Listing Matching and Personalization**: Use semantic search to find listings matching the buyer's preferences and personalize the descriptions using LLMs.
   ```python
    def collect_buyer_preferences():
    questions = [
        "What is the range of your budget for buying a house? (e.g., 700000-3000000)",
        "What size range are you looking for in square feet for your house? (e.g., 1000-2500)",
        "What is the minimum number of Bedrooms you are looking for? (e.g., 3)",
        "What is the minimum number of Bathrooms you are looking for? (e.g., 3)",
        "What are 3 most important things for you in choosing this property? (e.g., A quiet neighborhood, good local schools, and convenient shopping options.)",
        "Which amenities would you like? (e.g., A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.)",
        "Which transportation options are important to you? (e.g., Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.)",
        "Do you have any other preferences? (e.g., A balance between suburban tranquility and urban amenities.)",
    ]

    # Initialize an empty list to hold each row (question and answer)
    data = []

    # Iterate through each question, collecting the answer
    for question in questions:
        print(question)  # Display the question
        answer = input()  # Collect the user's answer
        data.append({"Question": question, "Answer": answer})  # Append the question and answer as a dict

    # Convert the list of dicts into a DataFrame
    answers_df = pd.DataFrame(data)

    # Return the DataFrame containing the answers
    return answers_df

# Collect preferences from the user and store them in a DataFrame
buyer_preferences_df = collect_buyer_preferences()
```

8. **Display Matched Listings**: Present the personalized listings to the user.
```python
# Load the QA chain
chain = load_qa_chain(llm, prompt=prompt, chain_type="stuff")

# Run the chain with the extracted Document objects and the query
results = chain.run(input_documents=similar_docs, query=query)

print(results)
```
## Additional Notes

- **Version Control**: Keep track of the `openai` library version and other dependencies to ensure compatibility.
- **Data Storage**: Adjust the `project_file_path` as necessary, based on your working environment and storage preferences.

## Conclusion

"HomeMatch" is designed to enhance the property search experience by offering personalized and engaging real estate listings. Follow the setup and running instructions to deploy this innovative solution and contribute to transforming the real estate industry.

## Environment and Dependencies

- Python
- OpenAI API GPT-3.5 (openai==0.28.1)
- LangChain (langchain==0.0.305)
- Vector Database (chromadb==0.4.15)
- tiktoken
- langchain_core 

## Checklist

- `HomeMatch.ipynb`: Application code.
- `real_estate_listings.csv`: At least 10 generated real estate listings for testing and development.
- `api_key.txt`: Dummy api_key, please replace by your own api key.
