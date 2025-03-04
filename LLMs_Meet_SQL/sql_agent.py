import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key and MySQL password from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
mysql_password = os.getenv("MYSQL_PASSWORD")

if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")
if not mysql_password:
    raise ValueError("Please set the MYSQL_PASSWORD environment variable")

# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = openai_api_key

# Directly using database connection details
host = "localhost"
user = "root"
password = mysql_password
database = "SalesOrderSchema"

# Setup database connection
db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}/{database}"
db = SQLDatabase.from_uri(db_uri)
llm = ChatOpenAI(model="gpt-4", temperature=0)
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# Streamlit app layout
st.title('SQL Chatbot')

# User input
user_query = st.text_area("Enter your SQL-related query:", "List Top 10 Employees by Salary?")

if st.button('Submit'):
    try:
        # Processing user input
        response = agent_executor.invoke({
            "agent_scratchpad": "",  # Assuming this needs to be an empty string if not used
            "input": user_query  # Changed from "query" to "input"
        })
        
        results = response['output']
        # Display the results in a table format
        st.subheader("Query Results:")

        if isinstance(results, list) and results:
            st.table(results)
        else:
            st.write(results)

    except Exception as e:
        st.error(f"An error occurred: {e}")