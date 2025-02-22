import streamlit as st
import pyodbc
import pandas as pd
from generate_sql_query import text2sql
from dotenv import load_dotenv
import os, json
from groq import Groq

load_dotenv()

# Database connection settings - using the exact working configuration
DB_CONFIG = {
    "server": os.getenv("SERVER"),
    "username": os.getenv("SQL_SERVER_USERNAME"),
    "password": os.getenv("SQL_SERVER_PASSWORD"),
    "driver": "{ODBC Driver 17 for SQL Server}",
    "database": os.getenv("DATABASE")
}

def create_sql_connection() -> pyodbc.Connection:
    """
    Create a connection to SQL Server using the working configuration
    """
    print(f"DEBUG - DB_CONFIG is {DB_CONFIG}")

    try:
        conn_str = (
            f"DRIVER={DB_CONFIG['driver']};"
            f"SERVER={DB_CONFIG['server']};"
            f"UID={DB_CONFIG['username']};"
            f"PWD={DB_CONFIG['password']}"
        )
        
        return pyodbc.connect(conn_str)
    except pyodbc.Error as e:
        st.error(f"Error connecting to database: {str(e)}")
        return None

def generate_response_text(query, data):
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))

    prompt = f"""
User's query :
{query}

Data fetched from SQL table : 
{data}

Strictly return a JSON in the format 
{{
"response" : <well-crafted response as a string only>
}}
"""
    print(f"DEBUG for response generation is {prompt}")

    response = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",  # or your preferred model
        messages=[
            {"role": "system", "content": "Craft a professional response given the user's query that was converted to SQL to fetch data from SQL Table"},
            {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for more consistent results
            max_tokens=8192,
            stream=False,
            response_format={"type": "json_object"},
            stop=None
        )
    print(f"Raw response is {response.choices[0].message.content}")
    try:
        json_response = json.loads(response.choices[0].message.content.replace("```json", "").replace("```", ""))
        return json_response
    except Exception as err:
        return {"error" : err}
                    
def main():
    st.title("SQL Server Data Viewer")

    if "query" not in st.session_state:
        st.session_state.query = None
    if "sql_query" not in st.session_state:
        st.session_state.sql_query = None
    if "tables" not in st.session_state:
        st.session_state.tables = None
    if "conn" not in st.session_state:
        st.session_state.conn = None
    
    # Connect button
    if st.session_state.conn is None and st.button("Connect to Database"):
        conn = create_sql_connection()
        
        if conn:
            st.session_state['conn'] = conn
            st.success("Connected to database successfully!")
            
            # Get list of tables
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name 
                    FROM sys.databases 
                    WHERE name NOT IN ('master', 'tempdb', 'model', 'msdb');
                """)
                databases = cursor.fetchall()
                for db in databases:
                    db_name = db[0]
                    if db_name != DB_CONFIG["database"]:
                        continue
                    #st.write(F"Connecting to {db_name}")
                    ## Connect to the specific database
                    db_conn_str = (
                        f"DRIVER={DB_CONFIG['driver']};"
                        f"SERVER={DB_CONFIG['server']};"
                        f"DATABASE={db_name};"
                        f"UID={DB_CONFIG['username']};"
                        f"PWD={DB_CONFIG['password']}"
                    )
                    db_conn = pyodbc.connect(db_conn_str)
                    if db_conn:
                        #st.success(f"""Connected to table {DB_CONFIG["database"]}""")
                        db_cursor = db_conn.cursor()
                        st.session_state['conn'] = db_conn

                    ## Fetch table names for the current database
                        db_cursor.execute("SELECT name FROM sys.tables")
                        tables = [table[0] for table in db_cursor.fetchall()]
                    else:
                        st.error(f"""Could not connect to SQL Tables""")
                #tables = [table.table_name for table in cursor.tables(tableType='TABLE')]
                if tables:
                    st.session_state['tables'] = tables
                    #st.write(f"Found {len(tables)} tables")
                else:
                    st.warning("No tables found in the database.")
            except pyodbc.Error as e:
                st.error(f"Error fetching tables: {str(e)}")
            
    if st.session_state.conn:
        # Check if tables exist in session state
        if st.session_state['tables']:
            #selected_table = st.selectbox("Select Table", st.session_state['tables'])
            selected_table = "BI_STOCK_TRANSACTION_DAILY"

            #Get user's query
            #Total number of Transactions with Zero or Negative Value
            query = st.chat_input("Enter Query")
            if query:
                st.session_state.query = query
                print(f"Query entered by user is {st.session_state.query}")

            if st.session_state.query:
                try:
                    sql_query = text2sql(st.session_state.query)
                    if sql_query == "NA":
                        #Ask query again
                        st.session_state.query = None
                        st.stop()
                    else:
                        st.session_state.sql_query = sql_query
                    df = pd.read_sql(st.session_state.sql_query, st.session_state.conn)

                    #query = f"SELECT * FROM {selected_table}"
                    #dfs = []
                    #for chunk in pd.read_sql(query, st.session_state['conn'], chunksize=10000):
                    #    dfs.append(chunk)
                    #    st.write(f"Fetched {len(chunk)} rows")

                    #df = pd.concat(dfs, ignore_index=True)  # Combine all chunks into a single DataFrame
                    #st.dataframe(df)
                    
                    #Display value using deepseek
                    data = df.to_string()

                    response = generate_response_text(st.session_state.query, data)
                    if "error" not in response.keys():
                        st.write(response["response"])
                    else:
                        st.error(f"""Error generating response : {response["error"]}""")
                    # Add export functionality
                    #if not df.empty:
                    #    csv = df.to_csv(index=False)
                    #    st.download_button(
                    #        label="Download data as CSV",
                    #        data=csv,
                    #        file_name=f'{selected_table}.csv',
                    #        mime='text/csv',
                    #    )
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
        else:
            st.warning("No relevant tables found in the database or connection lost. Please reconnect.")

if __name__ == "__main__":
    main()
