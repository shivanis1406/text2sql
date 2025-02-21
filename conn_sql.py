import streamlit as st
import pyodbc
import pandas as pd

# Database connection settings - using the exact working configuration
DB_CONFIG = {
    "server": "34.242.202.25",
    "username": "SA",
    "password": "LuckySingh@123",
    "driver": "{ODBC Driver 17 for SQL Server}",
    "database": "LAMARQUISE"
}

def create_sql_connection() -> pyodbc.Connection:
    """
    Create a connection to SQL Server using the working configuration
    """
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

def main():
    st.title("SQL Server Data Viewer")
    
    # Connect button
    if st.button("Connect to Database"):
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
                    st.write(F"Connecting to {db_name}")
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
                        st.success(f"""Connected to table {DB_CONFIG["database"]}""")
                        db_cursor = db_conn.cursor()
                        st.session_state['conn'] = db_conn

                    ## Fetch table names for the current database
                        db_cursor.execute("SELECT name FROM sys.tables")
                        tables = [table[0] for table in db_cursor.fetchall()]
                    else:
                        st.error(f"""Could not connect to table {DB_CONFIG["database"]}""")
                #tables = [table.table_name for table in cursor.tables(tableType='TABLE')]
                if tables:
                    st.session_state['tables'] = tables
                    st.write(f"Found {len(tables)} tables")
                else:
                    st.warning("No tables found in the database.")
            except pyodbc.Error as e:
                st.error(f"Error fetching tables: {str(e)}")
            
    if 'conn' in st.session_state:
        # Check if tables exist in session state
        if 'tables' in st.session_state and st.session_state['tables']:
            selected_table = st.selectbox("Select Table", st.session_state['tables'])
            
            if st.button("View Data"):
                try:
                    query = f"SELECT * FROM {selected_table}"
                    #df = pd.read_sql(query, st.session_state['conn'])
                    dfs = []
                    for chunk in pd.read_sql(query, st.session_state['conn'], chunksize=10000):
                        dfs.append(chunk)
                        st.write(f"Fetched {len(chunk)} rows")

                    df = pd.concat(dfs, ignore_index=True)  # Combine all chunks into a single DataFrame
                    st.dataframe(df)
                    
                    # Add export functionality
                    if not df.empty:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download data as CSV",
                            data=csv,
                            file_name=f'{selected_table}.csv',
                            mime='text/csv',
                        )
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
        else:
            st.warning("No tables found in the database or connection lost. Please reconnect.")

if __name__ == "__main__":
    main()