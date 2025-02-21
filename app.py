import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import json
#from openai import OpenAI
from groq import Groq
import os
from enum import Enum
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

class QueryError(Exception):
    """Custom exception for query processing errors"""
    pass

class ColumnType(Enum):
    """Enum for column types to help with query validation"""
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    BOOLEAN = "boolean"

@dataclass
class ColumnInfo:
    """Data class for storing column information"""
    name: str
    description: str
    column_type: ColumnType
    is_mandatory: bool = False
    example_values: List[str] = None

class NLToSQLConverter:
    def __init__(self, api_key: str = None):
        # Initialize OpenAI client
        self.client = Groq(api_key=api_key or os.getenv('GROQ_API_KEY'))
        #self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        '''
        
        # Initialize column information with detailed metadata
        self.columns_info = {
    'UNIQUEID': ColumnInfo(
        name='UNIQUEID',
        description='Unique identifier for the stock transaction record',
        column_type=ColumnType.NUMBER,
        is_mandatory=True
    ),
    'BRANCH_CODE': ColumnInfo(
        name='BRANCH_CODE',
        description='Code representing the branch where the transaction took place',
        column_type=ColumnType.TEXT
    ),
    'REFMID': ColumnInfo(
        name='REFMID',
        description='Reference ID for the transaction',
        column_type=ColumnType.TEXT
    ),
    'DIVISION_CODE': ColumnInfo(
        name='DIVISION_CODE',
        description='Code representing the division within the organization',
        column_type=ColumnType.TEXT
    ),
    'SUPPLIER': ColumnInfo(
        name='SUPPLIER',
        description='Supplier involved in the transaction',
        column_type=ColumnType.TEXT
    ),
    'VOCTYPE': ColumnInfo(
        name='VOCTYPE',
        description='Type of stock activity/transaction (e.g., SALE, PURCHASE, RETURN)',
        column_type=ColumnType.TEXT,
        example_values=['SALE', 'PURCHASE', 'RETURN']
    ),
    'VOCNO': ColumnInfo(
        name='VOCNO',
        description='Voucher number associated with the transaction',
        column_type=ColumnType.TEXT
    ),
    'SRNO': ColumnInfo(
        name='SRNO',
        description='Serial number of the transaction entry',
        column_type=ColumnType.NUMBER
    ),
    'YEARMONTH': ColumnInfo(
        name='YEARMONTH',
        description='Year and month of the transaction in YYYYMM format',
        column_type=ColumnType.TEXT,
        example_values=['202401', '202402']
    ),
    'VOCDATE': ColumnInfo(
        name='VOCDATE',
        description='Date when the stock transaction occurred',
        column_type=ColumnType.DATE,
        is_mandatory=True
    ),
    'VALUE_DATE': ColumnInfo(
        name='VALUE_DATE',
        description='Due date for payment',
        column_type=ColumnType.DATE
    ),
    'STOCK_CODE': ColumnInfo(
        name='STOCK_CODE',
        description='Code identifying the stock item',
        column_type=ColumnType.TEXT
    ),
    'PARTYCODE': ColumnInfo(
        name='PARTYCODE',
        description='Code representing the party/customer involved in the transaction',
        column_type=ColumnType.TEXT
    ),
    'ITEM_CURRENCY': ColumnInfo(
        name='ITEM_CURRENCY',
        description='Currency code in which the item is priced',
        column_type=ColumnType.TEXT,
        example_values=['USD', 'INR', 'EUR']
    ),
    'ITEM_CURR_RATE': ColumnInfo(
        name='ITEM_CURR_RATE',
        description='Exchange rate applied to the item currency',
        column_type=ColumnType.NUMBER
    ),
    'FIXED': ColumnInfo(
        name='FIXED',
        description='Indicates if the stock rate is fixed',
        column_type=ColumnType.BOOLEAN
    ),
    'PCS': ColumnInfo(
        name='PCS',
        description='Number of pieces in the stock transaction',
        column_type=ColumnType.NUMBER
    ),
    'GROSSWT': ColumnInfo(
        name='GROSSWT',
        description='Total weight of the stock including impurities',
        column_type=ColumnType.NUMBER
    ),
    'STONEWT': ColumnInfo(
        name='STONEWT',
        description='Weight of the stone in the stock item',
        column_type=ColumnType.NUMBER
    ),
    'NETWT': ColumnInfo(
        name='NETWT',
        description='Net weight of the stock item after deducting stone weight',
        column_type=ColumnType.NUMBER
    ),
    'PURITY': ColumnInfo(
        name='PURITY',
        description='Purity percentage of the stock item',
        column_type=ColumnType.NUMBER,
        example_values=[99.5, 91.6, 75.0]
    ),
    'PUREWT': ColumnInfo(
        name='PUREWT',
        description='Weight of the pure metal in the stock item',
        column_type=ColumnType.NUMBER
    ),
    'CHARGABLEWT': ColumnInfo(
        name='CHARGABLEWT',
        description='Weight used for billing calculations',
        column_type=ColumnType.NUMBER
    ),
    'MKG_RATEFC': ColumnInfo(
        name='MKG_RATEFC',
        description='Making charge rate in foreign currency',
        column_type=ColumnType.NUMBER
    ),
    'MKG_RATECC': ColumnInfo(
        name='MKG_RATECC',
        description='Making charge rate in local currency',
        column_type=ColumnType.NUMBER
    ),
    'MKGVALUEFC': ColumnInfo(
        name='MKGVALUEFC',
        description='Making charge value in foreign currency',
        column_type=ColumnType.NUMBER
    ),
    'MKGVALUECC': ColumnInfo(
        name='MKGVALUECC',
        description='Making charge value in local currency',
        column_type=ColumnType.NUMBER
    ),
    'RATE_TYPE': ColumnInfo(
        name='RATE_TYPE',
        description='Type of rate applied (e.g., FIXED, FLOATING)',
        column_type=ColumnType.TEXT,
        example_values=['FIXED', 'FLOATING']
    ),
    'METAL_RATE': ColumnInfo(
        name='METAL_RATE',
        description='Rate of the metal per unit weight',
        column_type=ColumnType.NUMBER
    ),
    'METAL_RATE_GMSFC': ColumnInfo(
        name='METAL_RATE_GMSFC',
        description='Metal rate per gram in foreign currency',
        column_type=ColumnType.NUMBER
    ),
    'METAL_RATE_GMSCC': ColumnInfo(
        name='METAL_RATE_GMSCC',
        description='Metal rate per gram in local currency',
        column_type=ColumnType.NUMBER
    ),
    'METALVALUEFC': ColumnInfo(
        name='METALVALUEFC',
        description='Value of metal in foreign currency',
        column_type=ColumnType.NUMBER
    ),
    'METALVALUECC': ColumnInfo(
        name='METALVALUECC',
        description='Value of metal in local currency',
        column_type=ColumnType.NUMBER
    )
}

        '''
        self.columns_info = {
            'VOCTYPE': ColumnInfo(
                name='VOCTYPE',
                description='Type of stock activity',
                column_type=ColumnType.TEXT
            ),
            'VOCDATE': ColumnInfo(
                name='VOCDATE',
                description='Date when the stock transaction occurred',
                column_type=ColumnType.DATE,
                is_mandatory=True
            ),
            'VALUE_DATE': ColumnInfo(
                name='VALUE_DATE',
                description='Due date for payment',
                column_type=ColumnType.DATE
            ),
            'STOCK_CODE': ColumnInfo(
                name='STOCK_CODE',
                description='Unique identifier for stock item',
                column_type=ColumnType.TEXT,
                is_mandatory=True
            ),
            'PARTYCODE': ColumnInfo(
                name='PARTYCODE',
                description='Customer or supplier unique identifier',
                column_type=ColumnType.TEXT
            ),
            'NETVALUECC': ColumnInfo(
                name='NETVALUECC',
                description='Total Transaction Amount for the respective VOCTYPE',
                column_type=ColumnType.NUMBER
            ),
            'USERNAME': ColumnInfo(
                name='USERNAME',
                description='Sales person or user who conducted the transaction',
                column_type=ColumnType.TEXT
            ),
            'SYSTEM_DATE': ColumnInfo(
                name='SYSTEM_DATE',
                description='Date and time when the entry was recorded in the ERP system',
                column_type=ColumnType.DATE
            ),
            'GROSSWT': ColumnInfo(
                name='GROSSWT',
                description='Gross weight of the items',
                column_type=ColumnType.NUMBER
            ),
            'STONEWT': ColumnInfo(
                name='STONEWT',
                description='Weight of stones in the items',
                column_type=ColumnType.NUMBER
            ),
            'NETWT': ColumnInfo(
                name='NETWT',
                description='Net weight of the items (GROSSWT - STONEWT)',
                column_type=ColumnType.NUMBER
            ),
            'PURITY': ColumnInfo(
                name='PURITY',
                description='Purity of the metal',
                column_type=ColumnType.NUMBER,
                possible_values=[0.0, 1.0]
            ),
            'METALVALUECC': ColumnInfo(
                name='METALVALUECC',
                description='Sales Amount for Metal',
                column_type=ColumnType.NUMBER
            ),
            'STONEVALUECC': ColumnInfo(
                name='STONEVALUECC',
                description='Transaction Amount for Stone',
                column_type=ColumnType.NUMBER
            )
        }
        

        # Common time-related phrases and their SQL equivalents
        self.time_patterns = {
            'today': "DATE(VOCDATE) = CURRENT_DATE",
            'yesterday': "DATE(VOCDATE) = DATE_SUB(CURRENT_DATE, INTERVAL 1 DAY)",
            'this week': "YEARWEEK(VOCDATE) = YEARWEEK(CURRENT_DATE)",
            'last week': "YEARWEEK(VOCDATE) = YEARWEEK(DATE_SUB(CURRENT_DATE, INTERVAL 1 WEEK))",
            'this month': "DATE_FORMAT(VOCDATE, '%Y-%m') = DATE_FORMAT(CURRENT_DATE, '%Y-%m')",
            'last month': "DATE_FORMAT(VOCDATE, '%Y-%m') = DATE_FORMAT(DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH), '%Y-%m')"
        }

    def generate_llm_prompt(self, user_query: str) -> str:
        """Generate an enhanced prompt for the LLM"""
        prompt = f"""As an SQL expert, analyze the following user query for a jewelry business database and provide the necessary SQL components.

Table: BI_STOCK_TRANSACTION_DAILY

Available columns and their details:
"""
        # Add detailed column information
        for col_name, col_info in self.columns_info.items():
            prompt += f"\n- {col_name}:"
            prompt += f"\n  Description: {col_info.description}"
            prompt += f"\n  Type: {col_info.column_type.value}"
            if col_info.example_values:
                prompt += f"\n  Examples: {', '.join(col_info.example_values)}"
            if col_info.is_mandatory:
                prompt += "\n  (Mandatory field)"

        prompt += f"""

User Query: {user_query}

Please analyze the query and provide:
1. Required columns and any necessary aggregations that are valid in SQL server
2. Appropriate WHERE conditions that are valid in SQL server
3. Any needed GROUP BY or ORDER BY clauses that are valid in SQL server
4. If the query is ambiguous, specify what clarification is needed
4. If the table does not have enough information to answer user's query, set is_table_info_sufficient to False else True

Response must strictly be a JSON:
{{
    "needs_clarification": boolean,
    "is_table_info_sufficient": <boolean - Can BI_STOCK_TRANSACTION_DAILY Table answer user's query?>
    "clarification_question": "string - question to ask if needed. Question must NOT contain any column's name",
    "query_understanding": "string - explanation of how you interpreted the query",
    "columns": ["list of columns"],
    "where_conditions": ["list of conditions"],
    "group_by": ["list of columns"],
    "order_by": [
        {{"column": "column name", "direction": "ASC/DESC"}}
    ],
    "aggregations": ["list of aggregation functions used"],
    "having_conditions": ["list of having conditions if needed"]
}}

Consider:
- Time periods mentioned in the query
- Any numerical comparisons or ranges
- Required aggregations for business metrics
- Proper grouping for meaningful analysis
- Sorting that would make the results most useful
- Whether multiple conditions need to be combined

If you need clarification, explain exactly what information is missing in the clarification question."""

        return prompt

    def query_llm(self, prompt: str) -> Dict:
        """Query the LLM and handle the response"""
        print(f"DEBUG - prompt is {prompt}")
        try:
            response = self.client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",  # or your preferred model
                messages=[
                    {"role": "system", "content": "You are an expert SQL query generator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for more consistent results
                max_tokens=8192,
                stream=False,
                response_format={"type": "json_object"},
                stop=None
            )
            
            # Parse the response
            print(f"DEBUG - response is {response.choices[0].message.content}")
            try:
                return json.loads(response.choices[0].message.content.replace("```json", "").replace("```", ""))
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse LLM response: {e}")
                raise QueryError("Failed to parse the query response")
                
        except Exception as e:
            self.logger.error(f"LLM query failed: {e}")
            raise QueryError(f"Failed to process query: {str(e)}")

    def validate_query_components(self, parsed_response: Dict) -> None:
        """Validate the query components returned by the LLM"""
        # Validate columns
        for column in parsed_response.get("columns", []):
            if not any(column.startswith(f"{agg}(") for agg in ["COUNT", "SUM", "AVG", "MIN", "MAX"]):
                if column not in self.columns_info:
                    raise QueryError(f"Invalid column: {column}")

        # Validate date conditions
        for condition in parsed_response.get("where_conditions", []):
            if "DATE" in condition.upper():
                try:
                    # Basic validation of date formats
                    date_pattern = r'\d{4}-\d{2}-\d{2}'
                    dates = re.findall(date_pattern, condition)
                    for date_str in dates:
                        datetime.strptime(date_str, '%Y-%m-%d')
                except ValueError as e:
                    raise QueryError(f"Invalid date format in condition: {condition}")

    def generate_sql_from_parsed_response(self, parsed_response: Dict) -> str:
        """Generate optimized SQL query from parsed LLM response"""
        try:
            # Validate components
            self.validate_query_components(parsed_response)
            
            # Build SELECT clause
            select_cols = ", ".join(parsed_response["columns"])
            sql = f"SELECT {select_cols} FROM BI_STOCK_TRANSACTION_DAILY"
            
            # Build WHERE clause
            if parsed_response.get("where_conditions"):
                conditions = " AND ".join(f"({condition})" for condition in parsed_response["where_conditions"])
                sql += f" WHERE {conditions}"
            
            # Build GROUP BY
            if parsed_response.get("group_by"):
                group_by = ", ".join(parsed_response["group_by"])
                sql += f" GROUP BY {group_by}"
            
            # Build HAVING
            if parsed_response.get("having_conditions"):
                having = " AND ".join(parsed_response["having_conditions"])
                sql += f" HAVING {having}"
            
            # Build ORDER BY
            if parsed_response.get("order_by"):
                order_clauses = []
                for order in parsed_response["order_by"]:
                    order_clauses.append(f"{order['column']} {order['direction']}")
                sql += f" ORDER BY {', '.join(order_clauses)}"
            
            return sql
            
        except Exception as e:
            self.logger.error(f"Failed to generate SQL: {e}")
            raise QueryError(f"Failed to generate SQL query: {str(e)}")

    def validate_sql_query(self, sql_query, text):
        try:
            response = self.client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",  # or your preferred model
                messages=[
                    {"role": "system", "content": """Will the following SQL query give "{text}" when run in SQL Server?
                     Response must strictly be a JSON
                     {{ "corrected_sql_query" : <corrected SQL query, or return the original SQL query if its correct>
                     }}
                     """},
                    {"role": "user", "content": f"""
                    SQL Query : {sql_query}
                    """}
                ],
                temperature=0.0,  # Low temperature for more consistent results
                max_tokens=8192,
                stream=False,
                response_format={"type": "json_object"},
                stop=None
            )
            
            # Parse the response
            print(f"DEBUG - raw response is {response.choices[0].message.content}")
            try:
                validated_query = json.loads(response.choices[0].message.content.replace("```json", "").replace("```", ""))
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to validate sql query: {e}")
                raise QueryError("Failed to validate SQL query")
                
        except Exception as e:
            self.logger.error(f"LLM query failed: {e}")
            raise QueryError(f"Failed to process query: {str(e)}")

        return validated_query["corrected_sql_query"]
    
    def process_query(self, user_query: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Process natural language query and return SQL query, clarification question, and query understanding"""
        try:
            # Generate and send prompt to LLM
            prompt = self.generate_llm_prompt(user_query)
            parsed_response = self.query_llm(prompt)
            
            # Log the interpreted query understanding
            self.logger.info(f"Query understanding: {parsed_response.get('query_understanding')}")
            
            if parsed_response.get("is_table_info_sufficient", False):
                return None, "Database does not have enough information to answer your query!", parsed_response.get("query_understanding")
            
            # If clarification is needed, return the question
            if parsed_response.get("needs_clarification", True):
                return None, parsed_response["clarification_question"], parsed_response.get("query_understanding")
            
            # Generate and validate SQL query
            sql = self.generate_sql_from_parsed_response(parsed_response)
            print(f"DEBUG - original SQL query is {sql}")

            return sql, None, parsed_response.get("query_understanding")
            
        except QueryError as e:
            raise e
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise QueryError(f"Unexpected error during query processing: {str(e)}")

    def start_interaction(self):
        """Start the interactive query process"""
        print("Welcome to the Enhanced Natural Language SQL Query Generator!")
        print("Ask your question about jewelery stock in natural language.")
        print("Type 'quit' to exit or 'help' for example queries.")
        
        while True:
            user_query = input("\nEnter your question: ").strip()
            
            if user_query.lower() == 'quit':
                break
                
            if user_query.lower() == 'help':
                print("\nExample queries:")
                print("- Show me total sales by customer for last month")
                print("- What are the top 10 selling items by value this year?")
                print("- Find all transactions with gold purity above 22K")
                print("- Show daily sales trends for the past week")
                continue
                
            try:
                sql_query, clarification, understanding = self.process_query(user_query)
                
                if understanding:
                    print("\nI understood your query as:")
                    print(understanding)
                
                while clarification:
                    print("\nI need some clarification:")
                    print(clarification)
                    response = input("Your response: ")
                    # Update query with clarification
                    sql_query, _, new_understanding = self.process_query(f"{user_query} {response}")
                
                if new_understanding and not clarification:
                        print("\nUpdated understanding after clarification:")
                        print(new_understanding)
                        validated_sql_query = self.validate_sql_query(sql_query, user_query)
                        print(f"DEBUG - validated SQL query is {validated_sql_query}")
                    
            except QueryError as e:
                print(f"\nError: {str(e)}")
                print("Please try rephrasing your question or type 'help' for examples.")
            except Exception as e:
                print(f"\nUnexpected error: {str(e)}")
                print("Please try again or contact support if the issue persists.")
            
            print("\n" + "="*50)

# Example usage
if __name__ == "__main__":
    #import asyncio
    
    def main():
        converter = NLToSQLConverter()
        converter.start_interaction()
    
    main()