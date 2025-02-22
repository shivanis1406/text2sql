import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import json
from openai import OpenAI
#from groq import Groq
import os
from enum import Enum
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

#model = "deepseek-r1-distill-llama-70b"
MODEL = "gpt-4o"

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
        #self.client = Groq(api_key=api_key or os.getenv('GROQ_API_KEY'))
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def generate_llm_prompt(self, user_query: str) -> str:
        """Generate an enhanced prompt for the LLM"""
        prompt = f"""As an SQL expert, analyze the following user query for a jewellery business database and provide the necessary SQL components.

Table: BI_STOCK_TRANSACTION_DAILY

Available columns and their details:
"""
        # Add detailed column information
        #for col_name, col_info in self.columns_info.items():
        #    prompt += f"\n- {col_name}:"
        #    prompt += f"\n  Description: {col_info.description}"
        #    prompt += f"\n  Type: {col_info.column_type.value}"
        #    if col_info.example_values:
        #        prompt += f"\n  Examples: {', '.join(col_info.example_values)}"
        #    if col_info.is_mandatory:
        #        prompt += "\n  (Mandatory field)"

        prompt += """
### **General Transaction Details**
1. **UNIQUEID** : A unique identifier for each transaction.
2. **BRANCH_CODE** : Code representing the branch where the transaction took place.
3. **REFMID** : A reference or mapping ID for the transaction.
4. **DIVISION_CODE** : Code representing the division associated with the transaction.
5. **SUPPLIER** : Name or code of the supplier involved in the transaction.
6. **VOCTYPE** : Type of voucher (e.g., purchase, sale, adjustment).
VOCTYPE -	Possible Meaning -	Description
AFP - Advance Payment -	Indicates an advance payment made by a customer or to a supplier before receiving goods or services.
CBA -	Cash Bank Adjustment -	Represents adjustments between cash and bank accounts, possibly for reconciliation purposes.
CBF -	Cash Bank Fund Transfer -	Denotes the transfer of funds between cash and bank accounts.
CBR -	Cash Bank Receipt -	Records a receipt of cash or bank deposit, typically when a customer makes a payment.
CPR -	Cash Purchase Return -	Used when purchased items are returned and a refund or credit is processed.
CPU -	Cash Purchase -	Represents a direct purchase made with cash instead of credit.
CRS -	Credit Sale -	Indicates a sale made on credit, where payment is due later.
CSL -	Cash Sale -	Represents an immediate sale where the payment is received in cash.

7. **VOCNO** : Unique voucher number.
8. **SRNO** : Serial number within the transaction.
10. **VOCDATE** : Date of the transaction in DD-MM-YYYY format.
11. **VALUE_DATE** : Date used for valuation purposes in DD-MM-YYYY format.
12. **STOCK_CODE** : Code representing the stock item involved.
13. **PARTYCODE** : Code for the customer or supplier involved.

### **Item Details**
14. **ITEM_CURRENCY** : Currency in which the transaction is recorded.
15. **ITEM_CURR_RATE** : Exchange rate of the item's currency.
16. **FIXED** : Indicates if the stock is fixed (binary: 1/0).
17. **PCS** : Number of pieces in the transaction.
18. **GROSSWT** : Gross weight of the item.
19. **STONEWT** : Weight of any stones included in the item.
20. **NETWT** : Net weight of the item after deductions.
21. **PURITY** : Purity level of the metal (e.g., gold purity in karats). Its range is 0 to 1.
22. **PUREWT** : Pure weight of the metal.
23. **CHARGABLEWT** : Weight used for charging purposes.

### **Pricing and Valuation**
24. **MKG_RATEFC** : Making charges in foreign currency.
25. **MKG_RATECC** : Making charges in local currency.
26. **MKGVALUEFC** : Total making charges in foreign currency.
27. **MKGVALUECC** : Total making charges in local currency.
28. **RATE_TYPE** : Type of rate applied (fixed, variable, etc.).
29. **METAL_RATE** : Metal rate applied for valuation.
30. **METAL_RATE_GMSFC** : Metal rate per gram in foreign currency.
31. **METAL_RATE_GMSCC** : Metal rate per gram in local currency.
32. **METALVALUEFC** : Metal value in foreign currency.
33. **METALVALUECC** : Metal value in local currency.
34. **STONE_RATEFC** : Stone rate in foreign currency.
35. **STONE_RATECC** : Stone rate in local currency.
36. **STONEVALUEFC** : Stone value in foreign currency.
37. **STONEVALUECC** : Stone value in local currency.
38. **NETVALUEFC** : Net value of the transaction of type given in VOCTYPE in foreign currency.
39. **NETVALUECC** : Net value of the transaction of type given in VOCTYPE in local currency.
40. **PUDIFF** : Difference in price due to purity.
41. **STONEDIFF** : Difference in price due to stone valuation.

### **Operational Details**
42. **PONO** : Purchase order number.
43. **LOCTYPE_CODE** : Location type code.
44. **SALESPERSON_CODE** : Salesperson responsible for the transaction.
45. **MKG_STOCKVALUE** : Making stock value.
46. **ADDL_AMTCC** : Additional amount in local currency.
47. **ADDL_AMTFC** : Additional amount in foreign currency.
48. **LANDING_COST** : Cost incurred for bringing the stock into inventory.
49. **SUB_STOCK_CODE** : Sub-category of the stock item.
50. **USERNAME** : User who entered the transaction.
51. **SYSTEM_DATE** : Date when the transaction was recorded in the system.
52. **FROMBRANCH** : Branch from which stock is transferred.
53. **NARRATION** : Description or remarks about the transaction.

### **Discounts and Charges**
54. **DISCOUNT_PER** : Discount percentage applied.
55. **DISCOUNTVALUEFC** : Discount value in foreign currency.
56. **DISCOUNTVALUECC** : Discount value in local currency.
57. **PROCESS_CODE** : Code representing the processing stage.
58. **JOBNO** : Job number related to the item.
59. **WORKER_CODE** : Worker involved in the process.
60. **FROM_PROCESS_CODE** : Previous process stage code.
61. **CARAT** : Carat weight of the item.
62. **QTY_LOSS** : Quantity lost during processing.
63. **LABOUR_AMOUNT** : Labour charges.

### **POS (Point of Sale) Adjustments**
64. **PROCESS_STATUS** : Status of the item in processing.
65. **POS_PURCHASE_VALUE_CC** : Purchase value at the POS in local currency.
66. **POS_ADJUST_ADVANCECC** : POS adjustment advance amount in local currency.
67. **POS_DISCOUNTCC** : POS discount in local currency.
68. **POS_SRETURN_VALUE_CC** : POS return value in local currency.

### **Branch Transfers and Wastage**
69. **UNQ_JOB_ID** : Unique job identifier.
70. **TOBRANCH** : Destination branch for the transfer.
71. **WASTAGEPER** : Wastage percentage.
72. **WASTAGEQTY** : Wastage quantity.
73. **WASTAGEAMOUNTFC** : Wastage value in foreign currency.
74. **WASTAGEAMOUNTCC** : Wastage value in local currency.
75. **COSTINGLANDINGCOST** : Costing-related landing cost.
76. **COSTINGLANDINGVALUE** : Costing-related landing value.
77. **COSTINGWASTAGECOST** : Costing wastage cost.
78. **COSTINGWASTAGEVALUE** : Costing wastage value.

### **Stone and Metal Details**
79. **METALGROSSWT** : Gross weight of the metal.
80. **DIAPCS** : Number of diamond pieces.
81. **DIACARAT** : Carat weight of diamonds.
82. **STONEPCS** : Number of stones.
83. **STONECARAT** : Carat weight of stones.
84. **PUDIFF_AMTLC** : Price difference amount in local currency.
85. **PUDIFF_AMTFC** : Price difference amount in foreign currency.

### **GST and Taxes**
86. **HSN_CODE** : Harmonized System Nomenclature (HSN) code for taxation.
87. **CGST_PER** : Central GST percentage.
88. **CGST_AMOUNTFC** : CGST amount in foreign currency.
89. **CGST_AMOUNTCC** : CGST amount in local currency.
90. **SGST_PER** : State GST percentage.
91. **SGST_AMOUNTFC** : SGST amount in foreign currency.
92. **SGST_AMOUNTCC** : SGST amount in local currency.
93. **IGST_PER** : Integrated GST percentage.
94. **IGST_AMOUNTFC** : IGST amount in foreign currency.
95. **IGST_AMOUNTCC** : IGST amount in local currency.
96. **GST_STATE_CODE** : GST state code for taxation.
97. **INVOICE_NUMBER** : Invoice number.
98. **SUPINVNO** : Supplier invoice number.
99. **SUPINVDATE** : Supplier invoice date.

### **Certification and Additional Charges**
100. **CERT_CGST_PER** : GST on certificate charges.
101. **CERT_SGST_PER** : GST on certificate charges.
102. **CERT_IGST_PER** : GST on certificate charges.
103. **CERT_CGST_AMT** : GST amount on certificate charges.
104. **CERT_SGST_AMT** : GST amount on certificate charges.
105. **CERT_IGST_AMT** : GST amount on certificate charges.

### **Pearl and Kundan Stock Details**
106. **UFX_PEARLPCS** : Number of pearl pieces.
107. **UFX_PEARLCARAT** : Carat weight of pearls.
108. **UFX_PEARLCARATVALUECC** : Pearl carat value in local currency.
109. **UFX_PEARLCARATVALUEFC** : Pearl carat value in foreign currency.
110. **KUNDAN_PCS** : Number of Kundan pieces.
111. **KUNDAN_CARAT** : Kundan carat weight.
112. **KUNDAN_WEIGHT** : Weight of Kundan items.

### **Additional Charges and Adjustments**  
113. **KUNDAN_CARATVALUECC** : Value of Kundan carat weight in local currency.  
114. **KUNDAN_CARATVALUEFC** : Value of Kundan carat weight in foreign currency.  
115. **UFX_MISCWT** : Miscellaneous weight, possibly for unclassified materials.  
116. **UFX_MISCVALUECC** : Value of miscellaneous items in local currency.  
117. **UFX_MISCVALUEFC** : Value of miscellaneous items in foreign currency.  
118. **UFX_MISCPCS** : Number of miscellaneous pieces.  

### **Additional Metal and Stone Attributes**  
119. **UFX_METALDIFFCC** : Difference in metal valuation in local currency.  
120. **UFX_METALDIFFFC** : Difference in metal valuation in foreign currency.  
121. **UFX_LABOURDIFFCC** : Difference in labour charges in local currency.  
122. **UFX_LABOURDIFFFC** : Difference in labour charges in foreign currency.  
123. **UFX_TOTALDIFFCC** : Total difference in valuation in local currency.  
124. **UFX_TOTALDIFFFC** : Total difference in valuation in foreign currency.  

### **Certification & Processing Fees**  
125. **UFX_CERTIFICATENOO** : Certificate number issued for the product.  
126. **UFX_CERTIFICATION_CHARGECC** : Certification charges in local currency.  
127. **UFX_CERTIFICATION_CHARGEFC** : Certification charges in foreign currency.  
128. **UFX_CERTIFICATION_STATUS** : Status of certification (Approved, Pending, etc.).  

### **Transaction and Inventory Movement Details**  
129. **UFX_ISSUE_STATUS** : Status of issuance (e.g., issued, pending, returned).  
130. **UFX_RECEIVED_STATUS** : Status of receipt (received, in-transit, pending).  
131. **UFX_TRANSFER_NO** : Number assigned for stock transfers.  
132. **UFX_TRANSFER_STATUS** : Status of stock transfer (completed, in-progress, failed).  

### **Inventory and Reconciliation Adjustments**  
133. **UFX_ADJUSTMENT_CODE** : Code representing any inventory adjustment.  
134. **UFX_ADJUSTMENT_REASON** : Reason for adjustment (damage, shrinkage, weight loss, etc.).  
135. **UFX_ADJUSTMENT_VALUECC** : Adjustment value in local currency.  
136. **UFX_ADJUSTMENT_VALUEFC** : Adjustment value in foreign currency.  
137. **UFX_ADJUSTMENT_DATE** : Date when adjustment was made.  

### **Miscellaneous & System Data**  
138. **UFX_LAST_UPDATED_BY** : Username of the person who last updated the transaction.  
139. **UFX_LAST_UPDATED_DATE** : Date when the transaction was last updated.  
140. **UFX_REMARKS** : Any additional comments or remarks regarding the transaction.

#### **Additional Charges & Fees**  
141. **UFX_STORAGE_CHARGECC** : Storage charges applied in local currency.  
142. **UFX_STORAGE_CHARGEFC** : Storage charges applied in foreign currency.  
143. **UFX_INSURANCE_CHARGECC** : Insurance charges for stock in local currency.  
144. **UFX_INSURANCE_CHARGEFC** : Insurance charges for stock in foreign currency.  
145. **UFX_HANDLING_CHARGECC** : Handling charges for processing the stock in local currency.  
146. **UFX_HANDLING_CHARGEFC** : Handling charges for processing the stock in foreign currency.  

#### **Taxation & Government Levies**  
147. **UFX_TAX_AMOUNTCC** : Total tax amount applied in local currency.  
148. **UFX_TAX_AMOUNTFC** : Total tax amount applied in foreign currency.  
149. **UFX_TAX_CATEGORY** : Type of tax applied (GST, VAT, etc.).  
150. **UFX_TAX_RATE** : Percentage of tax applied to the transaction.  

#### **Discounts & Promotions**  
151. **UFX_DISCOUNT_PERCENT** : Percentage of discount applied.  
152. **UFX_DISCOUNT_AMOUNTCC** : Discount amount in local currency.  
153. **UFX_DISCOUNT_AMOUNTFC** : Discount amount in foreign currency.  

#### **Product Lifecycle & Status**  
154. **UFX_PRODUCT_STAGE** : Stage of the product (Manufacturing, Processing, Retail, etc.).  
155. **UFX_PRODUCT_AGE** : Number of days the product has been in stock.  
156. **UFX_RESALE_STATUS** : Indicates whether the item is resalable.  
157. **UFX_DAMAGED_STATUS** : Status indicating whether the item is damaged.  

#### **Shipping & Delivery**  
158. **UFX_SHIPPING_MODE** : Mode of shipment (Air, Road, Sea, etc.).  
159. **UFX_SHIPPING_PARTNER** : Name of the shipping company or logistics provider.  
160. **UFX_SHIPPING_TRACKINGNO** : Tracking number assigned by the shipping company.  
161. **UFX_SHIPPING_DATE** : Date when the item was shipped.  
162. **UFX_EXPECTED_DELIVERY_DATE** : Estimated date of delivery.  
163. **UFX_ACTUAL_DELIVERY_DATE** : Actual date when the item was delivered.  

#### **System & Audit Information**  
164. **UFX_CREATED_BY** : User who created the transaction record.  
165. **UFX_CREATED_DATE** : Date when the record was created.  
166. **UFX_MODIFIED_BY** : User who last modified the record.  
167. **UFX_MODIFIED_DATE** : Date of the last modification.  

### **Customer & Supplier Details**  
168. **UFX_CUSTOMER_ID** : Unique identifier for the customer associated with the transaction.  
169. **UFX_CUSTOMER_NAME** : Name of the customer involved in the transaction.  
170. **UFX_SUPPLIER_ID** : Unique identifier for the supplier providing the stock.  
171. **UFX_SUPPLIER_NAME** : Name of the supplier providing the stock.  

### **Warehouse & Storage Information**  
172. **UFX_WAREHOUSE_ID** : Unique identifier for the warehouse where the stock is stored.  
173. **UFX_WAREHOUSE_LOCATION** : Physical location of the warehouse.  
174. **UFX_STORAGE_SECTION** : Section within the warehouse where the item is stored.  
175. **UFX_STORAGE_RACK_NO** : Rack number in the warehouse for better inventory tracking.  

### **Additional Financials**  
176. **UFX_PAYMENT_MODE** : Payment mode used (Cash, Credit, Bank Transfer, etc.).  
177. **UFX_PAYMENT_STATUS** : Status of payment (Paid, Pending, Overdue, etc.).  
178. **UFX_REFUND_AMOUNTCC** : Amount refunded in local currency.  
179. **UFX_REFUND_AMOUNTFC** : Amount refunded in foreign currency.  
180. **UFX_CREDIT_NOTE_NO** : Reference number for any credit note issued.  

### **Product Warranty & Returns**  
181. **UFX_WARRANTY_PERIOD** : Warranty period of the product, if applicable.  
182. **UFX_WARRANTY_STATUS** : Status of warranty (Active, Expired, etc.).  
183. **UFX_RETURN_REQUESTED** : Indicates if a return has been requested for the product.  
184. **UFX_RETURN_APPROVED** : Indicates if the return request was approved.  
185. **UFX_RETURN_REASON** : Reason for the return request.

### **Taxation & GST Details**  
- **CGST_PER** : Central Goods and Services Tax (CGST) percentage applicable to the transaction.  
- **CGST_AMOUNTFC** : CGST amount in foreign currency.  
- **CGST_AMOUNTCC** : CGST amount in local currency.  
- **SGST_PER** : State Goods and Services Tax (SGST) percentage applicable to the transaction.  
- **SGST_AMOUNTFC** : SGST amount in foreign currency.  
- **SGST_AMOUNTCC** : SGST amount in local currency.  
- **IGST_PER** : Integrated Goods and Services Tax (IGST) percentage applicable for interstate transactions.  
- **IGST_AMOUNTFC** : IGST amount in foreign currency.  
- **IGST_AMOUNTCC** : IGST amount in local currency.  
- **GST_STATE_CODE** : Code representing the state for GST calculations.  
- **CESS_PER** : CESS (additional tax) percentage applied to the transaction.  
- **CESS_AMOUNTFC** : CESS amount in foreign currency.  
- **CESS_AMOUNTCC** : CESS amount in local currency.  
- **CERT_CGST_PER** : CGST percentage applicable to certification charges.  
- **CERT_SGST_PER** : SGST percentage applicable to certification charges.  
- **CERT_IGST_PER** : IGST percentage applicable to certification charges.  
- **CERT_CGST_AMT** : CGST amount on certification charges.  
- **CERT_SGST_AMT** : SGST amount on certification charges.  
- **CERT_IGST_AMT** : IGST amount on certification charges.  
- **CERT_CGST_AMTCC** : CGST amount in local currency on certification charges.  
- **CERT_SGST_AMTCC** : SGST amount in local currency on certification charges.  
- **CERT_IGST_AMTCC** : IGST amount in local currency on certification charges.  

---

### **Certification, Premium & Other Charges**  
- **UFX_CERTCHARGECC** : Certification charges in local currency.  
- **UFX_CERTCHARGEFC** : Certification charges in foreign currency.  
- **PREMIUMCHARGESCC** : Premium charges in local currency.  
- **PREMIUMCHARGESFC** : Premium charges in foreign currency.  
- **UFX_LABOURCHARGECC** : Labour charges in local currency.  
- **UFX_LABOURCHARGEFC** : Labour charges in foreign currency.  
- **UFX_HMCHARGECC** : Hallmarking charges in local currency.  
- **UFX_HMCHARGEFC** : Hallmarking charges in foreign currency.  

---

### **Transaction & Invoice Details**  
- **INVOICE_NUMBER** : Invoice number assigned to the transaction.  
- **DOC_REF** : Reference number for the document associated with the transaction.  
- **SUPINVNO** : Supplier’s invoice number for the purchase.  
- **SUPINVDATE** : Date of the supplier’s invoice.  
- **SET_REF_CODE** : Reference code assigned to a set of items.  

---

### **TCS & TDS Details**  
- **TCS_AMOUNT** : Tax Collected at Source (TCS) amount in foreign currency.  
- **TCS_AMOUNTCC** : TCS amount in local currency.  
- **TDS_AMOUNTFC** : Tax Deducted at Source (TDS) amount in foreign currency.  
- **TDS_AMOUNTCC** : TDS amount in local currency.  

---

### **Miscellaneous Columns**  
- **POP_CUST_CODE** : Customer code for point-of-purchase transactions.  
- **UFX_DIACARATVALUECC** : Value of diamonds in carats in local currency.  
- **UFX_DIACARATVALUEFC** : Value of diamonds in carats in foreign currency.  
- **UFX_STONECARATVALUECC** : Value of other stones in carats in local currency.  
- **UFX_STONECARATVALUEFC** : Value of other stones in carats in foreign currency.  
        """

        prompt += f"""

User Query: {user_query}

Please analyze the query and provide:
1. Required columns and any necessary aggregations that are valid in SQL server
2. Appropriate WHERE conditions that are valid in SQL server
3. Any needed GROUP BY or ORDER BY clauses that are valid in SQL server
4. If the query is ambiguous, specify what clarification is needed

Response must strictly be a JSON:
{{
    "needs_clarification": boolean,
    "clarification_question": "string - question to ask if user's query does not have enough information for the SQL query to get data from the Table. Question must NOT contain any column's name",
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
        global MODEL
        print(f"DEBUG - prompt is {prompt}")
        try:
            if "gpt-4o" in MODEL:
                response = self.client.chat.completions.create(
                    model=MODEL,  # or your preferred model
                    messages=[
                        {"role": "system", "content": "You are an expert SQL query generator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for more consistent results
                    max_tokens=8192
                )
            if "deepseek" in MODEL:
                response = self.client.chat.completions.create(
                    model=MODEL,  # or your preferred model
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

    def generate_sql_from_parsed_response_old(self, parsed_response: Dict) -> str:
        """Generate optimized SQL query from parsed LLM response"""
        try:
            
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
        
    def generate_sql_from_parsed_response(self, parsed_response: Dict) -> str:
        """Generate optimized SQL query from parsed LLM response"""
        try:
            # Build SELECT clause
            select_cols = []
            if parsed_response.get("columns"):
                select_cols.extend(parsed_response["columns"])
            if parsed_response.get("aggregations"):
                select_cols.extend(parsed_response["aggregations"])
            sql = f"SELECT {', '.join(select_cols)} FROM BI_STOCK_TRANSACTION_DAILY"

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
                order_clauses = [f"{order['column']} {order['direction']}" for order in parsed_response["order_by"]]
                sql += f" ORDER BY {', '.join(order_clauses)}"

            return sql
        except Exception as e:
            return f"SQL Generation Error: {str(e)}"

    def validate_sql_query(self, user_query, sql_query):
        global MODEL
        try:
            if "gpt-4o" in MODEL:
                response = self.client.chat.completions.create(
                    model=MODEL,  # or your preferred model
                    messages=[
                        {"role": "system", "content": """Given the user's question and corresponding SQL query, correct the SQL query so that it matches user's intent and remove syntax errors, if any
                        Response must strictly be a JSON
                        {{ "corrected_sql_query" : <corrected SQL query, or return the original SQL query if its correct>
                        }}
                        """},
                        {"role": "user", "content": f"""
                        User's Question -
                        {user_query}

                        Generated SQL Query based on User's Question-
                        {sql_query}
                        """}
                    ],
                    temperature=0.0,  # Low temperature for more consistent results
                    max_tokens=8192
                )
                
            if "deepseek" in MODEL:
                response = self.client.chat.completions.create(
                    model=MODEL,  # or your preferred model
                    messages=[
                        {"role": "system", "content": """Is the given SQL query syntactically correct to run in SQL Server?
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
            
            # Log the parsed_response JSON
            self.logger.info(f"""Response JSON: {parsed_response}""")
                        
            # If clarification is needed, return the question
            if parsed_response["needs_clarification"]:
                return None, parsed_response["clarification_question"], parsed_response.get("query_understanding")
            
            # Generate and validate SQL query
            sql = self.generate_sql_from_parsed_response(parsed_response)
            #print(f"DEBUG - original SQL query is {sql}")

            return sql, None, parsed_response.get("query_understanding")
            
        except QueryError as e:
            raise e
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise QueryError(f"Unexpected error during query processing: {str(e)}")

    def start_interaction(self, user_query):
        """Start the interactive query process"""
        print("Welcome to the Enhanced Natural Language SQL Query Generator!")
        print("Ask your question about jewellery stock in natural language.")
        print("Type 'quit' to exit or 'help' for example queries.")
        
        while True:
            #user_query = input("\nEnter your question: ").strip()
            
            if user_query.lower() == 'quit':
                break
                
            if user_query.lower() == 'help':
                st.write("\nExample queries:")
                st.write("- Show me total sales by customer for last month")
                st.write("- What are the top 10 selling items by value this year?")
                st.write("- Find all transactions with gold purity above 22K")
                st.write("- Show daily sales trends for the past week")
                continue
                
            try:
                st.write("Processing query...")
                sql_query, clarification, understanding = self.process_query(user_query)
                
                if understanding:
                    st.write("\nI understood your query as:")
                    st.write(understanding)
                
                if clarification:
                    st.write("\nI need some clarification:")
                    st.write(clarification)
                    return "NA"
                
                if not clarification:
                    print(f"DEBUG - Original SQL query is {sql_query}")
                    validated_sql_query = self.validate_sql_query(user_query, sql_query)
                    print(f"DEBUG - validated SQL query is {validated_sql_query}")
                    return validated_sql_query
                    
            except QueryError as e:
                st.error(f"\nError: {str(e)}")
                st.write("Please try rephrasing your question or type 'help' for examples.")
                return None
            except Exception as e:
                st.error(f"\nUnexpected error: {str(e)}")
                st.write("Please try again or contact support if the issue persists.")
                return None
            st.write("\n" + "="*50)

def text2sql(query):
        converter = NLToSQLConverter()
        validated_sql_query = converter.start_interaction(query)
        return validated_sql_query
