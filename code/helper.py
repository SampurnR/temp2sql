import os
import sqlite3
import urllib.request
import pandas as pd
# from google import genai  # Gemini (Google GenAI)
# from openai import AzureOpenAI  # Azure OpenAI (direct SDK)
from langchain_openai import AzureChatOpenAI  # Azure OpenAI (LangChain)
from langchain_core.messages import SystemMessage, HumanMessage


# --- Paths ---

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_CODE_DIR, '..', 'data')
DB_PATH = os.path.join(DATA_DIR, 'ecommerce.db')


# --- Schema Definitions ---

CUSTOMERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(50),
    phone_number VARCHAR(50),
    address VARCHAR(50),
    city VARCHAR(50),
    country VARCHAR(50),
    postal_code VARCHAR(50),
    loyalty_points INT
);
"""

PRODUCTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    product_id INT PRIMARY KEY,
    product_name TEXT,
    description TEXT,
    price DECIMAL(10,2),
    discount_percentage DECIMAL(5,2),
    category VARCHAR(50),
    brand TEXT,
    stock_quantity INT,
    color VARCHAR(50),
    size VARCHAR(20),
    weight DECIMAL(5,2),
    dimensions TEXT,
    release_date DATE,
    rating DECIMAL(3,1),
    reviews_count INT,
    seller_name TEXT,
    seller_rating DECIMAL(3,1),
    seller_reviews_count INT,
    shipping_method VARCHAR(20),
    shipping_cost DECIMAL(6,2)
);
"""

ORDERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    product_id INT,
    quantity INT,
    unit_price DECIMAL(10,2),
    total_price DECIMAL(10,2),
    order_date DATE,
    shipping_address VARCHAR(255),
    payment_method VARCHAR(20),
    status VARCHAR(20),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
"""

COLUMN_DATA_TYPES = {
    'customers': {
        'customer_id': 'int64',
        'first_name': 'object',
        'last_name': 'object',
        'email': 'object',
        'phone_number': 'object',
        'address': 'object',
        'city': 'object',
        'country': 'object',
        'postal_code': 'object',
        'loyalty_points': 'int64'
    },
    'products': {
        'product_id': 'int64',
        'product_name': 'object',
        'description': 'object',
        'price': 'float64',
        'discount_percentage': 'float64',
        'category': 'object',
        'brand': 'object',
        'stock_quantity': 'int64',
        'color': 'object',
        'size': 'object',
        'weight': 'float64',
        'dimensions': 'object',
        'release_date': 'datetime64[ns]',
        'rating': 'float64',
        'reviews_count': 'int64',
        'seller_name': 'object',
        'seller_rating': 'float64',
        'seller_reviews_count': 'int64',
        'shipping_method': 'object',
        'shipping_cost': 'float64'
    },
    'orders': {
        'order_id': 'int64',
        'customer_id': 'int64',
        'product_id': 'int64',
        'quantity': 'int64',
        'unit_price': 'float64',
        'total_price': 'float64',
        'order_date': 'datetime64[ns]',
        'shipping_address': 'object',
        'payment_method': 'object',
        'status': 'object'
    }
}


SYSTEM_PROMPT = """
###ROLE###
You are a highly skilled Text-to-SQL translator with expertise in SQL syntax, database schema interpretation, and natural language understanding. You generate syntactically correct and semantically accurate SQL queries based on user input and a given database schema.

###CONTEXT###
The user is working with a relational database for an e-commerce platform. The database includes three main tables: `customers`, `products`, and `orders`. The goal is to allow users to input natural language queries (in English), and have the model return equivalent SQL statements that accurately extract the requested data using the given schema.

Here is the full schema:

**Customers Table**
```sql
CREATE TABLE IF NOT EXISTS customers (
    customer_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(50),
    phone_number VARCHAR(50),
    address VARCHAR(50),
    city VARCHAR(50),
    country VARCHAR(50),
    postal_code VARCHAR(50),
    loyalty_points INT
);
```

**Products Table**
```sql
CREATE TABLE IF NOT EXISTS products (
    product_id INT PRIMARY KEY,
    product_name TEXT,
    description TEXT,
    price DECIMAL(10,2),
    discount_percentage DECIMAL(5,2),
    category VARCHAR(50),
    brand TEXT,
    stock_quantity INT,
    color VARCHAR(50),
    size VARCHAR(20),
    weight DECIMAL(5,2),
    dimensions TEXT,
    release_date DATE,
    rating DECIMAL(3,1),
    reviews_count INT,
    seller_name TEXT,
    seller_rating DECIMAL(3,1),
    seller_reviews_count INT,
    shipping_method VARCHAR(20),
    shipping_cost DECIMAL(6,2)
);
```

**Orders Table**
```sql
CREATE TABLE IF NOT EXISTS orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    product_id INT,
    quantity INT,
    unit_price DECIMAL(10,2),
    total_price DECIMAL(10,2),
    order_date DATE,
    shipping_address VARCHAR(255),
    payment_method VARCHAR(20),
    status VARCHAR(20),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
```

###TASK###
Your task is to:

1. Read a natural language query about the e-commerce data.
2. Interpret the user's intent based on the schema provided.
3. Generate a valid SQL `SELECT` query that returns the expected result.
4. Ensure correct table joins, column selection, filtering, and grouping as necessary.
5. Handle aggregate functions (e.g., `COUNT`, `AVG`, `SUM`) whereever appropriate.
6. Disambiguate user terms based on schema details (e.g., "buyer" → `customers`, "product rating" → `products.rating`, etc.).

###CONSTRAINTS###

* Only return a valid SQL query as output — no explanations or extra text.
* The user is using sqllite database - respond with correct and valid sqllite syntax
* Use aliases (`AS`) for column names only when the original name is ambiguous.
* Do not create or modify tables.
* Do not assume the existence of tables or columns not provided in the schema.
* Avoid subqueries unless absolutely necessary for correctness or performance.
* Prefer readability: indent joins and clauses properly.

###EXAMPLES###
**Input:** "Show me the names and emails of customers from Canada who have more than 1000 loyalty points."
**Output:**

```sql
SELECT first_name, last_name, email
FROM customers
WHERE country = 'Canada' AND loyalty_points > 1000;
```

**Input:** "List the top 5 products with the highest ratings and their categories."
**Output:**

```sql
SELECT product_name, category, rating
FROM products
ORDER BY rating DESC
LIMIT 5;
```

**Input:** "How many orders were placed in August 2025?"
**Output:**

```sql
SELECT COUNT(*) AS total_orders
FROM orders
WHERE order_date BETWEEN '2025-08-01' AND '2025-08-31';
```

**Input:** "What is the average shipping cost for products sold by sellers with a rating above 4.5?"
**Output:**

```sql
SELECT AVG(shipping_cost) AS average_shipping_cost
FROM products
WHERE seller_rating > 4.5;
```

###OUTPUT FORMAT###
Return only the sqllite SQL query as a code block using triple backticks and the `sql` language tag, like this:

```sql
-- Your SQL query here
```
"""


def setup_database(db_name=DB_PATH, data_dir=DATA_DIR):
    """Create the SQLite database and load CSV data into tables."""
    if os.path.exists(db_name):
        os.remove(db_name)
        print(f"Removed existing database '{db_name}'.")

    # Download CSVs from Mockaroo if not already present
    os.makedirs(data_dir, exist_ok=True)
    csv_urls = {
        os.path.join(data_dir, 'customers.csv'): 'https://api.mockaroo.com/api/dde01370?count=1000&key=11149690',
        os.path.join(data_dir, 'products.csv'):  'https://api.mockaroo.com/api/8ba6f630?count=1000&key=11149690',
        os.path.join(data_dir, 'orders.csv'):    'https://api.mockaroo.com/api/6fa67fe0?count=3000&key=11149690',
    }
    for csv_path, url in csv_urls.items():
        if not os.path.exists(csv_path):
            print(f"Downloading '{csv_path}' from Mockaroo...")
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'curl/7.88.1'})
                with urllib.request.urlopen(req) as resp, open(csv_path, 'wb') as f:
                    f.write(resp.read())
                print(f"  -> Downloaded successfully.")
            except Exception as e:
                print(f"  -> Failed to download '{csv_path}': {e}")

    conn = None
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        print(f"Database '{db_name}' created and connected successfully. ✅")

        cursor.execute(CUSTOMERS_SCHEMA)
        cursor.execute(PRODUCTS_SCHEMA)
        cursor.execute(ORDERS_SCHEMA)
        print("Tables 'customers', 'products', and 'orders' created successfully.")

        csv_to_table_map = {
            os.path.join(data_dir, 'customers.csv'): 'customers',
            os.path.join(data_dir, 'products.csv'): 'products',
            os.path.join(data_dir, 'orders.csv'): 'orders'
        }

        for csv_file, table_name in csv_to_table_map.items():
            if os.path.exists(csv_file):
                print(f"\nProcessing '{csv_file}' for table '{table_name}'...")
                df = pd.read_csv(csv_file)

                expected_schema = COLUMN_DATA_TYPES[table_name]
                expected_cols = list(expected_schema.keys())

                df = df[df.columns.intersection(expected_cols)]
                for col in expected_cols:
                    if col not in df.columns:
                        df[col] = None
                df = df[expected_cols]

                for col, dtype in expected_schema.items():
                    if 'datetime' in dtype:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    else:
                        try:
                            df[col] = df[col].astype(dtype)
                        except (ValueError, TypeError) as e:
                            print(f"  - Warning: Could not convert column '{col}' to {dtype}. Error: {e}. Leaving as is.")

                df.to_sql(table_name, conn, if_exists='append', index=False)
                print(f"  -> Data from '{csv_file}' loaded into '{table_name}' table successfully.")
            else:
                print(f"Warning: '{csv_file}' not found. Skipping data load for '{table_name}'.")

        conn.commit()
        print("\nData committed to the database successfully. 🎉")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except pd.errors.EmptyDataError as e:
        print(f"Pandas error: {e}. One of the CSV files might be empty.")
    except KeyError as e:
        print(f"Schema definition error: A column is missing from the COLUMN_DATA_TYPES dictionary: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")


def get_sql_query(client, prompt, user_query):
    """Generate SQL from a natural language query using the Azure OpenAI API (LangChain)."""

    # --- Gemini (Google GenAI) Implementation ---
    # contents = f"""
    # {prompt}
    #
    # Here's the user query in english you need to work on:
    # {user_query}
    # """
    # response = client.models.generate_content(model='gemini-2.5-flash', contents=contents)
    #
    # usage_metadata = response.usage_metadata
    # print(f"Input Token Count: {usage_metadata.prompt_token_count}")
    # print(f"Thoughts Token Count: {response.usage_metadata.thoughts_token_count}")
    # print(f"Output Token Count: {usage_metadata.candidates_token_count}")
    # print(f"Total Token Count: {usage_metadata.total_token_count}")
    #
    # output = response.text.replace('```sql', '').replace('```', '').strip()
    # return output

    # --- Azure OpenAI (direct SDK) ---
    # response = client.chat.completions.create(
    #     model=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
    #     messages=[
    #         {"role": "system", "content": prompt},
    #         {"role": "user", "content": user_query}
    #     ]
    # )
    # usage = response.usage
    # print(f"Input Token Count: {usage.prompt_tokens}")
    # print(f"Output Token Count: {usage.completion_tokens}")
    # print(f"Total Token Count: {usage.total_tokens}")
    # output = response.choices[0].message.content.replace('```sql', '').replace('```', '').strip()
    # return output

    # --- Azure OpenAI (LangChain) ---
    response = client.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=user_query)
    ])

    token_usage = response.response_metadata.get('token_usage', {})
    print(f"Input Token Count: {token_usage.get('prompt_tokens', 'N/A')}")
    print(f"Output Token Count: {token_usage.get('completion_tokens', 'N/A')}")
    print(f"Total Token Count: {token_usage.get('total_tokens', 'N/A')}")

    output = response.content.replace('```sql', '').replace('```', '').strip()
    return output


def execute_query(query, db_name=DB_PATH):
    """Execute a SQL query against the database and return results as a DataFrame."""
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        print(f"\nExecuting query on '{db_name}':\n{query}")
        cursor.execute(query)

        results = cursor.fetchall()
        columns = [description[0] for description in cursor.description]

        results_as_dict = [dict(zip(columns, row)) for row in results]
        results_df = pd.DataFrame(results_as_dict)

        print("Query executed successfully.")
        return results_df

    except sqlite3.Error as e:
        print(f"Database error executing query: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    finally:
        if conn:
            conn.close()


def text2sql(client, user_query, db_name=DB_PATH):
    """Full pipeline: natural language → SQL → query results."""
    sql = get_sql_query(client, SYSTEM_PROMPT, user_query)
    results = execute_query(sql, db_name)
    return sql, results
