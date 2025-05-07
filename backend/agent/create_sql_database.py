# setup_product_db.py
import sqlite3
import os

DB_NAME = "product_database.db"

# Remove old DB if it exists, for a clean setup
if os.path.exists(DB_NAME):
    os.remove(DB_NAME)

conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# Create Categories Table
cursor.execute('''
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL UNIQUE
)
''')

# Create Products Table
cursor.execute('''
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    price REAL NOT NULL,
    category_id INTEGER,
    stock_quantity INTEGER DEFAULT 0,
    color TEXT,
    size TEXT,
    FOREIGN KEY (category_id) REFERENCES categories (category_id)
)
''')

# Insert Sample Data
categories_data = [
    ('Electronics',),
    ('Apparel',),
    ('Books',),
    ('Home Goods',)
]
cursor.executemany("INSERT INTO categories (category_name) VALUES (?)", categories_data)

products_data = [
    ('Laptop Pro 15"', 'High-performance laptop for professionals', 1299.99, 1, 50, 'Silver', '15-inch'),
    ('Wireless Mouse', 'Ergonomic wireless mouse with 5 buttons', 25.99, 1, 150, 'Black', None),
    ('Men\'s T-Shirt', 'Comfortable cotton t-shirt', 19.99, 2, 200, 'Blue', 'M'),
    ('Women\'s Jeans', 'Stylish slim-fit jeans', 49.99, 2, 120, 'Dark Wash', '28'),
    ('The Python Handbook', 'A comprehensive guide to Python programming', 35.00, 3, 80, None, None),
    ('Sci-Fi Novel X', 'An exciting adventure in space', 15.50, 3, 60, None, None),
    ('Coffee Maker Deluxe', '12-cup programmable coffee maker', 59.95, 4, 75, 'Black', None),
    ('Gaming Laptop XYZ', 'Top-tier gaming laptop with RTX GPU', 1999.00, 1, 25, 'Black', '17-inch'),
    ('Red Sneakers', 'Comfortable running sneakers', 79.99, 2, 90, 'Red', '10'),
    ('Blue Hoodie', 'Warm fleece hoodie', 39.99, 2, 110, 'Navy Blue', 'L'),
]
cursor.executemany("INSERT INTO products (name, description, price, category_id, stock_quantity, color, size) VALUES (?, ?, ?, ?, ?, ?, ?)", products_data)

conn.commit()
conn.close()

print(f"Database '{DB_NAME}' created and populated successfully.")