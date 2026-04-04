#!/usr/bin/env python3
"""
Flask Web App for Text-to-SQL Model
Simple web interface to interact with the Ollama text-to-SQL model
"""

from flask import Flask, render_template, request, jsonify, flash
import subprocess
import json
import logging
import os
import argparse
import re
import random
import sqlite3
import sys
from datetime import datetime, timedelta
import sqlparse
from utils import OllamaManager, SQLValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change in production

# Initialize Ollama manager
try:
    ollama_manager = OllamaManager("text-to-sql")
except Exception as e:
    logger.error(f"Failed to initialize Ollama manager: {e}")
    ollama_manager = None

@app.route('/')
def index():
    """Main page with the text-to-SQL interface"""
    return render_template('index.html')

@app.route('/api/generate-sql', methods=['POST'])
def generate_sql():
    """API endpoint to generate SQL from natural language"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        schema = data.get('schema', '').strip()
        query = data.get('query', '').strip()
        
        if not schema:
            return jsonify({
                'success': False,
                'error': 'Database schema is required'
            }), 400
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Natural language query is required'
            }), 400
        
        # Check if Ollama manager is available
        if not ollama_manager:
            return jsonify({
                'success': False,
                'error': 'Ollama model not available. Please check if the model is deployed.'
            }), 500
        
        # Format prompt
        prompt = f"Database Schema:\n{schema}\n\nRequest: {query}"
        
        # Generate SQL using the safer method
        logger.info(f"Generating SQL for query: {query[:50]}...")
        
        if hasattr(ollama_manager, 'query_model_safe'):
            response = ollama_manager.query_model_safe(prompt, timeout=60)
        else:
            response = ollama_manager.query_model(prompt, timeout=60)
        
        if not response:
            return jsonify({
                'success': False,
                'error': 'Failed to get response from model'
            }), 500
        
        # Check for token loop issue
        if "{ end }<|end|>" in response:
            return jsonify({
                'success': False,
                'error': 'Model encountered a token generation issue. Please try again or check model configuration.'
            }), 500
        
        # Strip ANSI escape codes from ollama terminal output
        response = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', response)
        response = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', response)

        # Extract SQL from response
        sql_query = SQLValidator.extract_sql_from_text(response)
        
        if sql_query:
            # Validate SQL syntax
            is_valid, validation_msg = SQLValidator.validate_sql(sql_query)
            
            return jsonify({
                'success': True,
                'sql_query': sql_query,
                'full_response': response,
                'is_valid': is_valid,
                'validation_message': validation_msg,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': True,
                'sql_query': None,
                'full_response': response,
                'is_valid': False,
                'validation_message': 'No SQL query found in response',
                'timestamp': datetime.now().isoformat()
            })
    
    except Exception as e:
        logger.error(f"Error generating SQL: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}'
        }), 500

@app.route('/api/validate-sql', methods=['POST'])
def validate_sql():
    """API endpoint to validate SQL syntax"""
    try:
        data = request.get_json()
        sql_query = data.get('sql', '').strip()
        
        if not sql_query:
            return jsonify({
                'success': False,
                'error': 'SQL query is required'
            })
        
        is_valid, message = SQLValidator.validate_sql(sql_query)
        
        return jsonify({
            'success': True,
            'is_valid': is_valid,
            'message': message
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def _generate_sample_value(col_name, col_type, row_idx=1, num_rows=10):
    """Generate realistic sample data based on column name and type."""
    col_name_lower = col_name.lower()
    col_type_upper = (col_type or '').upper()

    # ID columns
    if col_name_lower in ('id', 'pk'):
        return None  # handled by caller with sequential ids

    # Foreign key columns - use sequential IDs so every parent row gets referenced
    if col_name_lower.endswith('_id') or col_name_lower.startswith('id_'):
        return ((row_idx - 1) % num_rows) + 1

    # Person names
    person_names = ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Lee',
                    'Eve Wilson', 'Frank Garcia', 'Grace Kim', 'Hank Miller',
                    'Ivy Chen', 'Jack Davis']
    if 'name' in col_name_lower and any(kw in col_name_lower for kw in ['user', 'customer', 'patient', 'employee', 'doctor', 'person']):
        return person_names[row_idx % len(person_names)]
    if col_name_lower in ('first_name', 'firstname'):
        return ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Hank', 'Ivy', 'Jack'][row_idx % 10]
    if col_name_lower in ('last_name', 'lastname'):
        return ['Johnson', 'Smith', 'Brown', 'Lee', 'Wilson', 'Garcia', 'Kim', 'Miller', 'Chen', 'Davis'][row_idx % 10]
    if 'doctor' in col_name_lower and 'name' in col_name_lower:
        return ['Dr. Smith', 'Dr. Lee', 'Dr. Garcia', 'Dr. Chen', 'Dr. Wilson'][row_idx % 5]
    if 'product' in col_name_lower and 'name' in col_name_lower:
        return ['Laptop Pro', 'Wireless Mouse', 'USB-C Hub', 'Monitor 27"', 'Keyboard MX',
                'Webcam HD', 'Headphones', 'Desk Lamp', 'Phone Stand', 'Cable Kit'][row_idx % 10]
    if 'name' in col_name_lower:
        return person_names[row_idx % len(person_names)]
    if 'title' in col_name_lower:
        return ['Senior Engineer', 'Product Manager', 'Designer', 'Analyst', 'Director',
                'Intern', 'Lead Dev', 'QA Engineer', 'DevOps', 'CTO'][row_idx % 10]

    # Email
    if 'email' in col_name_lower:
        names = ['alice', 'bob', 'charlie', 'diana', 'eve', 'frank', 'grace', 'hank', 'ivy', 'jack']
        domains = ['example.com', 'test.org', 'company.io']
        return f"{names[row_idx % len(names)]}@{domains[row_idx % len(domains)]}"

    # Phone
    if 'phone' in col_name_lower:
        return f"+1-555-{100 + row_idx:03d}-{1000 + row_idx * 111:04d}"

    # Department
    if 'department' in col_name_lower or 'dept' in col_name_lower:
        return ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance',
                'Operations', 'Support', 'Design', 'Legal', 'Product'][row_idx % 10]

    # Region
    if 'region' in col_name_lower:
        return ['North', 'South', 'East', 'West', 'Central'][row_idx % 5]

    # Gender
    if 'gender' in col_name_lower or 'sex' in col_name_lower:
        return ['Male', 'Female'][row_idx % 2]

    # Address / city
    if 'address' in col_name_lower:
        return f"{100 + row_idx} Main Street"
    if 'city' in col_name_lower:
        return ['New York', 'London', 'Tokyo', 'Singapore', 'Sydney',
                'Berlin', 'Paris', 'Toronto', 'Dubai', 'Seoul'][row_idx % 10]
    if 'country' in col_name_lower:
        return ['US', 'UK', 'JP', 'SG', 'AU', 'DE', 'FR', 'CA', 'AE', 'KR'][row_idx % 10]

    # Status - every 3rd row gets future-related status (aligned with future dates)
    if 'status' in col_name_lower:
        if row_idx % 3 == 0:
            # Alternate between common "future appointment" terms models use
            return ['Upcoming', 'Scheduled', 'Upcoming', 'Scheduled', 'Upcoming',
                    'Scheduled', 'Upcoming', 'Scheduled', 'Upcoming', 'Scheduled'][row_idx % 10]
        return ['active', 'completed', 'active', 'pending', 'active',
                'completed', 'active', 'pending', 'completed', 'active'][row_idx % 10]

    # Category
    if 'category' in col_name_lower:
        return ['Electronics', 'Clothing', 'Books', 'Food', 'Toys',
                'Electronics', 'Sports', 'Home', 'Garden', 'Tools'][row_idx % 10]
    if 'type' in col_name_lower:
        return ['Type A', 'Type B', 'Type C', 'Type A', 'Type B'][row_idx % 5]

    # Price / amount / salary / total - varied realistic amounts
    if 'salary' in col_name_lower:
        return [55000, 72000, 48000, 95000, 63000, 81000, 45000, 110000, 67000, 58000][row_idx % 10]
    if 'price' in col_name_lower:
        return [29.99, 149.99, 9.99, 599.99, 79.99, 39.99, 199.99, 14.99, 24.99, 49.99][row_idx % 10]
    if any(kw in col_name_lower for kw in ['amount', 'total', 'cost', 'revenue', 'budget']):
        return round(random.uniform(50, 5000), 2)

    # Quantity / stock - include low values so "< 10" filters work
    if any(kw in col_name_lower for kw in ['quantity', 'stock', 'count', 'qty']):
        return [3, 45, 7, 120, 2, 89, 15, 5, 200, 8][row_idx % 10]

    # Age - include wide range including > 65 for healthcare queries
    if 'age' in col_name_lower:
        return [25, 34, 67, 45, 72, 28, 55, 70, 38, 80][row_idx % 10]

    # Date of birth - include elderly for healthcare queries
    if 'birth' in col_name_lower or 'dob' in col_name_lower:
        years = [1990, 1985, 1955, 1978, 1950, 1995, 1968, 1952, 1982, 1945]
        return f"{years[row_idx % 10]}-{1 + (row_idx * 3) % 12:02d}-{1 + (row_idx * 7) % 28:02d}"

    # Notes / description / text
    if any(kw in col_name_lower for kw in ['notes', 'description', 'comment', 'remark']):
        return ['Follow-up needed', 'Routine checkup', 'Urgent review', 'No issues found',
                'Pending approval', 'Completed', 'Requires attention', 'Standard procedure',
                'Review next week', 'All clear'][row_idx % 10]

    # Boolean
    if any(kw in col_name_lower for kw in ['is_', 'has_', 'enabled', 'flag']):
        return row_idx % 2

    # created_at / updated_at / registered - use recent dates so "last N days" queries work
    if any(kw in col_name_lower for kw in ['created', 'updated', 'registered', 'signed_up', 'joined']):
        now = datetime.now()
        # Spread across last 60 days so some are within 30 days and some aren't
        offsets = [-2, -45, -5, -35, -10, -50, -1, -25, -15, -55,
                   -3, -40, -7, -30, -12, -48, -4, -20, -8, -42,
                   -6, -38, -9, -28, -14, -52, -3, -22, -11, -46]
        offset = offsets[row_idx % len(offsets)]
        dt = now + timedelta(days=offset)
        if any(t in col_type_upper for t in ['TIME', 'STAMP']):
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        return dt.strftime('%Y-%m-%d')

    # Date/time types - spread across 2024 and recent dates for useful query results
    if any(t in col_type_upper for t in ['DATE', 'TIME', 'STAMP']) or \
       any(kw in col_name_lower for kw in ['date', 'created', 'updated', 'ordered', 'hired']):
        # 2024 dates covering all 12 months for analytics queries
        dates_2024 = [
            '2024-01-10', '2024-01-25', '2024-02-08', '2024-02-22', '2024-03-05',
            '2024-03-19', '2024-04-03', '2024-04-17', '2024-05-01', '2024-05-15',
            '2024-06-07', '2024-06-21', '2024-07-04', '2024-07-18', '2024-08-02',
            '2024-08-16', '2024-09-06', '2024-09-20', '2024-10-11', '2024-10-25',
            '2024-11-08', '2024-11-22', '2024-12-06', '2024-12-20',
        ]
        # Future dates for "upcoming" queries
        now = datetime.now()
        future_offsets = [1, 2, 3, 5, 7, 10, 4, 6, 8, 14]
        future_dates = [(now + timedelta(days=d)).strftime('%Y-%m-%d') for d in future_offsets]
        # Every 3rd row gets a future date, rest get 2024 (2:1 ratio for rich 2024 data)
        if row_idx % 3 == 0:
            dt_str = future_dates[row_idx % len(future_dates)]
        else:
            dt_str = dates_2024[row_idx % len(dates_2024)]
        if any(t in col_type_upper for t in ['TIME', 'STAMP']):
            return dt_str + ' 10:30:00'
        return dt_str

    # Numeric types
    if any(t in col_type_upper for t in ['INT', 'NUM', 'SERIAL']):
        return random.randint(1, 1000)
    if any(t in col_type_upper for t in ['FLOAT', 'REAL', 'DOUBLE', 'DECIMAL']):
        return round(random.uniform(1, 1000), 2)

    # Text fallback
    if any(t in col_type_upper for t in ['CHAR', 'TEXT', 'STRING', 'CLOB']):
        return f"item_{row_idx}"

    # Generic fallback
    return f"val_{row_idx}"


def _parse_columns_from_ddl(ddl):
    """Parse column names and types from CREATE TABLE DDL."""
    columns = []
    # Match column definitions inside parentheses
    create_match = re.search(r'CREATE\s+TABLE\s+\w+\s*\((.*)\)', ddl, re.DOTALL | re.IGNORECASE)
    if not create_match:
        return columns
    body = create_match.group(1)

    # Split by comma but respect parentheses (for things like DECIMAL(10,2))
    depth = 0
    current = []
    parts = []
    for ch in body:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())

    skip_keywords = {'PRIMARY', 'FOREIGN', 'UNIQUE', 'CHECK', 'INDEX', 'CONSTRAINT', 'KEY'}
    for part in parts:
        tokens = part.split()
        if not tokens:
            continue
        first = tokens[0].upper().strip('`"[]')
        if first in skip_keywords:
            continue
        col_name = tokens[0].strip('`"[]')
        col_type = tokens[1] if len(tokens) > 1 else 'TEXT'
        columns.append((col_name, col_type))

    return columns


def _convert_to_sqlite(sql):
    """Convert MySQL/PostgreSQL SQL syntax to SQLite-compatible syntax."""
    # Strip ANSI escape codes and invisible characters
    s = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', sql)
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)

    # NOW() / CURRENT_TIMESTAMP / CURDATE() / CURRENT_DATE
    s = re.sub(r'\bNOW\s*\(\s*\)', "datetime('now')", s, flags=re.IGNORECASE)
    s = re.sub(r'\bCURDATE\s*\(\s*\)', "date('now')", s, flags=re.IGNORECASE)
    s = re.sub(r'\bCURRENT_TIMESTAMP\b', "datetime('now')", s, flags=re.IGNORECASE)
    s = re.sub(r'\bCURRENT_DATE\b', "date('now')", s, flags=re.IGNORECASE)
    s = re.sub(r'\bGETDATE\s*\(\s*\)', "datetime('now')", s, flags=re.IGNORECASE)
    s = re.sub(r'\bSYSDATE\b', "datetime('now')", s, flags=re.IGNORECASE)

    # INTERVAL patterns: NOW() - INTERVAL 30 DAY -> datetime('now', '-30 day')
    # Also: CURRENT_DATE - INTERVAL '30' DAY, date - INTERVAL 30 DAY, etc.
    def replace_date_interval(match):
        date_expr = match.group(1).strip()
        sign = '-' if match.group(2).strip() == '-' else '+'
        num = match.group(3).strip().strip("'\"")
        unit = match.group(4).strip().lower()
        # Map units
        unit_map = {'days': 'day', 'months': 'month', 'years': 'year',
                    'hours': 'hour', 'minutes': 'minute', 'seconds': 'second',
                    'week': 'day', 'weeks': 'day'}
        mapped_unit = unit_map.get(unit, unit)
        if unit in ('week', 'weeks'):
            num = str(int(num) * 7)
        # If the date expression is already a datetime() call, add modifier
        if "datetime(" in date_expr or "date(" in date_expr:
            base = date_expr.rstrip(")")
            return f"{base}, '{sign}{num} {mapped_unit}')"
        else:
            return f"datetime('{date_expr}', '{sign}{num} {mapped_unit}')" if date_expr == 'now' else f"datetime({date_expr}, '{sign}{num} {mapped_unit}')"

    # Match: <expr> +/- INTERVAL <num> <unit>
    s = re.sub(
        r"(datetime\([^)]*\)|date\([^)]*\)|\w+)\s*([+-])\s*INTERVAL\s+['\"]?(\d+)['\"]?\s+(\w+)",
        replace_date_interval, s, flags=re.IGNORECASE
    )

    # DATE_SUB / DATE_ADD with nested-paren awareness
    def _parse_func_then_interval(inner):
        """Parse 'expr, INTERVAL n unit' respecting nested parens."""
        depth = 0
        comma_pos = -1
        for idx_c, ch in enumerate(inner):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == ',' and depth == 0:
                comma_pos = idx_c
                break
        if comma_pos == -1:
            return None
        expr = inner[:comma_pos].strip()
        rest = inner[comma_pos + 1:].strip()
        # Parse INTERVAL n unit
        iv_match = re.match(r"INTERVAL\s+['\"]?(\d+)['\"]?\s+(\w+)", rest, re.IGNORECASE)
        if not iv_match:
            return None
        num = iv_match.group(1)
        unit = iv_match.group(2).lower().rstrip('s')
        return expr, num, unit

    for func_name, sign in [('DATE_SUB', '-'), ('DATE_ADD', '+')]:
        pat = re.compile(func_name + r"\s*\(", re.IGNORECASE)
        while True:
            m = pat.search(s)
            if not m:
                break
            start = m.end()
            depth = 1
            i = start
            while i < len(s) and depth > 0:
                if s[i] == '(':
                    depth += 1
                elif s[i] == ')':
                    depth -= 1
                i += 1
            inner = s[start:i - 1]
            parsed = _parse_func_then_interval(inner)
            if parsed:
                expr, num, unit = parsed
                if "datetime(" in expr or "date(" in expr:
                    replacement = f"{expr.rstrip(')')}, '{sign}{num} {unit}')"
                else:
                    replacement = f"datetime({expr}, '{sign}{num} {unit}')"
            else:
                replacement = m.group(0) + inner + ')'
            s = s[:m.start()] + replacement + s[i:]

    # DATE(expr) -> date(expr) - MySQL DATE() cast to SQLite date()
    s = re.sub(r"\bDATE\s*\(\s*([^)]+)\s*\)", r"date(\1)", s, flags=re.IGNORECASE)

    # DATE_FORMAT(expr, 'fmt') -> strftime('fmt', expr) with nested-paren awareness
    def _parse_date_format_args(inner):
        """Split DATE_FORMAT args respecting nested parentheses."""
        depth = 0
        parts = []
        current = []
        for ch in inner:
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                parts.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            parts.append(''.join(current).strip())
        return parts

    df_pattern = re.compile(r"DATE_FORMAT\s*\(", re.IGNORECASE)
    while True:
        m = df_pattern.search(s)
        if not m:
            break
        start = m.end()
        depth = 1
        i = start
        while i < len(s) and depth > 0:
            if s[i] == '(':
                depth += 1
            elif s[i] == ')':
                depth -= 1
            i += 1
        inner = s[start:i - 1]
        args = _parse_date_format_args(inner)
        if len(args) >= 2:
            expr = args[0].strip()
            fmt = args[1].strip().strip("'\"")
            replacement = f"strftime('{fmt}', {expr})"
        else:
            replacement = m.group(0) + inner + ')'
        s = s[:m.start()] + replacement + s[i:]

    # TIMESTAMPDIFF(YEAR, a, b) -> CAST((julianday(b) - julianday(a)) / 365.25 AS INTEGER)
    def _parse_timestampdiff(s_inner):
        """Parse TIMESTAMPDIFF arguments handling nested parentheses."""
        args = []
        depth = 0
        current = []
        for ch in s_inner:
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            args.append(''.join(current).strip())
        return args

    def replace_timestampdiff(match):
        inner = match.group(1)
        args = _parse_timestampdiff(inner)
        if len(args) < 3:
            return match.group(0)
        unit = args[0].strip().upper()
        expr1 = args[1].strip()
        expr2 = args[2].strip()
        if unit == 'YEAR':
            return f"CAST((julianday({expr2}) - julianday({expr1})) / 365.25 AS INTEGER)"
        elif unit == 'MONTH':
            return f"CAST((julianday({expr2}) - julianday({expr1})) / 30.44 AS INTEGER)"
        elif unit in ('DAY', 'DAYS'):
            return f"CAST(julianday({expr2}) - julianday({expr1}) AS INTEGER)"
        elif unit in ('HOUR', 'HOURS'):
            return f"CAST((julianday({expr2}) - julianday({expr1})) * 24 AS INTEGER)"
        elif unit in ('MINUTE', 'MINUTES'):
            return f"CAST((julianday({expr2}) - julianday({expr1})) * 1440 AS INTEGER)"
        else:
            return f"CAST(julianday({expr2}) - julianday({expr1}) AS INTEGER)"

    # Match TIMESTAMPDIFF(...) handling nested parens
    ts_pattern = re.compile(r"TIMESTAMPDIFF\s*\(", re.IGNORECASE)
    while True:
        m = ts_pattern.search(s)
        if not m:
            break
        start = m.end()
        depth = 1
        i = start
        while i < len(s) and depth > 0:
            if s[i] == '(':
                depth += 1
            elif s[i] == ')':
                depth -= 1
            i += 1
        inner = s[start:i - 1]
        replacement = replace_timestampdiff(type('M', (), {'group': lambda self, x: inner})())
        s = s[:m.start()] + replacement + s[i:]

    # DATEDIFF(a, b) -> CAST(julianday(a) - julianday(b) AS INTEGER)
    s = re.sub(
        r"DATEDIFF\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)",
        r"CAST(julianday(\1) - julianday(\2) AS INTEGER)", s, flags=re.IGNORECASE
    )

    # YEAR(col) -> CAST(strftime('%Y', col) AS INTEGER)
    s = re.sub(r"\bYEAR\s*\(\s*(.+?)\s*\)", r"CAST(strftime('%Y', \1) AS INTEGER)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bMONTH\s*\(\s*(.+?)\s*\)", r"CAST(strftime('%m', \1) AS INTEGER)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bDAY\s*\(\s*(\w+)\s*\)", r"CAST(strftime('%d', \1) AS INTEGER)", s, flags=re.IGNORECASE)

    # EXTRACT(YEAR FROM col) -> CAST(strftime('%Y', col) AS INTEGER)
    s = re.sub(r"EXTRACT\s*\(\s*YEAR\s+FROM\s+(.+?)\s*\)", r"CAST(strftime('%Y', \1) AS INTEGER)", s, flags=re.IGNORECASE)
    s = re.sub(r"EXTRACT\s*\(\s*MONTH\s+FROM\s+(.+?)\s*\)", r"CAST(strftime('%m', \1) AS INTEGER)", s, flags=re.IGNORECASE)
    s = re.sub(r"EXTRACT\s*\(\s*DAY\s+FROM\s+(.+?)\s*\)", r"CAST(strftime('%d', \1) AS INTEGER)", s, flags=re.IGNORECASE)

    # IFNULL is already SQLite-compatible, but COALESCE works too
    # ISNULL(expr) -> (expr IS NULL)
    s = re.sub(r"\bISNULL\s*\(\s*(.+?)\s*\)", r"(\1 IS NULL)", s, flags=re.IGNORECASE)

    # LIMIT x, y -> LIMIT y OFFSET x (MySQL style)
    limit_match = re.search(r"\bLIMIT\s+(\d+)\s*,\s*(\d+)", s, re.IGNORECASE)
    if limit_match:
        offset_val = limit_match.group(1)
        limit_val = limit_match.group(2)
        s = s[:limit_match.start()] + f"LIMIT {limit_val} OFFSET {offset_val}" + s[limit_match.end():]

    # Fix strftime() compared to bare integers: strftime('%Y', x) = 2024 -> strftime('%Y', x) = '2024'
    s = re.sub(r"(strftime\([^)]+\))\s*=\s*(\d+)\b", r"\1 = '\2'", s)
    s = re.sub(r"(strftime\([^)]+\))\s*!=\s*(\d+)\b", r"\1 != '\2'", s)
    s = re.sub(r"(strftime\([^)]+\))\s*>\s*(\d+)\b", r"\1 > '\2'", s)
    s = re.sub(r"(strftime\([^)]+\))\s*<\s*(\d+)\b", r"\1 < '\2'", s)
    s = re.sub(r"(strftime\([^)]+\))\s*>=\s*(\d+)\b", r"\1 >= '\2'", s)
    s = re.sub(r"(strftime\([^)]+\))\s*<=\s*(\d+)\b", r"\1 <= '\2'", s)

    # BOOLEAN -> INTEGER
    s = re.sub(r'\bBOOLEAN\b', 'INTEGER', s, flags=re.IGNORECASE)

    # AUTO_INCREMENT -> (remove, SQLite uses AUTOINCREMENT with INTEGER PRIMARY KEY)
    s = re.sub(r'\s+AUTO_INCREMENT', '', s, flags=re.IGNORECASE)

    # UNSIGNED -> (remove)
    s = re.sub(r'\s+UNSIGNED\b', '', s, flags=re.IGNORECASE)

    # ENGINE=... -> (remove)
    s = re.sub(r'\)\s*ENGINE\s*=\s*\w+.*?;', ');', s, flags=re.IGNORECASE)

    # IF NOT EXISTS (already supported in SQLite, but make sure)

    return s


def _fix_hallucinated_joins(sql, valid_tables):
    """Remove JOINs to tables not in the schema and fix alias references."""
    # Find all JOIN clauses with table names
    # Pattern: JOIN <table> <alias> ON <condition>
    join_pattern = re.compile(
        r'\b(JOIN|INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|LEFT\s+OUTER\s+JOIN)\s+'
        r'(\w+)\s+(\w+)\s+ON\s+(.+?)(?=\bJOIN\b|\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|\)|\Z)',
        re.IGNORECASE | re.DOTALL
    )

    to_remove = []  # (alias, real_table, column_used_from_alias)
    for m in join_pattern.finditer(sql):
        table_name = m.group(2).lower()
        alias = m.group(3)
        if table_name not in valid_tables:
            # Find what column this alias is used for (e.g., r.region)
            to_remove.append((alias, table_name, m.start(), m.end(), m.group(0)))

    if not to_remove:
        return sql

    # Process removals from end to start to preserve positions
    result = sql
    for alias, table_name, start, end, full_match in reversed(to_remove):
        # Remove the JOIN clause
        # Be careful to also remove leading whitespace/newlines
        pre = result[:start].rstrip()
        post = result[end:]
        result = pre + ' ' + post

    # Replace alias.column references with just column (since the column is on the main table)
    for alias, table_name, _, _, _ in to_remove:
        result = re.sub(rf'\b{re.escape(alias)}\.(\w+)', r'\1', result)

    return result


@app.route('/api/test-sql', methods=['POST'])
def test_sql():
    """Create SQLite DB from schema, populate sample data, and run the generated SQL."""
    try:
        data = request.get_json()
        schema_ddl = data.get('schema', '').strip()
        sql_query = data.get('sql', '').strip()
        num_rows = min(data.get('num_rows', 30), 100)

        if not schema_ddl or not sql_query:
            return jsonify({'success': False, 'error': 'Schema and SQL query are required'})

        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Convert schema DDL to SQLite-compatible syntax
        schema_ddl = _convert_to_sqlite(schema_ddl)

        # Execute all CREATE TABLE statements
        statements = [s.strip() for s in sqlparse.split(schema_ddl) if s.strip()]
        tables_created = []
        for stmt in statements:
            if stmt.upper().startswith('CREATE'):
                cursor.execute(stmt)
                # Extract table name
                tbl_match = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)', stmt, re.IGNORECASE)
                if tbl_match:
                    table_name = tbl_match.group(1)
                    columns = _parse_columns_from_ddl(stmt)
                    tables_created.append((table_name, columns))

        conn.commit()

        # Insert sample data for each table
        sample_data_info = {}
        for table_name, columns in tables_created:
            col_names = [c[0] for c in columns]
            rows_inserted = 0
            for row_idx in range(1, num_rows + 1):
                values = []
                for col_name, col_type in columns:
                    if col_name.lower() in ('id', 'pk') or (col_name.lower().endswith('_id') and columns[0][0].lower() == col_name.lower()):
                        values.append(row_idx)
                    else:
                        values.append(_generate_sample_value(col_name, col_type, row_idx, num_rows))
                placeholders = ', '.join(['?'] * len(columns))
                col_list = ', '.join([f'"{c}"' for c in col_names])
                try:
                    cursor.execute(f'INSERT INTO {table_name} ({col_list}) VALUES ({placeholders})', values)
                    rows_inserted += 1
                except sqlite3.IntegrityError:
                    continue
            sample_data_info[table_name] = rows_inserted

        conn.commit()

        # Remove joins to non-existent tables (model hallucination fix)
        valid_tables = {t[0].lower() for t in tables_created}
        sql_query = _fix_hallucinated_joins(sql_query, valid_tables)

        # Convert SQL to SQLite-compatible dialect
        sqlite_query = _convert_to_sqlite(sql_query)
        logger.info(f"Original SQL: {sql_query}")
        logger.info(f"SQLite SQL:   {sqlite_query}")

        # Execute the generated SQL query
        cursor.execute(sqlite_query)
        rows = cursor.fetchall()
        col_names = [description[0] for description in cursor.description] if cursor.description else []
        result_rows = [dict(row) for row in rows]

        conn.close()

        return jsonify({
            'success': True,
            'columns': col_names,
            'rows': result_rows,
            'row_count': len(result_rows),
            'tables_created': list(sample_data_info.keys()),
            'sample_data_info': sample_data_info,
            'original_sql': sql_query,
            'sqlite_sql': sqlite_query if sqlite_query != sql_query else None,
        })

    except sqlite3.Error as e:
        return jsonify({'success': False, 'error': f'SQLite error: {str(e)}. Tip: The model may have generated MySQL/PostgreSQL-specific syntax.'})
    except Exception as e:
        logger.error(f"Error testing SQL: {str(e)}")
        return jsonify({'success': False, 'error': f'Error: {str(e)}'})


@app.route('/api/status')
def status():
    """Check the status of the text-to-SQL service"""
    try:
        # Check if Ollama is running
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        ollama_running = result.returncode == 0
        
        # Check if our model exists
        model_exists = False
        if ollama_running and ollama_manager:
            model_exists = ollama_manager.model_exists()
        
        return jsonify({
            'ollama_running': ollama_running,
            'model_exists': model_exists,
            'model_name': ollama_manager.model_name if ollama_manager else 'N/A',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'ollama_running': False,
            'model_exists': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

@app.route('/examples')
def examples():
    """Page with example schemas and queries"""
    return render_template('examples.html')

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Text-to-SQL Web Application')
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=int(os.environ.get('FLASK_PORT', 5000)),
        help='Port to run the web application on (default: 5000, can also set FLASK_PORT env var)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default=os.environ.get('FLASK_HOST', '0.0.0.0'),
        help='Host to bind the web application to (default: 0.0.0.0, can also set FLASK_HOST env var)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true',
        help='Enable debug mode (default: False, can also set FLASK_DEBUG env var)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=os.environ.get('OLLAMA_MODEL_NAME', 'text-to-sql'),
        help='Name of the Ollama model to use (default: text-to-sql, can also set OLLAMA_MODEL_NAME env var)'
    )
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()
    
    # Update Ollama manager with specified model name
    try:
        ollama_manager = OllamaManager(args.model_name)
        logger.info(f"Using Ollama model: {args.model_name}")
    except Exception as e:
        logger.error(f"Failed to initialize Ollama manager with model '{args.model_name}': {e}")
        ollama_manager = None
    
    # Check if templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
        logger.info("Created templates directory")
    
    if not os.path.exists('static'):
        os.makedirs('static')
        logger.info("Created static directory")
    
    # Print startup information
    print("🚀 Text-to-SQL Web Application")
    print("=" * 40)
    print(f"🌐 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"🤖 Model: {args.model_name}")
    print(f"🐛 Debug: {args.debug}")
    print(f"📍 URL: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
    print("🛑 Press Ctrl+C to stop")
    print("-" * 40)
    
    # Run the application
    app.run(debug=args.debug, host=args.host, port=args.port)
