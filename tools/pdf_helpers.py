"""
PDF helper utilities for extracting text and tables from bank statement PDFs.

Provides deterministic, tested functions that the agent can rely on when
generating parser code, reducing hallucination risk.
"""

import re
from decimal import Decimal, InvalidOperation
from typing import List, Optional

import pandas as pd
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract_text
from PyPDF2 import PdfReader


def extract_text(pdf_path: str, use_pdfminer: bool = True) -> str:
    """
    Extract all text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        use_pdfminer: If True, use pdfminer.six; otherwise use PyPDF2
        
    Returns:
        Extracted text as a single string
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If extraction fails
    """
    try:
        if use_pdfminer:
            text = pdfminer_extract_text(pdf_path)
            return text if text else ""
        else:
            # Fallback to PyPDF2
            reader = PdfReader(pdf_path)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n".join(text_parts)
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")


def extract_tables_as_dfs(pdf_path: str) -> List[pd.DataFrame]:
    """
    Extract all tables from a PDF as pandas DataFrames using pdfplumber.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of DataFrames, one per table found. Empty list if no tables found.
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If extraction fails
    """
    try:
        tables = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                if page_tables:
                    for table in page_tables:
                        if table and len(table) > 1:  # At least header + one row
                            # Convert to DataFrame
                            df = pd.DataFrame(table[1:], columns=table[0])
                            # Remove completely empty rows
                            df = df.dropna(how='all')
                            if not df.empty:
                                tables.append(df)
        return tables
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    except Exception as e:
        raise Exception(f"Failed to extract tables from PDF: {str(e)}")


def normalize_amount(amount_str: str) -> Optional[Decimal]:
    """
    Normalize an amount string to Decimal, handling various formats.
    
    Handles:
    - Comma thousands separators: "1,234.56" -> 1234.56
    - Currency symbols: "₹1234.56", "$1234.56" -> 1234.56
    - Negative amounts: "-123.45", "(123.45)" -> -123.45
    - Whitespace and special characters
    
    Args:
        amount_str: String representation of an amount
        
    Returns:
        Decimal value or None if parsing fails
    """
    if not amount_str or not isinstance(amount_str, str):
        return None
    
    # Clean the string
    cleaned = amount_str.strip()
    
    # Handle empty strings
    if not cleaned:
        return None
    
    # Remove currency symbols
    cleaned = re.sub(r'[₹$€£¥]', '', cleaned)
    
    # Handle parentheses as negative (accounting notation)
    is_negative = False
    if cleaned.startswith('(') and cleaned.endswith(')'):
        is_negative = True
        cleaned = cleaned[1:-1]
    
    # Remove commas (thousands separator)
    cleaned = cleaned.replace(',', '')
    
    # Remove any remaining whitespace
    cleaned = cleaned.strip()
    
    # Check for explicit negative sign
    if cleaned.startswith('-'):
        is_negative = True
        cleaned = cleaned[1:].strip()
    
    # Try to convert to Decimal
    try:
        value = Decimal(cleaned)
        return -value if is_negative else value
    except (InvalidOperation, ValueError):
        return None


def clean_text(text: str) -> str:
    """
    Clean extracted text by normalizing whitespace and removing artifacts.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive whitespace while preserving single newlines
    lines = []
    for line in text.split('\n'):
        cleaned_line = ' '.join(line.split())
        if cleaned_line:
            lines.append(cleaned_line)
    
    return '\n'.join(lines)


def extract_date_patterns(text: str, date_format: str = r'\d{2}-\d{2}-\d{4}') -> List[str]:
    """
    Extract all date patterns from text.
    
    Args:
        text: Text to search
        date_format: Regex pattern for date format (default: DD-MM-YYYY)
        
    Returns:
        List of date strings matching the pattern
    """
    if not text:
        return []
    
    dates = re.findall(date_format, text)
    return dates


def extract_transactions_from_text(
    text: str,
    date_pattern: str = r'\d{2}-\d{2}-\d{4}'
) -> List[dict]:
    """
    Extract transaction-like patterns from text using basic heuristics.
    
    This is a fallback when table extraction fails. It looks for lines
    containing dates followed by text (description) and numbers (amounts).
    
    Args:
        text: Extracted text from PDF
        date_pattern: Regex pattern to identify transaction dates
        
    Returns:
        List of dictionaries with 'line' and 'date' keys
    """
    transactions = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line contains a date
        dates = re.findall(date_pattern, line)
        if dates:
            transactions.append({
                'line': line,
                'date': dates[0]
            })
    
    return transactions
