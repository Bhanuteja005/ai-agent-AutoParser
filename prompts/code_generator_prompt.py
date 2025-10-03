"""Code generator prompt template."""

CODE_GENERATOR_PROMPT = """You are an expert Python code generator for bank statement parsers.

**Task**: Generate a complete, working parser module for {bank_name} bank statements.

**Extraction Plan**:
{plan}

**Contract** (MUST follow exactly):
```python
def parse(pdf_path: str) -> pd.DataFrame:
    \"\"\"Parse the bank statement PDF and return a DataFrame matching the expected schema.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        DataFrame with columns: {columns}
        Each row represents one transaction.
    \"\"\"
    pass
```

**Expected Output Schema**:
- Columns (exact order): {columns}
- Sample data (first row):
{sample_row}

**Allowed Imports ONLY**:
```python
import re
from decimal import Decimal
from typing import List, Optional
import pandas as pd
from tools.pdf_helpers import extract_text, extract_tables_as_dfs, normalize_amount, clean_text
```

**Critical Requirements**:
1. The function MUST be named `parse` and take `pdf_path: str` as the only parameter
2. Return a pandas DataFrame with EXACT column names: {columns}
3. Column order must match exactly
4. Use type hints and include a detailed docstring
5. Handle errors gracefully (return empty DataFrame if parsing completely fails)
6. Use the helper functions from tools.pdf_helpers
7. NO external network calls
8. NO additional dependencies beyond what's listed
9. Clean and readable code with small functions
10. Add inline comments for complex logic

**Data Type Handling**:
- Dates: Parse appropriately (consider day-first format DD-MM-YYYY)
- Amounts: Convert to float, handle empty values as empty strings or appropriate defaults
- Text fields: Clean and strip whitespace
- Balance: Convert to float

**Testing**: Your code will be tested by:
```python
expected_df = pd.read_csv(csv_path)
actual_df = parse(pdf_path)
assert expected_df.equals(actual_df), "DataFrames don't match"
```

Generate the COMPLETE parser code below. Include all necessary functions.
Output ONLY valid Python code, no markdown, no explanations outside of code comments.
"""

CODE_GENERATOR_WITH_FIXES_PROMPT = """You are an expert Python code generator for bank statement parsers.

**Previous Attempt Failed**. Here's the error:

{failure_summary}

**Previous Code**:
```python
{previous_code}
```

**Task**: Generate a FIXED, working parser module for {bank_name} bank statements.

**Extraction Plan**:
{plan}

**Contract** (MUST follow exactly):
```python
def parse(pdf_path: str) -> pd.DataFrame:
    \"\"\"Parse the bank statement PDF and return a DataFrame matching the expected schema.\"\"\"
    pass
```

**Expected Output Schema**:
- Columns (exact order): {columns}
- Sample data (first row):
{sample_row}

**Allowed Imports ONLY**:
```python
import re
from decimal import Decimal
from typing import List, Optional
import pandas as pd
from tools.pdf_helpers import extract_text, extract_tables_as_dfs, normalize_amount, clean_text
```

**Fix the Specific Issues**:
{fix_suggestions}

**Critical Requirements**:
1. Function named `parse`, parameter `pdf_path: str`, returns `pd.DataFrame`
2. Exact column names: {columns}
3. Handle all edge cases identified in the error
4. Use tools.pdf_helpers functions
5. Clean, well-commented code

Generate the FIXED parser code below.
Output ONLY valid Python code, no markdown formatting, no explanations.
Start with imports, end with the parse function.
"""
