"""Planner prompt template."""

PLANNER_PROMPT = """You are an expert planner for a PDF parsing agent.

**Goal**: Analyze a bank statement PDF and create a detailed extraction plan.

**Available Information**:
- Target bank: {bank_name}
- Expected CSV columns: {columns}
- Sample CSV rows (first 3):
{sample_rows}

**PDF Content Preview**:
{pdf_preview}

**Available Tools** (in tools/pdf_helpers.py):
1. extract_text(pdf_path) -> str: Extract all text from PDF
2. extract_tables_as_dfs(pdf_path) -> List[pd.DataFrame]: Extract tables using pdfplumber
3. normalize_amount(s: str) -> Decimal: Parse amount strings
4. clean_text(text: str) -> str: Clean extracted text

**Your Task**:
Create a concise extraction plan that includes:

1. **Data Location Strategy**: 
   - Is the data in a table format or raw text?
   - Which extraction method to use (tables vs text+regex)?

2. **Field Extraction Strategy** for each column:
   - How to extract each field (Date, Description, amounts, Balance)
   - Regex patterns needed (if any)
   - Data type conversions required

3. **Edge Cases to Handle**:
   - Multi-line descriptions
   - Negative amounts / debits/credits
   - Thousands separators in amounts
   - Date format (DD-MM-YYYY, etc.)
   - Empty values
   - Header/footer text to skip

4. **Validation Checks**:
   - Row count expectations
   - Column completeness
   - Data type validation

Output your plan as structured text with clear sections.
Be specific about regex patterns and pandas operations.
"""

PLANNER_WITH_FEEDBACK_PROMPT = """You are an expert planner for a PDF parsing agent.

**Previous Attempt Failed**. Here's what went wrong:

{failure_summary}

**Original Plan Was**:
{previous_plan}

**Goal**: Analyze the failure and create an IMPROVED extraction plan.

**Available Information**:
- Target bank: {bank_name}
- Expected CSV columns: {columns}
- Sample CSV rows (first 3):
{sample_rows}

**PDF Content Preview**:
{pdf_preview}

**Available Tools** (in tools/pdf_helpers.py):
1. extract_text(pdf_path) -> str
2. extract_tables_as_dfs(pdf_path) -> List[pd.DataFrame]
3. normalize_amount(s: str) -> Decimal
4. clean_text(text: str) -> str

**Your Task**:
Create a REVISED extraction plan that fixes the issues identified above.

Focus on:
1. Addressing the specific failure (error type, missing columns, wrong data types, etc.)
2. Proposing alternative extraction strategies if previous approach failed
3. Adding more validation and error handling
4. Being more specific about data transformations

Output your improved plan as structured text with clear sections.
"""
