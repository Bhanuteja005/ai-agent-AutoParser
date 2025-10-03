#!/usr/bin/env python3
"""
Agent-as-Coder: Bank Statement Parser Generator

This agent uses LLM-driven code generation to create parsers for bank statement PDFs.
It follows a plan-code-test-fix loop with up to N attempts to generate working code.

Usage:
    python agent.py --target icici
    python agent.py --target icici --attempts 3 --model gpt-4o-mini
"""

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from generators.llm_generator import create_generator
from prompts.planner_prompt import PLANNER_PROMPT, PLANNER_WITH_FEEDBACK_PROMPT
from prompts.code_generator_prompt import (
    CODE_GENERATOR_PROMPT,
    CODE_GENERATOR_WITH_FIXES_PROMPT,
)
from tools.pdf_helpers import extract_text, extract_tables_as_dfs


# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CUSTOM_PARSERS_DIR = PROJECT_ROOT / "custom_parsers"
AGENT_REPORTS_DIR = PROJECT_ROOT / "agent_reports"


class ParserAgent:
    """
    Agent that generates bank statement parsers using LLM-driven code generation.
    
    The agent follows a plan-code-test-fix loop:
    1. Planner: Analyze PDF/CSV and create extraction plan
    2. Code Generator: Generate parser code based on plan
    3. Sandbox Runner: Write code and run tests
    4. Observer: Analyze test failures
    5. Repeat up to max_attempts times
    """
    
    def __init__(
        self,
        target_bank: str,
        max_attempts: int = 3,
        model: str = "gemini-2.0-flash",
        verbose: bool = True
    ):
        """
        Initialize the parser agent.
        
        Args:
            target_bank: Name of the target bank (e.g., 'icici')
            max_attempts: Maximum number of generation attempts
            model: LLM model to use (default: gemini-2.0-flash)
            verbose: Print detailed logs
        """
        self.target_bank = target_bank
        self.max_attempts = max_attempts
        self.model = model
        self.verbose = verbose
        
        # Initialize LLM generator
        self.generator = create_generator(model=model)
        
        # Paths
        self.bank_data_dir = DATA_DIR / target_bank
        self.parser_file = CUSTOM_PARSERS_DIR / f"{target_bank}_parser.py"
        self.report_file = AGENT_REPORTS_DIR / f"{target_bank}_report.txt"
        
        # State
        self.plan: Optional[str] = None
        self.previous_code: Optional[str] = None
        self.failure_history: list = []
        
    def log(self, message: str, level: str = "INFO"):
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{level}] {message}")
    
    def validate_inputs(self) -> bool:
        """
        Validate that required input files exist.
        
        Returns:
            True if validation passes, False otherwise
        """
        # Check if data directory exists
        if not self.bank_data_dir.exists():
            self.log(f"Data directory not found: {self.bank_data_dir}", "ERROR")
            return False
        
        # Check for PDF file
        pdf_files = list(self.bank_data_dir.glob("*.pdf"))
        if not pdf_files:
            self.log(f"No PDF files found in {self.bank_data_dir}", "ERROR")
            return False
        
        self.pdf_path = pdf_files[0]
        self.log(f"Found PDF: {self.pdf_path.name}")
        
        # Check for CSV file
        csv_files = list(self.bank_data_dir.glob("*.csv"))
        if not csv_files:
            self.log(f"No CSV files found in {self.bank_data_dir}", "ERROR")
            return False
        
        self.csv_path = csv_files[0]
        self.log(f"Found CSV: {self.csv_path.name}")
        
        return True
    
    def load_csv_schema(self) -> Tuple[list, pd.DataFrame]:
        """
        Load CSV and extract schema information.
        
        Returns:
            Tuple of (column_names, sample_dataframe)
        """
        df = pd.read_csv(self.csv_path)
        columns = df.columns.tolist()
        sample_df = df.head(3)
        
        return columns, sample_df
    
    def get_pdf_preview(self, max_chars: int = 2000) -> str:
        """
        Get a preview of PDF content for the planner.
        
        Args:
            max_chars: Maximum characters to include
            
        Returns:
            PDF content preview
        """
        try:
            # Try table extraction first
            tables = extract_tables_as_dfs(str(self.pdf_path))
            if tables:
                preview = f"PDF contains {len(tables)} table(s).\n\n"
                preview += "First table preview:\n"
                preview += tables[0].head(5).to_string()
                return preview[:max_chars]
        except Exception as e:
            self.log(f"Table extraction failed: {e}", "WARNING")
        
        # Fallback to text extraction
        try:
            text = extract_text(str(self.pdf_path))
            return text[:max_chars]
        except Exception as e:
            self.log(f"Text extraction failed: {e}", "WARNING")
            return "[PDF content could not be extracted]"
    
    def create_plan(self, attempt: int) -> str:
        """
        Create or update extraction plan.
        
        Args:
            attempt: Current attempt number
            
        Returns:
            Plan text
        """
        self.log(f"\n{'='*60}")
        self.log(f"Attempt {attempt}/{self.max_attempts}: Creating plan...")
        self.log(f"{'='*60}")
        
        columns, sample_df = self.load_csv_schema()
        pdf_preview = self.get_pdf_preview()
        
        if attempt == 1:
            # First attempt: create initial plan
            prompt = PLANNER_PROMPT.format(
                bank_name=self.target_bank,
                columns=", ".join(columns),
                sample_rows=sample_df.to_string(index=False),
                pdf_preview=pdf_preview
            )
        else:
            # Subsequent attempts: refine plan based on failure
            failure_summary = self.failure_history[-1] if self.failure_history else "Unknown error"
            
            prompt = PLANNER_WITH_FEEDBACK_PROMPT.format(
                failure_summary=failure_summary,
                previous_plan=self.plan or "None",
                bank_name=self.target_bank,
                columns=", ".join(columns),
                sample_rows=sample_df.to_string(index=False),
                pdf_preview=pdf_preview
            )
        
        plan = self.generator.generate_plan(prompt)
        self.plan = plan
        
        self.log("Plan created:")
        self.log("-" * 60)
        self.log(plan)
        self.log("-" * 60)
        
        return plan
    
    def generate_parser_code(self, attempt: int) -> str:
        """
        Generate parser code based on plan.
        
        Args:
            attempt: Current attempt number
            
        Returns:
            Generated Python code
        """
        self.log("\nGenerating parser code...")
        
        columns, sample_df = self.load_csv_schema()
        sample_row = sample_df.iloc[0].to_dict() if not sample_df.empty else {}
        
        if attempt == 1 or not self.previous_code:
            # First attempt: generate fresh code
            prompt = CODE_GENERATOR_PROMPT.format(
                bank_name=self.target_bank,
                plan=self.plan,
                columns=", ".join(columns),
                sample_row=str(sample_row)
            )
        else:
            # Subsequent attempts: fix previous code
            failure_summary = self.failure_history[-1] if self.failure_history else "Unknown error"
            
            # Generate fix suggestions based on error type
            fix_suggestions = self._generate_fix_suggestions(failure_summary)
            
            prompt = CODE_GENERATOR_WITH_FIXES_PROMPT.format(
                failure_summary=failure_summary,
                previous_code=self.previous_code,
                bank_name=self.target_bank,
                plan=self.plan,
                columns=", ".join(columns),
                sample_row=str(sample_row),
                fix_suggestions=fix_suggestions
            )
        
        code = self.generator.generate_parser_code(prompt)
        self.previous_code = code
        
        self.log("Code generated successfully")
        
        return code
    
    def _generate_fix_suggestions(self, failure_summary: str) -> str:
        """Generate specific fix suggestions based on error type."""
        suggestions = []
        
        failure_lower = failure_summary.lower()
        
        if "importerror" in failure_lower or "modulenotfounderror" in failure_lower:
            suggestions.append("- Check all imports are from allowed list")
            suggestions.append("- Ensure tools.pdf_helpers import is correct")
        
        if "column" in failure_lower or "keyerror" in failure_lower:
            suggestions.append("- Verify exact column names match CSV schema")
            suggestions.append("- Check for typos in column names")
            suggestions.append("- Ensure column order matches expected order")
        
        if "dtype" in failure_lower or "type" in failure_lower:
            suggestions.append("- Check data type conversions (str, float, etc.)")
            suggestions.append("- Handle empty values appropriately")
            suggestions.append("- Ensure numeric columns are properly converted")
        
        if "shape" in failure_lower or "row" in failure_lower:
            suggestions.append("- Check row filtering logic")
            suggestions.append("- Ensure all transactions are captured")
            suggestions.append("- Verify no duplicate or missing rows")
        
        if "equals" in failure_lower or "value" in failure_lower:
            suggestions.append("- Check value formatting (whitespace, case, etc.)")
            suggestions.append("- Verify date parsing format")
            suggestions.append("- Check amount/balance number formatting")
        
        if not suggestions:
            suggestions.append("- Review the error carefully")
            suggestions.append("- Check each extraction step")
            suggestions.append("- Add defensive error handling")
        
        return "\n".join(suggestions)
    
    def write_parser_file(self, code: str):
        """
        Write generated parser code to file.
        
        Args:
            code: Python code to write
        """
        # Ensure custom_parsers directory exists
        CUSTOM_PARSERS_DIR.mkdir(exist_ok=True)
        
        # Write code to file
        self.parser_file.write_text(code, encoding="utf-8")
        
        self.log(f"Parser written to: {self.parser_file}")
    
    def run_tests(self) -> Tuple[bool, str]:
        """
        Run pytest on the generated parser.
        
        Returns:
            Tuple of (success, output)
        """
        self.log("\nRunning tests...")
        
        test_name = f"tests/test_parsers.py::test_{self.target_bank}_parser"
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_name, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=PROJECT_ROOT
            )
            
            output = result.stdout + "\n" + result.stderr
            success = result.returncode == 0
            
            if success:
                self.log("✓ Tests passed!", "SUCCESS")
            else:
                self.log("✗ Tests failed", "ERROR")
                self.log("Test output:")
                self.log(output)
            
            return success, output
            
        except subprocess.TimeoutExpired:
            error_msg = "Test execution timed out (60s limit)"
            self.log(error_msg, "ERROR")
            return False, error_msg
            
        except Exception as e:
            error_msg = f"Test execution failed: {str(e)}\n{traceback.format_exc()}"
            self.log(error_msg, "ERROR")
            return False, error_msg
    
    def analyze_failure(self, test_output: str) -> str:
        """
        Analyze test failure and create summary for next iteration.
        
        Args:
            test_output: Output from pytest
            
        Returns:
            Failure summary
        """
        summary_parts = []
        
        # Extract error type
        if "ImportError" in test_output or "ModuleNotFoundError" in test_output:
            summary_parts.append("ERROR TYPE: Import Error")
            summary_parts.append("Issue: Failed to import required modules")
        elif "AttributeError" in test_output:
            summary_parts.append("ERROR TYPE: Attribute Error")
            summary_parts.append("Issue: Missing expected function or attribute")
        elif "KeyError" in test_output:
            summary_parts.append("ERROR TYPE: Key Error")
            summary_parts.append("Issue: Missing expected column or key")
        elif "AssertionError" in test_output:
            summary_parts.append("ERROR TYPE: Assertion Error")
            summary_parts.append("Issue: Output doesn't match expected result")
        elif "SyntaxError" in test_output:
            summary_parts.append("ERROR TYPE: Syntax Error")
            summary_parts.append("Issue: Invalid Python syntax in generated code")
        else:
            summary_parts.append("ERROR TYPE: Unknown")
        
        # Extract relevant error messages
        lines = test_output.split("\n")
        error_lines = []
        capture = False
        
        for line in lines:
            if "FAILED" in line or "ERROR" in line or "assert" in line.lower():
                capture = True
            if capture:
                error_lines.append(line)
                if len(error_lines) > 20:  # Limit lines
                    break
        
        if error_lines:
            summary_parts.append("\nERROR DETAILS:")
            summary_parts.append("\n".join(error_lines[:20]))
        
        summary = "\n".join(summary_parts)
        return summary
    
    def save_failure_report(self):
        """Save failure report to file."""
        AGENT_REPORTS_DIR.mkdir(exist_ok=True)
        
        report_lines = [
            f"Parser Generation Report: {self.target_bank}",
            "=" * 60,
            f"\nStatus: FAILED after {self.max_attempts} attempts",
            f"\nFinal Plan:",
            "-" * 60,
            self.plan or "No plan generated",
            "-" * 60,
            f"\nFailure History:",
        ]
        
        for i, failure in enumerate(self.failure_history, 1):
            report_lines.append(f"\nAttempt {i}:")
            report_lines.append(failure)
            report_lines.append("-" * 60)
        
        report = "\n".join(report_lines)
        self.report_file.write_text(report, encoding="utf-8")
        
        self.log(f"\nFailure report saved to: {self.report_file}", "INFO")
    
    def run(self) -> bool:
        """
        Run the agent to generate and test a parser.
        
        Returns:
            True if successful, False otherwise
        """
        self.log(f"\n{'='*60}")
        self.log(f"Agent-as-Coder: Parser Generator")
        self.log(f"Target: {self.target_bank}")
        self.log(f"Model: {self.model}")
        self.log(f"Max attempts: {self.max_attempts}")
        self.log(f"{'='*60}\n")
        
        # Validate inputs
        if not self.validate_inputs():
            return False
        
        # Main agent loop
        for attempt in range(1, self.max_attempts + 1):
            try:
                # Step 1: Create/update plan
                self.create_plan(attempt)
                
                # Step 2: Generate parser code
                code = self.generate_parser_code(attempt)
                
                # Step 3: Write code to file
                self.write_parser_file(code)
                
                # Step 4: Run tests
                success, test_output = self.run_tests()
                
                if success:
                    self.log(f"\n{'='*60}")
                    self.log(f"SUCCESS! Parser generated in {attempt} attempt(s)", "SUCCESS")
                    self.log(f"Parser file: {self.parser_file}")
                    self.log(f"{'='*60}\n")
                    return True
                
                # Step 5: Analyze failure
                failure_summary = self.analyze_failure(test_output)
                self.failure_history.append(failure_summary)
                
                if attempt < self.max_attempts:
                    self.log(f"\nAttempt {attempt} failed. Retrying with improved plan...")
                
            except Exception as e:
                error_msg = f"Unexpected error in attempt {attempt}: {str(e)}\n{traceback.format_exc()}"
                self.log(error_msg, "ERROR")
                self.failure_history.append(error_msg)
                
                if attempt < self.max_attempts:
                    self.log("Retrying...")
                    continue
        
        # All attempts failed
        self.log(f"\n{'='*60}")
        self.log(f"FAILED after {self.max_attempts} attempts", "ERROR")
        self.log(f"{'='*60}\n")
        
        self.save_failure_report()
        return False


def main():
    """Main entry point for the agent CLI."""
    parser = argparse.ArgumentParser(
        description="Agent-as-Coder: Generate bank statement parsers using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py --target icici
  python agent.py --target icici --attempts 5 --model gemini-2.5-pro
  
Environment:
  Set GEMINI_API_KEY environment variable for LLM access.
        """
    )
    
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target bank name (e.g., 'icici')"
    )
    
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Maximum number of generation attempts (default: 3)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="LLM model to use (default: gemini-2.0-flash)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Create and run agent
    agent = ParserAgent(
        target_bank=args.target,
        max_attempts=args.attempts,
        model=args.model,
        verbose=not args.quiet
    )
    
    success = agent.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
