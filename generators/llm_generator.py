"""
LLM integration for generating parser code.

This module wraps the Google Gemini API and provides
a clean interface for the agent to generate plans and code.
"""

import os
import re
from typing import Optional

import google.generativeai as genai


class LLMGenerator:
    """Wrapper for LLM API calls to generate plans and parser code."""
    
    def __init__(self, model: str = "gemini-2.0-flash", api_key: Optional[str] = None):
        """
        Initialize the LLM generator.
        
        Args:
            model: Model name to use (default: gemini-2.0-flash for speed and reliability)
            api_key: Google Gemini API key (if None, reads from GEMINI_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Google Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(model)
    
    def generate_text(
        self, 
        prompt: str, 
        temperature: float = 0.3,
        max_tokens: int = 4000
    ) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature (0.0-2.0). Lower = more deterministic.
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated text
            
        Raises:
            Exception: If API call fails
        """
        try:
            # Configure generation settings
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            # Configure safety settings to be less restrictive for code generation
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]
            
            # Add system instruction to the prompt
            full_prompt = (
                "You are an expert Python developer specializing in PDF parsing and data extraction.\n\n"
                + prompt
            )
            
            response = self.client.generate_content(
                full_prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Check if response was blocked
            if not response.candidates:
                raise Exception("Response was blocked. No candidates returned.")
            
            candidate = response.candidates[0]
            
            # Check finish reason
            finish_reason = candidate.finish_reason
            
            # Finish reason codes: 0=UNSPECIFIED, 1=STOP (success), 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
            # If blocked by safety (3), try to get partial content
            if finish_reason == 3:  # SAFETY
                # Check if there's any partial content
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    try:
                        partial_text = ''.join([part.text for part in candidate.content.parts if hasattr(part, 'text')])
                        if partial_text:
                            return partial_text.strip()
                    except:
                        pass
                
                # Provide helpful error message
                safety_ratings = getattr(candidate, 'safety_ratings', [])
                blocked_categories = []
                for rating in safety_ratings:
                    if hasattr(rating, 'blocked') and rating.blocked:
                        category = getattr(rating, 'category', 'UNKNOWN')
                        blocked_categories.append(str(category))
                
                if blocked_categories:
                    raise Exception(f"Content filtered by safety system. Blocked categories: {', '.join(blocked_categories)}. Try rephrasing your prompt.")
                else:
                    raise Exception("Content filtered by safety system. Try rephrasing your prompt or using a different model.")
            
            # Check if we got valid content (finish_reason 0, 1, or 2 are acceptable)
            if finish_reason not in [0, 1, 2]:  # 0=UNSPECIFIED, 1=STOP, 2=MAX_TOKENS (all OK)
                finish_reasons = {
                    4: "RECITATION (copyright)",
                    5: "OTHER",
                    6: "BLOCKLIST",
                    7: "PROHIBITED_CONTENT",
                    8: "SPII"
                }
                reason = finish_reasons.get(finish_reason, f"UNKNOWN ({finish_reason})")
                raise Exception(f"Generation stopped: {reason}")
            
            return response.text.strip()
        
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")
    
    def generate_plan(self, prompt: str) -> str:
        """
        Generate an extraction plan.
        
        Args:
            prompt: Formatted planner prompt
            
        Returns:
            Plan text
        """
        return self.generate_text(prompt, temperature=0.3, max_tokens=2000)
    
    def generate_parser_code(self, prompt: str) -> str:
        """
        Generate parser code.
        
        Args:
            prompt: Formatted code generator prompt
            
        Returns:
            Python code as a string
        """
        # Add explicit instruction to output raw code
        enhanced_prompt = prompt + "\n\nIMPORTANT: Output raw Python code ONLY. Do NOT wrap in markdown code blocks (```). Start directly with 'import' statements."
        
        code = self.generate_text(enhanced_prompt, temperature=0.2, max_tokens=4000)
        
        # Clean up code if it's wrapped in markdown
        code = self._extract_code_from_markdown(code)
        
        # Final sanity check: remove any remaining markdown artifacts
        if code.startswith('```'):
            lines = code.split('\n')
            # Remove lines containing only ```
            lines = [l for l in lines if l.strip() not in ['```', '```python', '```py']]
            code = '\n'.join(lines).strip()
        
        return code
    
    def _extract_code_from_markdown(self, text: str) -> str:
        """
        Extract Python code from markdown code blocks if present.
        
        Args:
            text: Potentially markdown-formatted text
            
        Returns:
            Clean Python code
        """
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Look for ```python ... ``` or ``` ... ``` blocks with flexible newlines
        pattern = r'```(?:python)?\s*[\r\n]+(.*?)[\r\n]+```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            # Return the largest code block (in case there are multiple)
            code = max(matches, key=len).strip()
            return code
        
        # Check if text starts with markdown fence without finding matches
        if text.startswith('```'):
            # Remove first line and last line manually
            lines = text.split('\n')
            if len(lines) > 2:
                # Remove first line (```python or ```)
                lines = lines[1:]
                # Remove last line if it contains ```
                if lines and '```' in lines[-1]:
                    lines = lines[:-1]
                return '\n'.join(lines).strip()
        
        # If no markdown blocks found, return as-is
        return text


def create_generator(model: str = "gemini-2.0-flash", api_key: Optional[str] = None) -> LLMGenerator:
    """
    Factory function to create an LLM generator.
    
    Args:
        model: Model name (default: gemini-2.5-pro)
        api_key: Gemini API key
        
    Returns:
        LLMGenerator instance
    """
    return LLMGenerator(model=model, api_key=api_key)
