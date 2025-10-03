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
    
    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        """
        Initialize the LLM generator.
        
        Args:
            model: Model name to use (default: gemini-2.5-flash for speed and cost efficiency)
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
            
            # If blocked by safety, try to get partial content or provide helpful error
            if finish_reason == 2:  # SAFETY
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
            
            # Check other finish reasons
            if finish_reason not in [1, 0]:  # 1=STOP, 0=UNSPECIFIED (success)
                finish_reasons = {
                    3: "RECITATION (copyright)",
                    4: "OTHER"
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
        code = self.generate_text(prompt, temperature=0.2, max_tokens=4000)
        
        # Clean up code if it's wrapped in markdown
        code = self._extract_code_from_markdown(code)
        
        return code
    
    def _extract_code_from_markdown(self, text: str) -> str:
        """
        Extract Python code from markdown code blocks if present.
        
        Args:
            text: Potentially markdown-formatted text
            
        Returns:
            Clean Python code
        """
        # Look for ```python ... ``` or ``` ... ``` blocks
        pattern = r'```(?:python)?\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            # Return the first (or largest) code block
            return max(matches, key=len).strip()
        
        # If no markdown blocks found, return as-is
        return text.strip()


def create_generator(model: str = "gemini-2.5-flash", api_key: Optional[str] = None) -> LLMGenerator:
    """
    Factory function to create an LLM generator.
    
    Args:
        model: Model name (default: gemini-2.5-flash)
        api_key: Gemini API key
        
    Returns:
        LLMGenerator instance
    """
    return LLMGenerator(model=model, api_key=api_key)
