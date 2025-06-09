import pandas as pd
import requests
import json
import time
import logging
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabricksConfig:
    """Configuration for Databricks API"""
    api_url: str
    api_key: str
    model_name: str = "databricks-meta-llama-3-1-70b-instruct"
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 30

class DatabricksLLMClient:
    """Client for interacting with Databricks LLM API"""
    
    def __init__(self, config: DatabricksConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    def expand_text(self, text: str, custom_prompt: Optional[str] = None) -> str:
        """
        Expand a short text into a detailed summary
        
        Args:
            text: Short text to expand
            custom_prompt: Optional custom prompt template
            
        Returns:
            Expanded text summary
        """
        if custom_prompt:
            prompt = custom_prompt.format(text=text)
        else:
            prompt = self._create_default_prompt(text)
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        try:
            response = requests.post(
                f"{self.config.api_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            expanded_text = result["choices"][0]["message"]["content"].strip()
            
            logger.info(f"Successfully expanded text of length {len(text)} to {len(expanded_text)}")
            return expanded_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return f"Error expanding text: {str(e)}"
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format: {e}")
            return "Error: Unexpected response format from API"
    
    def _create_default_prompt(self, text: str) -> str:
        """Create default prompt for text expansion"""
        return f"""
Please expand the following short text into a comprehensive, detailed summary. 
Add context, explanations, and relevant details while maintaining the original meaning and intent.
Make it informative and well-structured.

Short text: "{text}"

Expanded summary:
"""

class TextExpansionProcessor:
    """Main processor for expanding text summaries from DataFrame"""
    
    def __init__(self, databricks_config: DatabricksConfig):
        self.llm_client = DatabricksLLMClient(databricks_config)
        
    def process_dataframe(
        self, 
        df: pd.DataFrame, 
        text_column: str = "notes",
        output_column: str = "expanded_summary",
        custom_prompt: Optional[str] = None,
        batch_size: int = 5,
        max_workers: int = 3
    ) -> pd.DataFrame:
        """
        Process DataFrame to generate expanded summaries
        
        Args:
            df: Input DataFrame
            text_column: Column containing short text
            output_column: Column name for expanded summaries
            custom_prompt: Optional custom prompt template
            batch_size: Number of texts to process in each batch
            max_workers: Number of concurrent workers
            
        Returns:
            DataFrame with expanded summaries
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        df_copy = df.copy()
        texts = df_copy[text_column].fillna("").tolist()
        
        logger.info(f"Processing {len(texts)} texts with {max_workers} workers")
        
        expanded_summaries = []
        
        # Process in batches to avoid overwhelming the API
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self._process_batch(
                batch, custom_prompt, max_workers
            )
            expanded_summaries.extend(batch_results)
            
            # Add delay between batches to respect rate limits
            if i + batch_size < len(texts):
                time.sleep(1)
        
        df_copy[output_column] = expanded_summaries
        return df_copy
    
    def _process_batch(
        self, 
        texts: List[str], 
        custom_prompt: Optional[str],
        max_workers: int
    ) -> List[str]:
        """Process a batch of texts concurrently"""
        results = [None] * len(texts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.llm_client.expand_text, text, custom_prompt): i
                for i, text in enumerate(texts)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    logger.error(f"Error processing text at index {index}: {e}")
                    results[index] = f"Error: {str(e)}"
        
        return results

def setup_databricks_config() -> DatabricksConfig:
    """
    Setup Databricks configuration from environment variables or manual input
    """
    api_url = os.getenv("DATABRICKS_API_URL")
    api_key = os.getenv("DATABRICKS_API_KEY")
    
    if not api_url:
        api_url = input("Enter Databricks API URL (e.g., https://your-workspace.databricks.com/serving-endpoints): ")
    
    if not api_key:
        api_key = input("Enter Databricks API Key: ")
    
    return DatabricksConfig(
        api_url=api_url,
        api_key=api_key,
        model_name="databricks-meta-llama-3-1-70b-instruct",  # Adjust as needed
        max_tokens=1000,
        temperature=0.7
    )

def main():
    """Main function to demonstrate usage"""
    
    # Example DataFrame creation (replace with your actual sample_df)
    sample_data = {
        'notes': [
            "Meeting about Q4 budget",
            "Customer complaint about delivery delay",
            "New product launch discussion",
            "Team performance review",
            "Marketing campaign results"
        ]
    }
    sample_df = pd.DataFrame(sample_data)
    
    print("Sample DataFrame:")
    print(sample_df)
    print("\n" + "="*50 + "\n")
    
    # Setup configuration
    config = setup_databricks_config()
    
    # Initialize processor
    processor = TextExpansionProcessor(config)
    
    # Custom prompt example (optional)
    custom_prompt = """
    Expand the following brief note into a detailed, professional summary. 
    Include potential context, implications, and relevant details that would be 
    helpful for someone reading this later.
    
    Brief note: "{text}"
    
    Detailed summary:
    """
    
    try:
        # Process the DataFrame
        result_df = processor.process_dataframe(
            sample_df,
            text_column="notes",
            output_column="expanded_summary",
            custom_prompt=custom_prompt,
            batch_size=3,
            max_workers=2
        )
        
        print("Results:")
        for idx, row in result_df.iterrows():
            print(f"\nOriginal: {row['notes']}")
            print(f"Expanded: {row['expanded_summary']}")
            print("-" * 80)
            
        # Save results
        result_df.to_csv("expanded_summaries.csv", index=False)
        print(f"\nResults saved to 'expanded_summaries.csv'")
        
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise

# Alternative function for simple single text expansion
def expand_single_text(text: str, config: DatabricksConfig) -> str:
    """
    Expand a single text using Databricks API
    
    Args:
        text: Text to expand
        config: Databricks configuration
        
    Returns:
        Expanded text
    """
    client = DatabricksLLMClient(config)
    return client.expand_text(text)

if __name__ == "__main__":
    main()
