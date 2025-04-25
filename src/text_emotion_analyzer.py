from typing import List, Dict, Optional
import os
import json
from openai import OpenAI

class TextEmotionAnalyzer:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize text emotion analyzer using OpenAI API.

        Args:
            api_key: OpenAI API key. If None, will use environment variable.
            model_name: OpenAI model to use for emotion analysis.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name

        # Configure OpenAI client
        if self.api_key != "dummy":
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def analyze(self, texts: List[str]) -> Dict[str, float]:
        """
        Given a list of strings (transcript or comments), returns a mapping
        from emotion label → confidence score (0.0-1.0).

        Args:
            texts: List of text strings to analyze

        Returns:
            Dictionary mapping emotion labels to confidence scores
        """
        # Return dummy data for testing
        if self.api_key == "dummy":
            return {
                "joy": 0.8,
                "sadness": 0.1,
                "anger": 0.05,
                "fear": 0.03,
                "surprise": 0.02
            }

        # Combine texts into a single analysis request
        combined_text = "\n".join(texts)

        # Create prompt for emotion analysis
        system_prompt = """
        You are an expert in emotion analysis. Analyze the following text and determine the emotional content.
        Return a JSON object with the following emotions and their corresponding confidence scores (0.0-1.0):
        - joy
        - sadness
        - anger
        - fear
        - surprise
        - disgust
        - neutral

        The scores should sum to 1.0.
        Only return the JSON object, nothing else.
        """

        # Make API call
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_text}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        # Extract and parse JSON from response
        try:
            result = json.loads(response.choices[0].message.content)
            # Ensure all values are floats
            for key in result:
                result[key] = float(result[key])
            return result
        except (json.JSONDecodeError, AttributeError, IndexError) as e:
            # Fallback in case of parsing errors
            return {
                "joy": 0.0,
                "sadness": 0.0,
                "anger": 0.0,
                "fear": 0.0,
                "surprise": 0.0,
                "disgust": 0.0,
                "neutral": 1.0
            }
