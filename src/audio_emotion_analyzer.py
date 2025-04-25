from typing import Dict, Optional
import os
import json
import tempfile
from pathlib import Path
from openai import OpenAI

class AudioEmotionAnalyzer:
    def __init__(self, api_key: Optional[str] = None, openai_api_key: Optional[str] = None):
        """
        Initialize audio emotion analyzer using AssemblyAI (or simulation)
        and OpenAI for transcript analysis.

        Args:
            api_key: AssemblyAI API key (or "dummy" for simulation)
            openai_api_key: OpenAI API key. If None, will use environment variable.
        """
        self.api_key = api_key or os.environ.get("ASSEMBLYAI_API_KEY")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

        # Configure OpenAI client
        if self.openai_api_key != "dummy":
            self.client = OpenAI(api_key=self.openai_api_key)
        else:
            self.client = None

    def _transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file using OpenAI's Whisper API or simulate.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        # Dummy mode or simulation
        if self.api_key == "dummy":
            # Return a simulated transcript
            return "I'm really excited about this project. It's going to be amazing!"

        # Use OpenAI for transcription in this implementation
        # In a real implementation, you might use AssemblyAI
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                return transcript.text
        except Exception as e:
            # Fallback in case of errors
            return "Error transcribing audio. Using fallback text for analysis."

    def analyze(self, audio_path: str) -> Dict[str, float]:
        """
        Given a path to an audio file (or simulated), returns a mapping
        from emotion label → confidence score.

        Args:
            audio_path: Path to audio file to analyze

        Returns:
            Dictionary mapping emotion labels to confidence scores
        """
        # Return dummy data for testing
        if self.api_key == "dummy" and self.openai_api_key == "dummy":
            return {
                "joy": 0.7,
                "sadness": 0.1,
                "anger": 0.1,
                "fear": 0.05,
                "surprise": 0.05,
                "disgust": 0.0,
                "neutral": 0.0
            }

        # Transcribe audio to text
        transcript = self._transcribe_audio(audio_path)

        # If we have no transcript or in dummy mode, return dummy data
        if not transcript:
            return {
                "neutral": 1.0,
                "joy": 0.0,
                "sadness": 0.0,
                "anger": 0.0,
                "fear": 0.0,
                "surprise": 0.0,
                "disgust": 0.0
            }

        # Use OpenAI to analyze emotions in the transcript
        system_prompt = """
        You are an expert in emotion analysis. Analyze the following audio transcript and determine the emotional content.
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
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Audio transcript: {transcript}"}
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
                "neutral": 1.0,
                "joy": 0.0,
                "sadness": 0.0,
                "anger": 0.0,
                "fear": 0.0,
                "surprise": 0.0,
                "disgust": 0.0
            }
