"""
Cooper Video Analysis Library.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set VERSION
__version__ = "0.1.0"

# Export all modules
from src.scraper import Scraper, VideoData
from src.video_finder import VideoFinder
from src.text_emotion_analyzer import TextEmotionAnalyzer
from src.audio_emotion_analyzer import AudioEmotionAnalyzer
from src.correlator import Correlator
from src.insight_generator import InsightGenerator
