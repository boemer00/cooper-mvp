import pytest
from src.audio_emotion_analyzer import AudioEmotionAnalyzer
from unittest.mock import patch, MagicMock
import os

def test_analyze_returns_scores_for_dummy_audio(tmp_path):
    # Create dummy WAV file
    audio_file = tmp_path / "dummy.wav"
    audio_file.write_bytes(b"RIFF....WAVEfmt ")

    analyzer = AudioEmotionAnalyzer(api_key="dummy", openai_api_key="dummy")

    # Test the dummy mode directly
    result = analyzer.analyze(str(audio_file))

    # Assertions
    assert isinstance(result, dict)
    assert all(isinstance(k, str) for k in result.keys())
    assert all(isinstance(v, float) for v in result.values())
    assert all(0 <= v <= 1 for v in result.values())
    assert "joy" in result
    assert "sadness" in result

def test_transcribe_audio_with_openai(tmp_path):
    # Create dummy WAV file
    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"RIFF....WAVEfmt ")

    # Create analyzer with mocked OpenAI client
    analyzer = AudioEmotionAnalyzer(api_key="test_key", openai_api_key="test_key")

    # Mock transcription response
    mock_transcript = MagicMock()
    mock_transcript.text = "This is a test transcript with positive emotions."

    # Mock the client.audio.transcriptions.create method
    mock_transcription = MagicMock(return_value=mock_transcript)
    analyzer.client = MagicMock()
    analyzer.client.audio = MagicMock()
    analyzer.client.audio.transcriptions = MagicMock()
    analyzer.client.audio.transcriptions.create = mock_transcription

    # Call the transcribe method
    result = analyzer._transcribe_audio(str(audio_file))

    # Assertions
    assert isinstance(result, str)
    assert "test transcript" in result
    assert mock_transcription.called

def test_analyze_with_mocked_api(tmp_path):
    # Create dummy WAV file
    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"RIFF....WAVEfmt ")

    # Create analyzer with mocked services
    analyzer = AudioEmotionAnalyzer(api_key="test_key", openai_api_key="test_key")

    # Mock transcribe_audio to return a fixed transcript
    with patch.object(analyzer, '_transcribe_audio', return_value="I'm feeling excited!"):
        # Create mock chat completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = """
        {
            "joy": 0.7,
            "sadness": 0.05,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.15,
            "disgust": 0.0,
            "neutral": 0.1
        }
        """

        # Mock the chat completions create method
        mock_completions = MagicMock(return_value=mock_response)
        mock_chat = MagicMock()
        mock_chat.completions = MagicMock()
        mock_chat.completions.create = mock_completions
        analyzer.client = MagicMock()
        analyzer.client.chat = mock_chat

        # Call the analyze method
        result = analyzer.analyze(str(audio_file))

        # Assertions
        assert isinstance(result, dict)
        assert all(isinstance(k, str) for k in result.keys())
        assert all(isinstance(v, float) for v in result.values())
        assert all(0 <= v <= 1 for v in result.values())
        assert mock_completions.called

        # Check the expected emotions are present
        assert "joy" in result
        assert "sadness" in result
        assert "anger" in result
        assert "fear" in result
        assert "surprise" in result
