import pytest
from src.text_emotion_analyzer import TextEmotionAnalyzer
from unittest.mock import patch, MagicMock

def test_analyze_returns_scores_for_sample_texts():
    analyzer = TextEmotionAnalyzer(api_key="dummy")

    # Test the dummy mode directly
    result = analyzer.analyze(["I love this!", "This is sad"])

    # Assertions
    assert isinstance(result, dict)
    assert all(isinstance(k, str) for k in result.keys())
    assert all(isinstance(v, float) for v in result.values())
    assert all(0 <= v <= 1 for v in result.values())
    assert "joy" in result
    assert "sadness" in result

@pytest.mark.parametrize("texts", [
    ["I am happy today!"],
    ["I'm feeling sad and disappointed."],
    ["I'm angry about what happened."],
    ["I'm happy about the news but worried about the implications."],
])
def test_analyze_with_openai_api(texts, monkeypatch):
    # Create an analyzer with mocked OpenAI client
    analyzer = TextEmotionAnalyzer(api_key="test_key")

    # Create mock response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = """
    {
        "joy": 0.6,
        "sadness": 0.1,
        "anger": 0.05,
        "fear": 0.05,
        "surprise": 0.1,
        "disgust": 0.0,
        "neutral": 0.1
    }
    """

    # Mock the client.chat.completions.create method
    mock_completions = MagicMock(return_value=mock_response)
    mock_chat = MagicMock()
    mock_chat.completions.create = mock_completions
    analyzer.client = MagicMock()
    analyzer.client.chat = mock_chat

    # Call the analyze method
    result = analyzer.analyze(texts)

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
