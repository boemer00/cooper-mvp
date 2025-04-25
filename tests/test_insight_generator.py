import pytest
from unittest.mock import patch, MagicMock
from src.insight_generator import InsightGenerator

@pytest.fixture
def dummy_guidelines():
    """Fixture for dummy brand guidelines."""
    return """
    Brand Voice Guidelines

    Our brand voice is confident, friendly, and knowledgeable. We speak directly to our audience
    as trusted experts, avoiding jargon and overly technical language.

    Tone principles:
    - Be conversational but professional
    - Use active voice and present tense
    - Focus on benefits, not features
    - Be concise and clear

    Visual Guidelines

    Our visual identity uses bright colors, clean typography, and authentic photography.
    Avoid stock photos that feel staged or inauthentic.
    """

@pytest.fixture
def dummy_correlations():
    """Fixture for dummy correlation data."""
    return {
        "joy_vs_likes": 2.5,
        "surprise_vs_comments": 3.2,
        "fear_vs_shares": 0.4,
        "neutral_vs_views": 1.0
    }

def test_init_with_dummy_mode():
    """Test that InsightGenerator initializes correctly in dummy mode."""
    ig = InsightGenerator(
        brand_guidelines="Test guidelines",
        pinecone_api_key="dummy",
        openai_api_key="dummy"
    )

    assert ig.brand_guidelines == "Test guidelines"
    assert ig._guideline_chunks == ["Test guidelines"]
    assert ig._embedder is None
    assert ig._index is None
    assert ig.client is None

def test_init_with_custom_environment():
    """Test custom Pinecone environment is set correctly."""
    ig = InsightGenerator(
        brand_guidelines="Test guidelines",
        pinecone_api_key="dummy",
        openai_api_key="dummy",
        pinecone_environment="custom-env"
    )

    assert ig.pinecone_environment == "custom-env"

def test_parse_guidelines():
    """Test that guidelines are parsed correctly."""
    guidelines = """
    Paragraph 1

    Paragraph 2

    Paragraph 3
    """

    ig = InsightGenerator(
        brand_guidelines=guidelines,
        pinecone_api_key="dummy",
        openai_api_key="dummy"
    )

    assert len(ig._guideline_chunks) == 3
    assert "Paragraph 1" in ig._guideline_chunks[0]
    assert "Paragraph 2" in ig._guideline_chunks[1]
    assert "Paragraph 3" in ig._guideline_chunks[2]

def test_generate_insights_dummy_mode(dummy_correlations):
    """Test that generate returns dummy insights in dummy mode."""
    ig = InsightGenerator(
        brand_guidelines="Test guidelines",
        pinecone_api_key="dummy",
        openai_api_key="dummy"
    )

    insights = ig.generate(dummy_correlations, n_insights=2)

    assert len(insights) == 2
    assert isinstance(insights[0], str)
    assert isinstance(insights[1], str)

def test_generate_insights_none_key(dummy_correlations):
    """Test that generate handles None API key."""
    ig = InsightGenerator(
        brand_guidelines="Test guidelines",
        pinecone_api_key=None,
        openai_api_key=None
    )

    insights = ig.generate(dummy_correlations, n_insights=2)

    assert len(insights) == 2
    assert isinstance(insights[0], str)
    assert isinstance(insights[1], str)

def test_suggest_pr_hooks_dummy_mode():
    """Test that suggest_pr_hooks returns dummy hooks in dummy mode."""
    ig = InsightGenerator(
        brand_guidelines="Test guidelines",
        pinecone_api_key="dummy",
        openai_api_key="dummy"
    )

    insights = ["Insight 1", "Insight 2"]
    hooks = ig.suggest_pr_hooks(insights, n_hooks=2)

    assert len(hooks) == 2
    assert isinstance(hooks[0], str)
    assert isinstance(hooks[1], str)

@pytest.mark.parametrize("n_insights", [1, 2, 3])
def test_generate_respects_n_insights(dummy_correlations, n_insights):
    """Test that generate returns the requested number of insights."""
    ig = InsightGenerator(
        brand_guidelines="Test guidelines",
        pinecone_api_key="dummy",
        openai_api_key="dummy"
    )

    insights = ig.generate(dummy_correlations, n_insights=n_insights)
    assert len(insights) == n_insights

@pytest.mark.parametrize("mock_client_return", [
    '{"insights": ["Mocked insight 1", "Mocked insight 2"]}'
])
def test_generate_with_openai(dummy_guidelines, dummy_correlations, mock_client_return):
    """Test generate with direct OpenAI mock."""
    # Create mock OpenAI client
    mock_client = MagicMock()

    # Setup completion response
    mock_message = MagicMock()
    mock_message.content = mock_client_return

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create.return_value = mock_response

    # Create instance
    ig = InsightGenerator(
        brand_guidelines=dummy_guidelines,
        pinecone_api_key="dummy",
        openai_api_key="fake_key"
    )

    # Set our mock client
    ig.client = mock_client

    # Test generate
    insights = ig.generate(dummy_correlations, n_insights=2)

    # Verify results
    assert len(insights) == 2
    assert insights[0] == "Mocked insight 1"
    assert insights[1] == "Mocked insight 2"
    assert mock_client.chat.completions.create.called

@pytest.mark.parametrize("mock_client_return", [
    '{"hooks": ["Mocked PR hook 1", "Mocked PR hook 2"]}'
])
def test_suggest_pr_hooks_with_openai(dummy_guidelines, mock_client_return):
    """Test suggest_pr_hooks with direct OpenAI mock."""
    # Create mock OpenAI client
    mock_client = MagicMock()

    # Setup completion response
    mock_message = MagicMock()
    mock_message.content = mock_client_return

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create.return_value = mock_response

    # Create instance
    ig = InsightGenerator(
        brand_guidelines=dummy_guidelines,
        pinecone_api_key="dummy",
        openai_api_key="fake_key"
    )

    # Set our mock client
    ig.client = mock_client

    # Test suggest_pr_hooks
    hooks = ig.suggest_pr_hooks(["Test insight 1", "Test insight 2"], n_hooks=2)

    # Verify results
    assert len(hooks) == 2
    assert hooks[0] == "Mocked PR hook 1"
    assert hooks[1] == "Mocked PR hook 2"
    assert mock_client.chat.completions.create.called
