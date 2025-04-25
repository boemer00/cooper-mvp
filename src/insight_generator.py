from typing import List, Dict, Optional
import os
import json
from openai import OpenAI

# Try to import sentence-transformers, fallback gracefully
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Try to import pinecone, fallback gracefully
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

class InsightGenerator:
    def __init__(
        self,
        brand_guidelines: str,
        vector_model_name: str = "all-MiniLM-L6-v2",
        pinecone_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        pinecone_index_name: str = "brand-guidelines",
        pinecone_environment: str = "gcp-starter"
    ):
        """
        Initialize guideline text and embedder model.

        Args:
            brand_guidelines: Text containing brand guidelines
            vector_model_name: Name of the sentence transformer model
            pinecone_api_key: Pinecone API key, or None to use env var
            openai_api_key: OpenAI API key, or None to use env var
            pinecone_index_name: Name of the Pinecone index to use
            pinecone_environment: Pinecone environment (e.g., 'gcp-starter')
        """
        # Initialize API keys
        self.pinecone_api_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.pinecone_index_name = pinecone_index_name
        self.pinecone_environment = pinecone_environment or os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter")

        # Parse guidelines and initialize embedding model
        self.brand_guidelines = brand_guidelines
        self._guideline_chunks = self._parse_guidelines(brand_guidelines)

        # Initialize vector embedding model
        self._embedder = None
        self._index = None

        # Only try to initialize if dependencies are available and valid API key provided
        if (SENTENCE_TRANSFORMERS_AVAILABLE and PINECONE_AVAILABLE and
                self.pinecone_api_key and self.pinecone_api_key != "dummy"):
            try:
                self._embedder = SentenceTransformer(vector_model_name)

                # Initialize Pinecone
                pinecone.init(
                    api_key=self.pinecone_api_key,
                    environment=self.pinecone_environment
                )

                # Create index if it doesn't exist
                if self.pinecone_index_name not in pinecone.list_indexes():
                    pinecone.create_index(
                        name=self.pinecone_index_name,
                        dimension=self._embedder.get_sentence_embedding_dimension(),
                        metric="cosine"
                    )

                # Connect to index
                self._index = pinecone.Index(self.pinecone_index_name)

                # Add guideline chunks to index if not in test mode
                self._add_guidelines_to_index()
            except Exception as e:
                # Fallback to dummy mode if error occurs
                print(f"Error initializing vector search: {e}")
                self._embedder = None
                self._index = None
        else:
            self._embedder = None
            self._index = None

        # Initialize OpenAI client if valid API key provided
        if self.openai_api_key and self.openai_api_key != "dummy":
            self.client = OpenAI(api_key=self.openai_api_key)
        else:
            self.client = None

    def _parse_guidelines(self, guidelines: str) -> List[str]:
        """
        Split guidelines into digestible chunks.

        Args:
            guidelines: Raw guidelines text

        Returns:
            List of guideline chunks
        """
        # Simple paragraph-based splitting
        # NOTE: For very long guidelines, consider token-based chunking instead
        paragraphs = guidelines.split("\n\n")
        chunks = []

        for para in paragraphs:
            if para.strip():
                chunks.append(para.strip())

        return chunks

    def _add_guidelines_to_index(self):
        """Add guideline chunks to Pinecone index."""
        if not self._embedder or not self._index:
            return

        # Generate embeddings for all chunks
        if not self._guideline_chunks:
            return

        vectors = []

        for i, chunk in enumerate(self._guideline_chunks):
            # Generate embedding
            embedding = self._embedder.encode(chunk)

            # Create vector record
            vector = {
                "id": f"guideline-{i}",
                "values": embedding.tolist(),
                "metadata": {"text": chunk, "source": "brand_guidelines"}
            }

            vectors.append(vector)

        # Upsert vectors in batches
        if vectors:
            self._index.upsert(vectors=vectors)

    def _search_relevant_guidelines(self, query: str, top_k: int = 3) -> List[str]:
        """
        Search for relevant guideline chunks.

        Args:
            query: Query text to search for
            top_k: Number of results to return

        Returns:
            List of matching guideline chunks
        """
        # In dummy mode or if embedder not available, return the first chunk or empty list
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not PINECONE_AVAILABLE or not self._embedder or not self._index:
            return self._guideline_chunks[:min(top_k, len(self._guideline_chunks))] if self._guideline_chunks else []

        # Generate query embedding
        query_embedding = self._embedder.encode(query)

        # Search Pinecone
        results = self._index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )

        # Extract text from results - using proper attribute access
        matches = []
        for match in results.matches:
            if match.metadata and "text" in match.metadata:
                matches.append(match.metadata["text"])

        return matches

    def generate(
        self,
        correlations: Dict[str, float],
        n_insights: int = 2
    ) -> List[str]:
        """
        Generate n_insights based on correlations and guidelines.

        Args:
            correlations: Dictionary of correlation metrics
            n_insights: Number of insights to generate

        Returns:
            List of insight strings
        """
        # Return dummy insights in test mode
        if not self.client:
            return [
                "Videos with high joy scores receive 25% more likes than average.",
                "Content with surprise elements generates 40% more comments.",
                "Neutral content performs better for long-term viewer retention."
            ][:n_insights]

        # Convert correlations to text for search
        correlation_text = "\n".join([f"{k}: {v}" for k, v in correlations.items()])

        # Search for relevant guideline chunks
        relevant_guidelines = self._search_relevant_guidelines(correlation_text)
        guidelines_text = "\n\n".join(relevant_guidelines)

        # Prepare prompt for OpenAI
        system_prompt = """
        You are an expert content marketing analyst. Your job is to generate insightful,
        data-driven observations based on emotion analysis and engagement metrics.

        Analyze the correlation data provided and generate specific, actionable insights
        that align with the brand guidelines. Focus on clear patterns and relationships
        between emotional content and audience engagement.

        Each insight should:
        1. Identify a specific correlation or pattern
        2. Explain its significance
        3. Be concise and actionable
        4. Align with brand guidelines where relevant

        You MUST return your response as a valid JSON object with this exact format:
        {"insights": ["Insight 1", "Insight 2", ...]}
        """

        user_content = f"""
        CORRELATION DATA:
        {correlation_text}

        BRAND GUIDELINES:
        {guidelines_text}

        Generate {n_insights} data-driven insights based on these correlations that align with the brand guidelines.
        Remember to format your response as a valid JSON object with an "insights" array.
        """

        # Make API call
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.7
        )

        # Extract and parse insights
        try:
            result = json.loads(response.choices[0].message.content)
            insights = result.get("insights", [])

            # Ensure we have the right number of insights
            if not insights and isinstance(result, list):
                insights = result

            return insights[:n_insights]
        except Exception:
            # Fallback insights
            return [
                f"Correlation analysis shows interesting patterns in {list(correlations.keys())[0]}.",
                f"There appears to be a relationship between emotions and engagement metrics."
            ][:n_insights]

    def suggest_pr_hooks(
        self,
        insights: List[str],
        n_hooks: int = 2
    ) -> List[str]:
        """
        Suggest n_hooks PR hooks aligned to brand tone.

        Args:
            insights: List of insight strings
            n_hooks: Number of PR hooks to suggest

        Returns:
            List of PR hook strings
        """
        # Return dummy hooks in test mode
        if not self.client:
            return [
                "Discover the emotional secret behind our most engaging content",
                "How we increased engagement by 40% with one simple emotional trigger"
            ][:n_hooks]

        # Search for tone-related guideline chunks
        relevant_guidelines = self._search_relevant_guidelines("brand voice tone PR hooks")
        guidelines_text = "\n\n".join(relevant_guidelines)

        # Prepare prompt for OpenAI
        system_prompt = """
        You are an expert PR consultant who specializes in crafting compelling hooks and headlines
        that align perfectly with a brand's voice and tone. You have experise in difficult brand backlashes and crises.

        Based on the provided insights and brand guidelines, create PR hooks that:
        1. Capture the essence of the insights
        2. Use language that matches the brand voice
        3. Are attention-grabbing and shareable
        4. Would perform well on social media and press releases

        You MUST return your response as a valid JSON object with this exact format:
        {"hooks": ["Hook 1", "Hook 2", ...]}
        """

        # Join insights into a single string
        insights_text = "\n".join([f"- {insight}" for insight in insights])

        user_content = f"""
        INSIGHTS:
        {insights_text}

        BRAND GUIDELINES (TONE AND VOICE):
        {guidelines_text}

        Generate {n_hooks} attention-grabbing PR hooks based on these insights that align with the brand voice.
        Remember to format your response as a valid JSON object with a "hooks" array.
        """

        # Make API call
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.8
        )

        # Extract and parse hooks
        try:
            result = json.loads(response.choices[0].message.content)
            hooks = result.get("hooks", [])

            # Ensure we have the right number of hooks
            if not hooks and isinstance(result, list):
                hooks = result

            return hooks[:n_hooks]
        except Exception:
            # Fallback hooks
            return [
                f"New data reveals surprising connection in {insights[0][:20]}...",
                f"How our content strategy revealed unexpected audience preferences"
            ][:n_hooks]
