from typing import List, Dict, Optional
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from src.video_finder import VideoFinder
from src.scraper import Scraper, ScrapeConfig, VideoData
from src.text_emotion_analyzer import TextEmotionAnalyzer
from src.audio_emotion_analyzer import AudioEmotionAnalyzer
from src.correlator import Correlator
from src.insight_generator import InsightGenerator

app = FastAPI(title="Cooper API", description="Video content analysis API")

class ChatResponse(BaseModel):
    videos: List[str]
    emotions: Dict[str, Dict[str, float]]
    correlations: Dict[str, float]
    insights: List[str]
    pr_hooks: List[str]

@app.get("/chat", response_model=ChatResponse)
def chat_endpoint(
    query: str = Query(..., description="Topic to analyze"),
    limit: int = Query(10, ge=1, le=20),
    url: Optional[str] = Query(None, description="Direct TikTok URL to analyze"),
) -> ChatResponse:
    """
    Orchestrates the full Cooper pipeline:
    VideoFinder → Scraper → EmotionAnalyzers → Correlator → InsightGenerator

    Supports either topic-based search or direct TikTok URL analysis.
    """
    # Step 1: Find videos based on query or direct URL
    video_finder = VideoFinder()
    video_urls = video_finder.get_videos(query, direct_url=url)[:limit]

    if not video_urls:
        raise HTTPException(status_code=404, detail="No videos found for query or invalid TikTok URL")

    # Step 2: Scrape video data
    try:
        scraper = Scraper()
        config = ScrapeConfig(
            commentsPerPost=10,
            excludePinnedPosts=True,
            maxRepliesPerComment=5,
            resultsPerPage=20,
            postURLs=video_urls
        )
        scraper.start_scrape(config)
        video_data_list = scraper.get_result()
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Scraping operation timed out")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error during scraping: {str(e)}")

    if not video_data_list:
        raise HTTPException(status_code=404, detail="Failed to scrape any video data")

    # Step 3: Analyze emotions in text and audio
    try:
        text_analyzer = TextEmotionAnalyzer()
        audio_analyzer = AudioEmotionAnalyzer()

        text_emotions = text_analyzer.analyze(video_data_list)
        audio_emotions = audio_analyzer.analyze(video_data_list)

        emotions = {
            "text": text_emotions,
            "audio": audio_emotions
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error during emotion analysis: {str(e)}")

    # Step 4: Compute correlations
    try:
        correlator = Correlator()
        correlations = correlator.compute(video_data_list, text_emotions, audio_emotions)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error during correlation analysis: {str(e)}")

    # Step 5: Generate insights and PR hooks
    try:
        insight_generator = InsightGenerator()
        insights = insight_generator.generate(video_data_list, text_emotions, audio_emotions, correlations)
        pr_hooks = insight_generator.suggest_pr_hooks(insights)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error during insight generation: {str(e)}")

    # Construct response
    return ChatResponse(
        videos=video_urls,
        emotions=emotions,
        correlations=correlations,
        insights=insights,
        pr_hooks=pr_hooks
    )
