from fastapi import FastAPI
from app.services.fetch_news import fetch_good_news

app = FastAPI()

@app.get("/good-news")
def get_good_news():
    try:
        # Fetch good news articles
        articles = fetch_good_news()
        if not articles:
            return {"message": "No good news found."}
    
        return articles

    
    except Exception as e:
        return {"status":"failure", "error": str(e)}
