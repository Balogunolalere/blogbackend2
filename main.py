import re
# Utility to remove markdown elements from text
def cleanup_markdown(text: str) -> str:
    if not text:
        return ""
    # Remove headings
    text = re.sub(r"^#+\\s*", "", text, flags=re.MULTILINE)
    # Remove bold/italic (**text**, *text*, __text__, _text_)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\\2", text)
    # Remove inline code
    text = re.sub(r"`([^`]*)`", r"\\1", text)
    # Remove blockquotes
    text = re.sub(r"^>\\s*", "", text, flags=re.MULTILINE)
    # Remove unordered/ordered list markers
    text = re.sub(r"^[-*+]\\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\\d+\\.\\s+", "", text, flags=re.MULTILINE)
    # Remove links but keep text
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # Remove images
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\\1", text)
    # Remove extra whitespace
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()
# main.py  (drop-in replacement)
import json
import os
import time
import logging
from typing import List, Dict, Callable, Optional
from functools import wraps
from urllib.parse import quote, unquote


import markdown           # NEW: handles markdown -> html
from pymongo import MongoClient
from gnews import GNews
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, FileResponse
import datetime
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
import textwrap
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Database ----------
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
mongo_db_name = os.getenv("MONGODB_DB", "blogdb")
mongo_collection_name = os.getenv("MONGODB_COLLECTION", "articles")

mongo_client = MongoClient(mongo_uri)
db = mongo_client[mongo_db_name]
if mongo_collection_name not in db.list_collection_names():
    db.create_collection(mongo_collection_name)
articles_collection = db[mongo_collection_name]
articles_collection.create_index("url", unique=True)

# ---------- FastAPI ----------
app = FastAPI(title="Blog API")
templates = Jinja2Templates(directory="templates")

# ---------- Pydantic ----------
class Article(BaseModel):
    title: str
    url: str
    published_date: str
    description: str
    publisher: Dict[str, str]
    image: Optional[str] = None
    source: str = "gnews"
    category: str = "general"
    is_custom: bool = False

class CustomPost(BaseModel):
    title: str
    content: str
    author: str = "Anonymous"
    image: Optional[str] = None
    category: str = "custom"
    published_date: Optional[str] = None

# ---------- Utilities ----------
def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise e
                    wait = backoff_in_seconds * 2 ** x
                    logger.warning(f"Retry {x + 1} in {wait}s …")
                    time.sleep(wait)
                    x += 1
        return wrapper
    return decorator

@retry_with_backoff(retries=3)
def fetch_news(news_func, *args):
    return news_func(*args)

def format_article(article, category="general"):
    url = article.get("url", "")
    base_url = url.split('?')[0]
    return {
        "title": article.get("title", "").strip(),
        "url": base_url,
        "published_date": article.get("published date", ""),
        "description": article.get("description", "").strip(),
        "image": article.get("image", ""),
        "publisher": {
            "href": article.get("publisher", {}).get("href", "").strip(),
            "title": article.get("publisher", {}).get("title", "").strip()
        },
        "source": "gnews",
        "category": category,
    }

# ---------- GNews init ----------
google_news = GNews(
    language=os.getenv("GNEWS_LANGUAGE", "en"),
    country=os.getenv("GNEWS_COUNTRY", "NG"),
    period=os.getenv("GNEWS_PERIOD", "7d"),
    max_results=int(os.getenv("GNEWS_MAX_RESULTS", 10))
)

TOPICS = ['WORLD', 'NATION', 'BUSINESS', 'TECHNOLOGY', 'ENTERTAINMENT', 'SPORTS',
          'SCIENCE', 'HEALTH', 'POLITICS', 'CELEBRITIES', 'ECONOMY', 'FINANCE',
          'EDUCATION', 'FOOD', 'TRAVEL']

COUNTRIES = {
    'Nigeria': 'NG',
    'United_States': 'US',
    'Canada': 'CA',
    'Russia': 'RU',
    'Israel': 'IL',
    'Germany': 'DE'
}

# ---------- Routes ----------
@app.get("/news", response_model=List[Article])
async def get_news(category: str = None, limit: int = 10):
    query = {} if not category else {"category": category}
    articles = list(articles_collection.find(query).sort("published_date", -1).limit(limit))
    for a in articles:
        a.pop("_id", None)
    return articles

@app.post("/news", response_model=Article)
async def add_custom_news(article: Article):
    data = article.dict()
    articles_collection.update_one({"url": data["url"]}, {"$set": data}, upsert=True)
    return data

@app.get("/", response_class=HTMLResponse)
async def home_page(
    request: Request,
    page: int = Query(1, ge=1),
    search: str = Query(None),
    category: str = Query(None)
):
    query = {}
    if search:
        query["$text"] = {"$search": search}
    if category:
        query["category"] = category
    per_page = 12
    offset = (page - 1) * per_page
    total = articles_collection.count_documents(query)
    articles = list(articles_collection.find(query).sort("published_date", -1).skip(offset).limit(per_page))
    for a in articles:
        a.pop("_id", None)
    total_pages = (total + per_page - 1) // per_page
    categories = TOPICS + [f"country_{c}" for c in COUNTRIES.values()]
    return templates.TemplateResponse("index.html", {
        "request": request,
        "articles": articles,
        "current_page": page,
        "total_pages": total_pages,
        "search": search,
        "category": category,
        "categories": categories,
        "date": datetime.datetime.now(),
    })

@app.get("/article/{url:path}", response_class=HTMLResponse)
async def article_details(request: Request, url: str):
    decoded_url = unquote(url).replace('_', '/')
    article = articles_collection.find_one({"url": decoded_url})
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")


    article.pop("_id", None)

    # Clean up markdown from details if present
    if "details" in article:
        article["details"] = markdown.markdown(article["details"])

    related = list(
        articles_collection.find(
            {"category": article["category"], "url": {"$ne": decoded_url}}
        ).limit(3)
    )
    for r in related:
        r.pop("_id", None)

    structured_data = {
        "@context": "https://schema.org",
        "@type": "NewsArticle",
        "headline": article["title"],
        "description": article.get("description", ""),
        "datePublished": article["published_date"],
        "url": article["url"],
        "publisher": {
            "@type": "Organization",
            "name": article["publisher"]["title"],
            "url": article["publisher"]["href"],
        },
        "mainEntityOfPage": {"@type": "WebPage", "@id": str(request.url)},
        "articleSection": article["category"],
    }

    return templates.TemplateResponse(
        "article.html",
        {
            "request": request,
            "article": article,
            "related": related,
            "structured_data": json.dumps(structured_data, ensure_ascii=False),
            "date": datetime.datetime.now(),
        },
    )

# ---------- Custom post ----------
@app.post("/custom-post", response_model=Article)
async def create_custom_post(post: CustomPost):
    import uuid, time
    post_id = str(uuid.uuid4())
    url = f"custom-{post_id}"
    published_date = post.published_date or time.strftime('%Y-%m-%d %H:%M:%S')
    doc = {
        "title": post.title.strip(),
        "url": url,
        "published_date": published_date,
        "description": post.content.strip(),
        "custom_content": post.content.strip(),
        "image": post.image,
        "publisher": {"title": post.author, "href": ""},
        "source": "custom",
        "category": post.category,
        "is_custom": True,
        "created_at": published_date,
    }
    articles_collection.insert_one(doc)
    return Article(**doc)

# ---------- Image share (stub) ----------
@app.get("/api/share-image/{url:path}")
async def generate_share_image(url: str):
    decoded_url = unquote(url).replace('_', '/')
    article = articles_collection.find_one({"url": decoded_url})
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    img = Image.new("RGB", (1200, 630), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = textwrap.fill(article["title"], width=35)
    draw.text((100, 100), text, font=font, fill="black")
    tmp = f"temp_{url.replace('/', '_')}.png"
    img.save(tmp)
    return FileResponse(tmp, media_type="image/png", filename=f"article-{url}.png")

# ---------- Entry point ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)