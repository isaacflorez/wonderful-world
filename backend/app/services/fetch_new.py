import os
import requests
from dotenv import load_dotenv
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

# TODO: Add this to a config file
positive_words = "celebrate OR anniversary OR birthday OR wedding OR graduation OR promotion OR achievement OR success OR milestone OR joy OR happiness OR love OR kindness OR compassion OR generosity OR gratitude OR hope OR inspiration OR motivation OR encouragement OR support OR teamwork OR community OR togetherness OR unity OR peace OR harmony OR understanding OR acceptance"

# Function to fetch good news based on a query and date range
def fetch_good_news(query=positive_words, language="en", page_size=30):
    # set parameters for the API request
    params = {
        "q": query,
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
        "language": language
    }
    
    # make the API request
    print("----------------------------------------------------")
    print(f"Fetching {page_size} news articles....")
    print("----------------------------------------------------")
    response = requests.get(NEWS_API_URL, params=params)
    data = response.json()

    # 200 status code indicates success
    # return the list of articles if the request was successful
    if response.status_code == 200:
        print(f"Successfully fetched {len(data.get('articles', []))} articles.")
        trimmed_articles = trim_news_article_data(data.get("articles", []))
        # return trimmed_articles
        print("----------------------------------------------------")
        print("Summarizing articles........")
        print("----------------------------------------------------")
        articles = summarize_articles(trimmed_articles)
        
        print(f"Summarized {len(articles)}")
        print("----------------------------------------------------")
        print("Classifying articles........")
        print("----------------------------------------------------")
        # classify the articles using zero-shot classification
        output = classify_articles(articles)
        print(f"Classified {len(output)} articles.")
        print("----------------------------------------------------")
        return output

    else:
        print(f"Error fetching news: {response.status_code}")
        return []

# Function to trim the content of the news article and filter out non-positive articles
def trim_news_article_data(contents):
    """
    Trims the content of the news article to only contain title, description, content, author, and url.
    It also performs sentiment analysis to filter out non-positive articles. 

    returns a list of trimmed articles with positive sentiment from TextBlob polarity score
    """
    # For now, just return a new article list with trimmed attributes
    if not contents:
        print("trim_news_article_data: No content to trim.")
        return ""
    
    trimmed_news_data = []
    filter_count = 0
    for article in contents:
        news_article = {
            "title": article.get("title", ""),
            "description": article.get("description", ""),
            "content": article.get("content", ""),
            "author": article.get("author", ""),
            "url": article.get("url", "")
        }
        # Combine title, description, and content for sentiment analysis
        # NOTE: should we summarize the content first? Or just use the content as is?
        # For now, we will use the content as is
        # This will be used to check if the article is positive or not
        # If the article is positive, we will keep it in the list

        # Uncomment the following lines to filter out non-positive articles
        # text_preview = f"{article['title']} {article['description']} {article['content']}"
        # if is_positive_article(text_preview):
        #     # news_article["is_positive"] = True
        #     trimmed_news_data.append(news_article)
        # else:
        #     filter_count += 1

        trimmed_news_data.append(news_article)

    print("----------------------------------------------------")
    print(f"After filtering out by sentiment, there is now {len(trimmed_news_data)} articles")
    return trimmed_news_data

# Helper function to check if the article text contains positive sentiment
def is_positive_article(text):
    """
    Check if the article text contains positive sentiment.
    Returns True if positive, False otherwise.
    """
    return TextBlob(text).sentiment.polarity >= .2    

# Main function to summarize a list of articles
def summarize_articles(articles):
    """
    This function summarizes a list of articles using a summarization model.
    It uses the Hugging Face Transformers library to load a pre-trained summarization model.
    The model is fine-tuned specifically to summarize long documents into concise summaries.
    Returns a list of articles with summaries added to each article dictionary under the key "summary".
    If the article is too short, it will return the original text instead of a summary.
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    for article in articles:
        article["summary"] = summarize_article_content(article, summarizer, tokenizer)
    return articles

# Helper function to summarize the content of a single article
def summarize_article_content(article, summarizer=None, tokenizer=None):
    """
    Summarizes the content of the article using a summarization model.
    This model fine-tuned specifically to summarize long documents into concise summaries
    The summary is what we will pass into zero shot classification model to classify the article.
    Returns content summary.
    """
    if not article or not article.get("content"):
        return "No article or content to summarize."
    
    # summarizer uses tokenization, so we should check the length of the text in tokens
    text_preview = f"{article['title']} {article['description']} {article['content']}"
    input_length = len(tokenizer.encode(text_preview))
    
    if input_length > 100:
        summary = summarizer(text_preview, max_length=100, min_length=20, do_sample=False)
        return summary[0]['summary_text'] if summary else "No summary available."
    else:
        return text_preview  # If the text is short, no need to summarize, return the original text

# Function to generate a rating from the NPL model scores
def generate_rating_from_npl_scores(scores):
    """
    Using the scores from the NPL model, gnerate a good news rating.
    This rating is the average of the scores from the NPL model based on the classification labels.

    FOR NOW:
    return the average score rounded to 5 decimal places.
    """
    if not scores:
        return "No scores available."
    total = sum(scores)
    average_score = total / len(scores)
    return round(average_score, 5)

# Function to classify articles using zero-shot classification
def classify_articles_old(articles):
    """
    Classifies each article's summary using the zero-shot classification model.
    Appends the classification result to the article dictionary under the key "rating".
    Returns a list of articles with ratings and scores.

    NOTE: Summary is required for classification, so this function should be called after summarizing the articles.
    """
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = ["uplifting", "positive", "hopeful", "heartwarming", "inspiring", "joyful", "funny", "cheerful", "happy", "good news", "feel good", "lighthearted", "entertaining", "motivational", "supportive", "community", "togetherness"]
    output = []
    for article in articles:
        result = classifier(article["summary"], candidate_labels)
        article['scores'] = result['scores']  # Store the scores in the article dictionary
        article['rating'] = generate_rating_from_npl_scores(result['scores'])
        output.append({
            "title": article.get("title", "No title available."),
            "description": article.get("description", "No description available."),
            "content": article.get("content", ""),
            "author": article.get("author", ""),
            "url": article.get("url", ""),
            "summary": article.get("summary", "No summary available."),
            "rating": article.get("rating", "No rating available."),
            "scores": article.get("scores", "No scores available.")
        })
    return output

# TODO: Combine all logic into a single function that fetches, summarizes, and classifies the articles
# TODO: Are all of the scores trying to add up to a singular, set whole number? If all of the ratings are adding up to one
#then we need to find a way to get a better average score.
# Should we do a single classification for each label and then average the scores?
# IDEA TIME: Loop over the list of candidate labels and classify each article summary for each label.
# This will give us a score for each label, and we can then average those scores to get a final rating for the article.
# This will allow us to have a more granular understanding of the article's sentiment and how it relates to each candidate label.

def classify_articles(articles):
    """
    Classifies each article's summary using the zero-shot classification model.
    Appends the classification result to the article dictionary under the key "rating".
    Returns a list of articles with ratings and scores.
    
    If classifier or candidate_labels are not provided, it uses default values.
    """
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = ["uplifting", "positive", "hopeful", "heartwarming", "inspiring", "joyful", "funny", "cheerful", "happy", "good news", "feel good", "lighthearted", "entertaining", "motivational", "supportive", "community", "togetherness"]
    
    output = []
    for article in articles:
        # this will loop over each label and create an average score for the article
        result = classify_article(article["summary"], classifier, candidate_labels)
        article['scores'] = result['scores']
        article['rating'] = result['rating']
        output.append({
            "title": article.get("title", "No title available."),
            "description": article.get("description", "No description available."),
            "content": article.get("content", ""),
            "author": article.get("author", ""),
            "url": article.get("url", ""),
            "summary": article.get("summary", "No summary available."),
            "rating": article.get("rating", "No rating available."),
            "scores": article.get("scores", "No scores available.")
        })
    
    return output

def classify_article(summary, classifier, candidate_labels):
    """
    Classify each label for the article summary using the zero-shot classification model.
    Create a average score for the article based on the scores for each label.

    Returns a dictionary with labels mapped to scores and a rating based on average score
    """
    label_to_score = {}
    score_list = []
    for label in candidate_labels:
        result = classifier(summary, [label])
        if 'scores' not in result:
            print(f"Error: No scores found in the result for label '{label}'.")
            continue
        label_to_score[label] = result['scores'][0]
        score_list.append(result['scores'][0])
    
    return {
        'scores': label_to_score,
        'rating': generate_rating_from_npl_scores(score_list)
    }



# NPL example usage
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# candidate_labels = ["uplifting", "positive", "hopeful", "heartwarming", "inspiring", "joyful", "funny", "cheerful", "happy", "good news", "feel good", "lighthearted", "entertaining", "motivational", "supportive", "community", "togetherness"]

# EXAMPLE USAGE
articles = fetch_good_news()
# articles = summarize_articles(articles)
# print(f"Fetched {len(articles)} articles with summaries.")
# print(f"Example article after summary: {articles[0] if articles else 'No articles available.'}")
# output = classify_articles(articles, classifier, candidate_labels)
# print(f"Example article after classification: {output[0] if output else 'No articles available.'}")
# print("---------------------------------------------------")
# print("Articles with ratings:")

print("Printing articles with ratings greater than 0.5:")
for item in articles:
    if item.get('rating') > 0.5:
        print(f"Title: {item['title']}, Rating: {item['rating']}, URL: {item['url']}")

# fetch news -> summarize articles -> classify articles
# This is the flow of the application.
# The fetch_news.py file is responsible for fetching the news articles, summarizing them, and classifying them.
