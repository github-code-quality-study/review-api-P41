import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, List, Dict, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

VALID_LOCATIONS = [
    'Denver, Colorado',
    'San Diego, California'
]

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body: str) -> Dict[str, float]:
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: Dict[str, Any], start_response: Callable[[str, List[tuple[str, str]]], None]) -> List[bytes]:
        global reviews

        method = environ.get("REQUEST_METHOD", "GET")

        if method == "GET":
            query_string = environ.get("QUERY_STRING", "")
            params = parse_qs(query_string)

            filtered_reviews = reviews
            if 'location' in params:
                location = params['location'][0]
                filtered_reviews = [r for r in filtered_reviews if r['Location'] == location]
            if 'start_date' in params:
                try:
                    start_date = datetime.strptime(params['start_date'][0], '%Y-%m-%d')
                    filtered_reviews = [r for r in filtered_reviews if datetime.strptime(r['Timestamp'], '%Y-%m-%d %H:%M:%S') >= start_date]
                except ValueError:
                    pass
            if 'end_date' in params:
                try:
                    end_date = datetime.strptime(params['end_date'][0], '%Y-%m-%d')
                    filtered_reviews = [r for r in filtered_reviews if datetime.strptime(r['Timestamp'], '%Y-%m-%d %H:%M:%S') <= end_date]
                except ValueError:
                    pass

            for review in filtered_reviews:
                review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

            filtered_reviews.sort(key=lambda x: x['sentiment']['compound'], reverse=True)

            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])

            return [response_body]

        elif method == "POST":
            try:
                request_body_size = int(environ.get('CONTENT_LENGTH', '0'))
            except (ValueError):
                request_body_size = 0

            request_body = environ['wsgi.input'].read(request_body_size).decode('utf-8')

            if not request_body:
                start_response("400 Bad Request", [("Content-Type", "text/plain")])
                return [b"Empty request body"]

            content_type = environ.get('CONTENT_TYPE', '')

            if content_type == 'application/json':
                try:
                    post_data = json.loads(request_body)
                except json.JSONDecodeError:
                    start_response("400 Bad Request", [("Content-Type", "text/plain")])
                    return [b"Invalid JSON"]
            else:
                post_data = parse_qs(request_body)

            if 'Location' not in post_data or 'ReviewBody' not in post_data:
                start_response("400 Bad Request", [("Content-Type", "text/plain")])
                return [b"Missing Location or ReviewBody"]

            location = post_data['Location'] if isinstance(post_data['Location'], str) else post_data['Location'][0]
            review_body = post_data['ReviewBody'] if isinstance(post_data['ReviewBody'], str) else post_data['ReviewBody'][0]

            if location not in VALID_LOCATIONS:
                start_response("400 Bad Request", [("Content-Type", "text/plain")])
                return [b"Invalid Location"]

            new_review = {
                "ReviewId": str(uuid.uuid4()),
                "ReviewBody": review_body,
                "Location": location,
                "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }

            reviews.append(new_review)
            response_body = json.dumps(new_review, indent=2).encode('utf-8')

            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])

            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
