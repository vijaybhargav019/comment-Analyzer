from flask import Flask, render_template, request
import re
import emoji
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# Loading the pre-trained sentiment analysis model
analyzer = SentimentIntensityAnalyzer()

# YouTube API key (Replace with your API key)
API_KEY = 'AIzaSyCApYtAxqjbgmz3m63ShbrP0zSgMfZmPR8'
youtube = build('youtube', 'v3', developerKey=API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    video_url = request.form['video_url']
    video_id = re.search(r"(?<=v=)[a-zA-Z0-9_-]+", video_url).group()

    video_title = get_video_title(video_id)

    uploader_channel_id = get_uploader_channel_id(video_id)
    comments = fetch_comments(video_id, uploader_channel_id)

    threshold_ratio = 0.65
    relevant_comments = filter_comments(comments, uploader_channel_id, threshold_ratio)

    sentiment_results = analyze_sentiments(relevant_comments)

    return render_template('results.html', video_url=video_url,video_title=video_title, sentiment_results=sentiment_results)

def get_video_title(video_id):
    video_response = youtube.videos().list(
        part='snippet',
        id=video_id
    ).execute()
    video_snippet = video_response['items'][0]['snippet']
    video_title = video_snippet['title']
    return video_title
def get_uploader_channel_id(video_id):
    video_response = youtube.videos().list(
        part='snippet',
        id=video_id
    ).execute()
    video_snippet = video_response['items'][0]['snippet']
    uploader_channel_id = video_snippet['channelId']
    return uploader_channel_id

def fetch_comments(video_id, uploader_channel_id):
    comments = []
    nextPageToken = None

    while len(comments) < 600:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            pageToken=nextPageToken
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            if comment['authorChannelId']['value'] != uploader_channel_id:
                comments.append(comment['textDisplay'])

        nextPageToken = response.get('nextPageToken')
        if not nextPageToken:
            break

    return comments

def filter_comments(comments, uploader_channel_id, threshold_ratio):
    hyperlink_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    relevant_comments = []

    for comment_text in comments:
        comment_text = comment_text.lower().strip()
        emojis = emoji.emoji_count(comment_text)
        text_characters = len(re.sub(r'\s', '', comment_text))

        if (any(char.isalnum() for char in comment_text)) and not hyperlink_pattern.search(comment_text):
            if emojis == 0 or (text_characters / (text_characters + emojis)) > threshold_ratio:
                relevant_comments.append(comment_text)

    return relevant_comments

def analyze_sentiments(comments):
    polarity = []
    positive_comments = []
    negative_comments = []
    neutral_comments = []

    for comment in comments:
        sentiment_dict = analyzer.polarity_scores(comment)
        polarity.append(sentiment_dict['compound'])

        if sentiment_dict['compound'] > 0.05:
            positive_comments.append(comment)
        elif sentiment_dict['compound'] < -0.05:
            negative_comments.append(comment)
        else:
            neutral_comments.append(comment)

    avg_polarity = sum(polarity) / len(polarity)

    sentiment_results = {
        'avg_polarity': avg_polarity,
        'positive_comments': positive_comments,
        'negative_comments': negative_comments,
        'neutral_comments': neutral_comments,
        'positive_count': len(positive_comments),
        'negative_count': len(negative_comments),
        'neutral_count': len(neutral_comments)
    }

    return sentiment_results



if __name__ == '__main__':
    app.run(debug=True)
