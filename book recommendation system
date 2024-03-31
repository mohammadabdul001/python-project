from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

# Sample book dataset
books_data = {
    'Title': ['NARUTO', 'NARUTO-PART 1', 'NARUTO-PART 2', 'NARUTO-PART 3', 'NARUTO-PART 4'],
    'Author': ['Author X', 'Author Y', 'Author X', 'Author Z', 'Author Y'],
    'Genre': ['Fiction', 'Non-Fiction', 'Fiction', 'Science Fiction', 'Fiction'],
    'Synopsis': ['Synopsis of AVENGERS', 'Synopsis of NARUTO', 'Synopsis of SIPERMAN', 'Synopsis of X-MEN', 'Synopsis of POKEMON']
}

# Create DataFrame
books_df = pd.DataFrame(books_data)

# Initialize TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the synopsis data
tfidf_matrix = tfidf.fit_transform(books_df['Synopsis'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, cosine_sim=cosine_sim):

 # Get index of the book
    idx = books_df[books_df['Title'] == title].index[0]
 # Get pairwise similarity scores of the book
    sim_scores = list(enumerate(cosine_sim[idx]))
 # Sort books based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
  # Get top 5 recommendations
    top_similar_books = sim_scores[1:6]
 # Get titles of recommended books
    recommended_books = [books_df.iloc[i[0]]['Title'] for i in top_similar_books]
    return recommended_books



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    if request.method == 'POST':
        book_title = request.form['book_title']
        recommended_books = get_recommendations(book_title)
        return render_template('recommendations.html', book_title=book_title, recommended_books=recommended_books)

if __name__ == '__main__':
    app.run(debug=True)
