import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recom(books_user_likes):
    def get_title_from_index(index):
        return df[df.index == index]["Title"].values[0]

    def get_index_from_title(Title):
        return df[df.Title == Title]["index"].values[0]

    books = pd.read_csv("BookDataset/Bookz.csv")
    books = books[:1000]
    df = books
    img = pd.read_csv("BookDataset/Imagez.csv")

    features = ['Title', 'Author', 'Publisher']
    for feature in features:
        df[feature] = df[feature].fillna('')

    def combine_features(row):
        try:
            return row['Title'] + " " + row['Author'] + " " + row['Publisher']
        except:
            print("Error:", row)

    df["combined_features"] = df.apply(combine_features, axis=1)

    # Create count matrix from this new combined column
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df["combined_features"])

    # Compute the Cosine Similarity based on the count_matrix
    cosine_sim = cosine_similarity(count_matrix)

    # Get index of this book from its title
    books_index = get_index_from_title(books_user_likes)
    similar_books = list(enumerate(cosine_sim[books_index]))

    # Get a list of similar books in descending order of similarity score
    sorted_similar_books = sorted(similar_books, key=lambda x: x[1], reverse=True)

    # titles of first 10 books
    l = []
    t = []
    i = 0
    for element in sorted_similar_books:
        l.append(get_title_from_index(element[0]))
        t.append(get_index_from_title(l[i]))
        i += 1
        if i > 9:
            break

    output = l
    index = t

    imgg = []
    year = []
    author = []
    final_list = []
    for i in index:
        imgg.append(img["Image-URL-M"][i - 1])
        year.append(books["Year"][i - 1])
        author.append(books["Author"][i - 1])
    for i in range(len(index)):
        temp = []
        temp.append(output[i])
        temp.append(imgg[i])
        temp.append(year[i])
        temp.append(author[i])
        final_list.append(temp)
    return final_list


def bookdisp():
    books = pd.read_csv("BookDataset/Bookz.csv")
    img = pd.read_csv("BookDataset/Imagez.csv")

    title = []
    imgg = []
    year = []
    author = []
    finallist = []

    r = np.random.randint(2, 1000, 10)

    for i in r:
        title.append(books["Title"][i - 1])
        imgg.append(img["Image-URL-M"][i - 1])
        year.append(books["Year"][i - 1])
        author.append(books["Author"][i - 1])

    for i in range(10):
        temp = []
        temp.append(title[i])
        temp.append(imgg[i])
        temp.append(year[i])
        temp.append(author[i])
        finallist.append(temp)

    return finallist
def realtime_recommend(emotion, gaze):
    # Step 1: Get books displayed on screen (10 books shown via bookdisp())
    books_on_screen = bookdisp()  # this returns list of [title, img, year, author]

    # Step 2: Map gaze direction to book index
    gaze_to_index = {
        'left': 0,
        'center': 4,
        'right': 9
    }

    if gaze not in gaze_to_index:
        return []  # invalid gaze

    book_index = gaze_to_index[gaze]
    if book_index >= len(books_on_screen):
        return []

    book_title = books_on_screen[book_index][0]  # Get title from screen books
    print(f"Detected Gaze on Book: {book_title} with Emotion: {emotion}")

    # Step 3: If emotion is positive (happy, excited), generate recommendations from it
    if emotion.lower() in ['happy', 'excited', 'positive']:
        return recom(book_title)  # Your existing recommendation logic
    else:
        return []  # Don't recommend if user is sad/angry etc.
