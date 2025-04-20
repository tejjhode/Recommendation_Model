from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sqlalchemy
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# Connect to MySQL Database
engine = sqlalchemy.create_engine(
    'mysql+pymysql://admin:#Tej1234@database-2.cfyq26cwa8xv.ap-south-1.rds.amazonaws.com:3306/database-2'
)

# Load product data
df = pd.read_sql("SELECT id, name, description, price, category, imageUrl FROM products", con=engine)

# Handle missing values
df['description'] = df['description'].fillna('')
df['category'] = df['category'].fillna('')
df['price'] = df['price'].fillna(df['price'].mean())

# Normalize price
scaler = MinMaxScaler()
df['price_scaled'] = scaler.fit_transform(df[['price']])

# Combine description and category into text features
df['text_features'] = df['description'] + " " + df['category']
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['text_features'])

# Combine text features and price
combined_features = hstack([tfidf_matrix, df[['price_scaled']].values])

# Cosine similarity matrix
cosine_sim = cosine_similarity(combined_features, combined_features)

# FastAPI app
app = FastAPI()

class ProductRecommendationRequest(BaseModel):
    product_id: int
    top_n: int = 5

class Product(BaseModel):
    id: int
    name: str
    description: str
    price: float
    category: str
    image_url: str

@app.post("/recommendations/")
async def get_recommendations(request: ProductRecommendationRequest):
    try:
        if request.product_id not in df['id'].values:
            raise HTTPException(status_code=404, detail="Product not found.")

        # Get the index of the product
        idx = df.index[df['id'] == request.product_id][0]

        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top N similar products (excluding itself)
        sim_scores = [score for score in sim_scores if score[0] != idx][:request.top_n]
        product_indices = [i[0] for i in sim_scores]
        recommended_products = df.iloc[product_indices]

        return {
            "recommendations": recommended_products[[
                'id', 'name', 'description', 'price', 'category', 'imageUrl'
            ]].to_dict(orient='records')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Product Recommendation API!"}

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
