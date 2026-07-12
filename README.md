# PythiaRankflow

https://grouplens.org/datasets/movielens/1m/

**"How can we systematically combine and enhance classical recommendation models to achieve state-of-the-art performance on this specific dataset, without using deep learning?"**

look up to this

- data sparsity checking
- rating bias
- cold bias
- long tail problem

- EDA --> informed data pruning and Constraint Verification
- preprocessing(K-Core filtering)
- Data splitting(Temporal Split/cross validation)
- Feature Engineering(Create rich tabular features/calculated ONLY on train sets)
- Ranking(Use gradient boosting to combine everything)
- Model Training(Classical Models & Gradient Boost)
- Evaluation - RMSE, MAE, Precision@K
- Compare accross models


# Overview

Implement collaborative filtering with content based filtering that uses item metadata and user demographics. Sticking to pure classical text processing: TF-IDF(Term Frequency-Inverse Document Frequency) combined with SVD(Truncated SVD) for latent Semantic Indexing. For the ranking we can use the gradient boosted decision trees (GBDT) also known as LambdaMART in the market



# What i think i am gonna do 

0. Exploratory Data Analysis
- Data Sparsity & Power-Law verifiction: Computing the global densisty of the matrix
(tells us why the single matrix factorization fails and we can justify using two stage candidate retrieval + GBDT ranking architecture)

- Bias formulation. -> Optimistic vs Pessimistic raters
(calculate the mean rating for each user and plot a distribution histogram)
(some user has mean rating of 4.5(optimistic) while other sits at 2.1(pessimistic). This variation justifies substracting the user's mean from explicit ratings during feature engineering to expose **true relative preference**, minimizing user bias)

- Item Popularity and Long tail Problem: Sort movies by their interaction count in descending order. Plot the interaction count (Y-axis) against the sorted item index(X-index). Highlight the point where 80% of all the interaction are exhausted Popularity and Long tail Problem: Sort movies by their interaction count in descending order. Plot the interaction count (Y-axis) against the sorted item index(X-index). Highlight the point where 80% of all the interaction are exhausted.

- Temporal drift tracking: Dividing the dataset's chronological timeline into 10 buckets. Plot the preference shift dynamically, providing empirical proof for why a random 80/20 train/test split suffers from future data leakage, validating the choice of the strict temporal split.


1. **Preprocessing**
	1. k-core filtering
- Keep only users who rated at least 20 movies
- Keep only movies who got at least 20 reviews
- Removes everything else

	2. Temporal Split
- Instead of the random 80/20 split we split by time the older data trains and newer data tests(sort each user's rating by `timestamp`)

`Raw data(1M ratings)-> K-Core Filtering (removes noise, sparse users/items) -> Clean Dense Data -> Temporal Split(honest train/test split) -> Train Set -> Fit SVD, KNN, NMF Test Set -> Evaluate RMSE, MAE`

2. **Implementing the baselines(we need to record RMSE, MAE, Precision@10, Recall@K)**

**Surprise library** it implements things out of the box(scikit learn but for recommendation system)
- Global Popularity 
- CatPop (Category Popularity)
- UserCF
- ItemCF
- ExplicitMF(SVD)
- ImplicitMF(ALS)
- Content Based Filtering(CBF)

3. **Candidate Retrieval(ItemCF)**

- Candidate Retrieval(Narrow down 3700 movies to ~220 candidates per user motive is for the GBDT)
- ItemCF narrows the millions of items into the few hundreds

4. **Feature Engineering for the GBDT**

For the 200-300 candidate movies, we create the feature vector for that(User, Movie) pair.
And we inject the "hybrid" approach.

Group A: Collaborative Features(From SVD/ALS)
- Latent factor dot product(the raw prediction from SVD)
- Latent factor dot product(from ALS)
- Similarity score from ItemCF

Group B(Strictly classical TF-IDF)
- Text Features: Combine `title`+`genres`. fit TF-IDF. compute the cosine similarity between users average rated movie TF-IDF vector and candidate movie's TF-IDF.(Our replacement for BERT/GloVe)
- Genre match: count how many genres overlap between the user's favourite genres and the candidate movies

Group C(User/Movie Demographics metadata)
- Movie Side: Average rating of the movie, total rating count
- User side: User's average rating, rating variance, users age, occupation, gender(one-hot encoded)

Group D(Implicit Signals Weighted)
- Weighted interaction score: if they viewed/rated it , weight it by `1/(1+days_since_rating)` to give recent movies higher impact

5. **The Ranking Model(LambdaMART/GBDT)**
- Gradient Boosted Decision Trees(LigthGBM or XGBoost) 
- Training (200 candidate movies, we know the true rating the user gave, We gotta frame this as Regression task(predict the rating)) or Ranking test (use lambdarank objective to optimize NDCG)
- Feed into lightgbm

6. Evaluation of the model and the error function
- ROC curves
- visually plot
- Precision@K
