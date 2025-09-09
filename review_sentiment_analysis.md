# Sentiment analysis of AirBnB reviews

Downloading AirBnB reviews from insideairbnb.com, and perform sentiment analysis to estimate if the review is positive or not.
I utilize the Sentiment Intensity Analyzer and the Vader lexicon from the nltk library for Natural Language Processing.

The Sentiment Analysis is based on a machine learning model trained on the Vader lexicon to learn which words and sentences are associated with positive, negative or neutral sentiments of text. The model, trained on the Vader lexicon, can be used to predict the sentiment of AirBnB reviews.


```python
import numpy as np
import polars as pl
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

import matplotlib.pyplot as plt
```

Download and view the review file:


```python
reviews_url = "https://data.insideairbnb.com/denmark/hovedstaden/copenhagen/2025-03-23/data/reviews.csv.gz"
reviews = pl.read_csv(reviews_url)
reviews
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (405_687, 6)</small><table border="1" class="dataframe"><thead><tr><th>listing_id</th><th>id</th><th>date</th><th>reviewer_id</th><th>reviewer_name</th><th>comments</th></tr><tr><td>i64</td><td>i64</td><td>str</td><td>i64</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>31094</td><td>79346</td><td>&quot;2010-08-16&quot;</td><td>171607</td><td>&quot;Ben&quot;</td><td>&quot;We had a great stay. Convenien…</td></tr><tr><td>31094</td><td>166275</td><td>&quot;2011-01-05&quot;</td><td>306860</td><td>&quot;Makita&quot;</td><td>&quot;It was a very good stay. The a…</td></tr><tr><td>31094</td><td>1452299</td><td>&quot;2012-06-10&quot;</td><td>1321058</td><td>&quot;Pierre&quot;</td><td>&quot;Really enjoyed my time at Ebbe…</td></tr><tr><td>31094</td><td>6766430</td><td>&quot;2013-08-24&quot;</td><td>2182771</td><td>&quot;Sussie&quot;</td><td>&quot;The apartment was very well lo…</td></tr><tr><td>31094</td><td>6827217</td><td>&quot;2013-08-26&quot;</td><td>8025926</td><td>&quot;Wil&quot;</td><td>&quot;This is a great flat, very cle…</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>1362905566995344056</td><td>1378257872266725942</td><td>&quot;2025-03-16&quot;</td><td>563191034</td><td>&quot;Mikola-Mai&quot;</td><td>&quot;Virkelig hyggelig lejlighed, p…</td></tr><tr><td>1363132462929472370</td><td>1374597761849701910</td><td>&quot;2025-03-11&quot;</td><td>452066787</td><td>&quot;Nataliia&quot;</td><td>&quot;Sofie is very sweet host and a…</td></tr><tr><td>1363485456773382165</td><td>1378929518100585000</td><td>&quot;2025-03-17&quot;</td><td>567854733</td><td>&quot;Alina&quot;</td><td>&quot;mange tak, jeg kunne lide alt&quot;</td></tr><tr><td>1363535386485887401</td><td>1365157305570941931</td><td>&quot;2025-02-26&quot;</td><td>12083165</td><td>&quot;Mette&quot;</td><td>&quot;Skøn, lys, lækker, zen, stille…</td></tr><tr><td>1363580787176334093</td><td>1379615616406078440</td><td>&quot;2025-03-18&quot;</td><td>525703345</td><td>&quot;Matthew&quot;</td><td>&quot;Great location if you want to …</td></tr></tbody></table></div>



nltk's polarity scores ranges from -1 (very negative) to 1 (very positive). We choose a fairly liberal cutoff of (+/-) 0.05, meaning that everything between -0.05 and 0.05 is neutral.


```python
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):

    if isinstance(text, str):

        score = sia.polarity_scores(text)["compound"]
        if score > 0.05:
                return "positive"
        elif score < -0.05:
            return "negative"
        else:
            return "neutral"

    else:
        return np.nan



reviews = reviews.with_columns(
    pl.col("comments").map_elements(get_sentiment, return_dtype=pl.String).alias("sentiment")
)

```

    [nltk_data] Downloading package vader_lexicon to
    [nltk_data]     C:\Users\Bruger\AppData\Roaming\nltk_data...
    [nltk_data]   Package vader_lexicon is already up-to-date!
    

### Quality control:
We check a random sample of reviews with low polarity scores (< -0.05), if they do in fact sound negative and vice versa.


```python
# Negative reviews?
reviews.filter(pl.col("sentiment") == "negative").head(5).select("comments")
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 1)</small><table border="1" class="dataframe"><thead><tr><th>comments</th></tr><tr><td>str</td></tr></thead><tbody><tr><td>&quot;A localização era óptima com v…</td></tr><tr><td>&quot;Nous avons passé un superbe sé…</td></tr><tr><td>&quot;Vor unserer Ankunft bekamen wi…</td></tr><tr><td>&quot;L&#x27;appartement était charmant, …</td></tr><tr><td>&quot;Flott opphold i København i st…</td></tr></tbody></table></div>




```python
# Positive reviews?
reviews.filter(pl.col("sentiment") == "positive").head(5).select("comments")
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 1)</small><table border="1" class="dataframe"><thead><tr><th>comments</th></tr><tr><td>str</td></tr></thead><tbody><tr><td>&quot;We had a great stay. Convenien…</td></tr><tr><td>&quot;It was a very good stay. The a…</td></tr><tr><td>&quot;Really enjoyed my time at Ebbe…</td></tr><tr><td>&quot;The apartment was very well lo…</td></tr><tr><td>&quot;This is a great flat, very cle…</td></tr></tbody></table></div>




```python
# Neutral reviews?
reviews.filter(pl.col("sentiment") == "neutral").head(5).select("comments")
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 1)</small><table border="1" class="dataframe"><thead><tr><th>comments</th></tr><tr><td>str</td></tr></thead><tbody><tr><td>&quot;Fin lägenhet med allt man kan …</td></tr><tr><td>&quot;Wunderschöne Wohnung in toller…</td></tr><tr><td>&quot;Bellissima posizione e bellima…</td></tr><tr><td>&quot;Leiligheten var stilig, ren og…</td></tr><tr><td>&quot;Ruim appartement met veel rame…</td></tr></tbody></table></div>



Well, it turns out that negative and neutral reviews are in fact reviews in other langugages than english.

We need ways to account for other languagues or to remove non-english reviews. For this we can use the langdetect library with langugage detection models. We create a dummy variable indicating wether or not the review is in english or not and then perform the sentiment analysis on the english reviews.



```python
from tqdm import tqdm
import fasttext

model = fasttext.load_model("path/to/fasttext/dict/lid.176.bin")  # Make sure it's in your working directory

def detect_language(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return "unknown"
    prediction = model.predict(text.strip().replace('\n', ' '), k=1)
    lang = prediction[0][0].replace("__label__", "")
    return lang


reviews = reviews.with_columns(
    pl.col("comments").map_elements(detect_language, return_dtype=pl.String).alias("language")
)
```

    Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
    


```python
reviews
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (405_687, 8)</small><table border="1" class="dataframe"><thead><tr><th>listing_id</th><th>id</th><th>date</th><th>reviewer_id</th><th>reviewer_name</th><th>comments</th><th>sentiment</th><th>language</th></tr><tr><td>i64</td><td>i64</td><td>str</td><td>i64</td><td>str</td><td>str</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>31094</td><td>79346</td><td>&quot;2010-08-16&quot;</td><td>171607</td><td>&quot;Ben&quot;</td><td>&quot;We had a great stay. Convenien…</td><td>&quot;positive&quot;</td><td>&quot;en&quot;</td></tr><tr><td>31094</td><td>166275</td><td>&quot;2011-01-05&quot;</td><td>306860</td><td>&quot;Makita&quot;</td><td>&quot;It was a very good stay. The a…</td><td>&quot;positive&quot;</td><td>&quot;en&quot;</td></tr><tr><td>31094</td><td>1452299</td><td>&quot;2012-06-10&quot;</td><td>1321058</td><td>&quot;Pierre&quot;</td><td>&quot;Really enjoyed my time at Ebbe…</td><td>&quot;positive&quot;</td><td>&quot;en&quot;</td></tr><tr><td>31094</td><td>6766430</td><td>&quot;2013-08-24&quot;</td><td>2182771</td><td>&quot;Sussie&quot;</td><td>&quot;The apartment was very well lo…</td><td>&quot;positive&quot;</td><td>&quot;en&quot;</td></tr><tr><td>31094</td><td>6827217</td><td>&quot;2013-08-26&quot;</td><td>8025926</td><td>&quot;Wil&quot;</td><td>&quot;This is a great flat, very cle…</td><td>&quot;positive&quot;</td><td>&quot;en&quot;</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>1362905566995344056</td><td>1378257872266725942</td><td>&quot;2025-03-16&quot;</td><td>563191034</td><td>&quot;Mikola-Mai&quot;</td><td>&quot;Virkelig hyggelig lejlighed, p…</td><td>&quot;neutral&quot;</td><td>&quot;da&quot;</td></tr><tr><td>1363132462929472370</td><td>1374597761849701910</td><td>&quot;2025-03-11&quot;</td><td>452066787</td><td>&quot;Nataliia&quot;</td><td>&quot;Sofie is very sweet host and a…</td><td>&quot;positive&quot;</td><td>&quot;en&quot;</td></tr><tr><td>1363485456773382165</td><td>1378929518100585000</td><td>&quot;2025-03-17&quot;</td><td>567854733</td><td>&quot;Alina&quot;</td><td>&quot;mange tak, jeg kunne lide alt&quot;</td><td>&quot;neutral&quot;</td><td>&quot;da&quot;</td></tr><tr><td>1363535386485887401</td><td>1365157305570941931</td><td>&quot;2025-02-26&quot;</td><td>12083165</td><td>&quot;Mette&quot;</td><td>&quot;Skøn, lys, lækker, zen, stille…</td><td>&quot;neutral&quot;</td><td>&quot;da&quot;</td></tr><tr><td>1363580787176334093</td><td>1379615616406078440</td><td>&quot;2025-03-18&quot;</td><td>525703345</td><td>&quot;Matthew&quot;</td><td>&quot;Great location if you want to …</td><td>&quot;positive&quot;</td><td>&quot;en&quot;</td></tr></tbody></table></div>



### Language detection
Using fasttext's language detection model we see that ca. 283K reviews out of 400K are in english language, at close seconds are german, french and danish. We lose a lot of information, but since english has the highest reach of any language here. So we assume that english review have the highest impact on availability and potential reveneue of a listing. We go by removing the non-english reviews for now. In a later project I want to explore multilingual sentiment analysis, but that's a later project


```python
reviews.group_by("language").count().sort("count", descending=True)
```

    C:\Users\Bruger\AppData\Local\Temp\ipykernel_3792\116894271.py:1: DeprecationWarning: `GroupBy.count` is deprecated. It has been renamed to `len`.
      reviews.group_by("language").count().sort("count", descending=True)
    




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (70, 2)</small><table border="1" class="dataframe"><thead><tr><th>language</th><th>count</th></tr><tr><td>str</td><td>u32</td></tr></thead><tbody><tr><td>&quot;en&quot;</td><td>283748</td></tr><tr><td>&quot;de&quot;</td><td>30625</td></tr><tr><td>&quot;fr&quot;</td><td>25469</td></tr><tr><td>&quot;da&quot;</td><td>21178</td></tr><tr><td>&quot;no&quot;</td><td>8801</td></tr><tr><td>&hellip;</td><td>&hellip;</td></tr><tr><td>&quot;sw&quot;</td><td>1</td></tr><tr><td>&quot;ast&quot;</td><td>1</td></tr><tr><td>&quot;fy&quot;</td><td>1</td></tr><tr><td>&quot;lb&quot;</td><td>1</td></tr><tr><td>&quot;ta&quot;</td><td>1</td></tr></tbody></table></div>




```python
en_reviews = reviews.filter(pl.col("language") == "en").with_columns(
    pl.col("comments").map_elements(get_sentiment, return_dtype=pl.String).alias("sentiment")
)
```


```python
# Negative reviews?
en_reviews.filter(pl.col("sentiment") == "negative").select("comments")[10].item()
```




    'The property was disappointing due to an obvious lack of cleanliness . There are cobwebs on windows , stains on many surfaces , the shower was off putting and the bed linen old and in one room ill fitting. Most windows lacked blinds making it difficult to sleep late, and whilst we were aware that we were renting a private residence we felt that little preparation/consideration had been made for our stay . The toilet consistently blocked and the kitchen felt dated, grubby and cluttered . The space is good and the location great but we felt uncomfortable due to the reasons mentioned.'




```python
# Positive reviews?
en_reviews.filter(pl.col("sentiment") == "positive").head(5).select("comments")
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 1)</small><table border="1" class="dataframe"><thead><tr><th>comments</th></tr><tr><td>str</td></tr></thead><tbody><tr><td>&quot;We had a great stay. Convenien…</td></tr><tr><td>&quot;It was a very good stay. The a…</td></tr><tr><td>&quot;Really enjoyed my time at Ebbe…</td></tr><tr><td>&quot;The apartment was very well lo…</td></tr><tr><td>&quot;This is a great flat, very cle…</td></tr></tbody></table></div>




```python
# Neutral reviews?
en_reviews.filter(pl.col("sentiment") == "neutral").head(5).select("comments")
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 1)</small><table border="1" class="dataframe"><thead><tr><th>comments</th></tr><tr><td>str</td></tr></thead><tbody><tr><td>&quot;Unfortunately Andrea travelled…</td></tr><tr><td>&quot;Me and my freind, had a wonder…</td></tr><tr><td>&quot;We travelkedSigrid&#x27;s place is …</td></tr><tr><td>&quot;A stylish place with cosy envi…</td></tr><tr><td>&quot;In case of heavy luggage: the …</td></tr></tbody></table></div>




```python
en_reviews = en_reviews.with_columns(
    pl.when(pl.col.sentiment == "positive").then(1).otherwise(0).alias("positive_reviews"),
    pl.when(pl.col.sentiment == "neutral").then(1).otherwise(0).alias("neutral_reviews"),
    pl.when(pl.col.sentiment == "negative").then(1).otherwise(0).alias("negative_reviews"),

)


reviews_sentiments = en_reviews.group_by("listing_id").agg(
    pl.sum("positive_reviews"),
    pl.sum("neutral_reviews"),
    pl.sum("negative_reviews"),

)

reviews_sentiments.write_csv("path/to/disc/review_sentiment.csv")
```


```python
reviews_sentiments
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (17_718, 4)</small><table border="1" class="dataframe"><thead><tr><th>listing_id</th><th>positive_reviews</th><th>neutral_reviews</th><th>negative_reviews</th></tr><tr><td>i64</td><td>i32</td><td>i32</td><td>i32</td></tr></thead><tbody><tr><td>972397314515315847</td><td>1</td><td>0</td><td>0</td></tr><tr><td>9407356</td><td>25</td><td>1</td><td>0</td></tr><tr><td>32029136</td><td>6</td><td>1</td><td>1</td></tr><tr><td>33851914</td><td>10</td><td>0</td><td>0</td></tr><tr><td>34813295</td><td>12</td><td>0</td><td>1</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>33874452</td><td>21</td><td>0</td><td>1</td></tr><tr><td>939836032656880517</td><td>9</td><td>0</td><td>0</td></tr><tr><td>24751558</td><td>10</td><td>0</td><td>0</td></tr><tr><td>893185768164204700</td><td>1</td><td>0</td><td>0</td></tr><tr><td>23975040</td><td>28</td><td>0</td><td>0</td></tr></tbody></table></div>



