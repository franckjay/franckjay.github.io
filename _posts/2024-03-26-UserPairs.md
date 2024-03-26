---
layout: post
comments: true
title:  "Learning to Rank — Contextual Item Recommendations for User Pairs"
excerpt: "Coffee Tasting"
date:   2024-03-26
---

# Learning to Rank — Contextual Item Recommendations for User Pairs
Jay Franck

Imagine you are sitting on your couch, friends or family present. You have your preferred game console/streaming service/music app open, and each item is a glittering jewel of possibility, tailored for you. But those personalized results may be for the solo version of yourself, and does not reflect the version of yourself when surrounded by this particular mix of others.

This project truly started with coffee. I am enamored with roasting my own green coffee sourced from Sweet Maria’s (no affiliation), as it has such a variety of delicious possibilities. Colombian? Java-beans? Kenyan Peaberry? Each description is more tantalizing than the last. It is so hard to choose even for myself as an individual. What happens if you are buying green coffee for your family or guests?

I wanted to create a Learning to Rank (LTR) model that could potentially solve this coffee conundrum. For this project, I began by building a simple TensorFlow Ranking project to predict user-pair rankings of different coffees. I had some experience with TFR, and so it seemed like a natural fit.

However, I realized I had never made a ranking model from scratch before! I set about constructing a very hacky PyTorch ranking model to see if I could throw one together and learn something in the process. This is obviously not intended for a production system, and I made a lot of shortcuts along the way, but it has been an amazing pedagogical experience.

## Data

Our supreme goal is the following:

* develop a ranking model that learns the pairwise preferences of users
* apply this to predict the listwise ranking of `k` items
* What signal might lie in user and item feature combinations to produce a set of recommendations for that user pair?

To collect this data, I had to perform painful research of taste-testing amazing coffees with my wife. Each of us then rated them on a 10-point scale. The target value is simply the sum of our two scores (20 point maximum). The object of the model is to Learn to Rank coffees that we will both enjoy, and not just one member of any pair. The contextual data that we will be using is the following:

* ages of both users in the pair
* user ids that will be turned into embeddings

SweetMarias.com provides a lot of item data:

* the origin of the coffee
* Processing and cultivation notes tasting descriptions
* professional grading scores (100 point scale)

So for each training example, we will have the user data as the contextual information and each item’s feature set will be concatenated.

TensorFlow Ranking models are typically trained on data in ELWC format: ExampleListWithContext. You can think of it like a dictionary with 2 keys: CONTEXT and EXAMPLES (list). Inside each EXAMPLE is a dictionary of features per item you wish to rank.

For example, let us assume that I was searching for a new coffee to try out, and some candidate pool was presented to me of k=10 coffee varietals. An ELWC would consist of the context/user information, as well as a list of 10 items, each with its own feature set.

As I was no longer using TensorFlow Ranking, I made my own hacky ranking/list building aspect of this project. I grabbed random samples of k items from which we have scores and added them to a list. I split the first coffees I tried into a training set, and later examples became a small validation set to evaluate the model.

## Feature Intuition
In this toy example, we have a fairly rich dataset. Context-wise, we ostensibly know what the users age and can learn their respective preference embeddings. Through subsequent layers inside the LTR, these contextual features can be compared and contrasted. Does one user in the pair like dark, fruity flavors, while the other enjoys invigorating citrus and fruity notes in their cup?


For the item features, we have a generous helping of rich, descriptive text of each coffee’s tasting notes, origin, etc. More on this later, but the general idea is that we can capture the meaning of these descriptions and match the descriptions with the context (user-pair) data. Finally, we have some numerical features like the product expert tasting score per item that (should) have some semblance to reality.

## Preprocessing
A stunning shift is underway in text embeddings from when I was starting out in the ML industry. Long gone are the GLOVE and Word2Vec models that I used to use to try to capture some semantic meaning from a word or phrase. If you head on over to https://huggingface.co/blog/mteb, you can easily compare what are the latest and greatest embedding models for a variety of purposes.

For the sake of simplicity and familiarity, we will be using https://huggingface.co/BAAI/bge-base-en-v1.5 embeddings to help us project our text features into something understandable by a LTR model. Specifically we will use this for the product descriptions and product names that Sweet Marias provides.

We will also need to convert all of our user- and item-id values into an embedding space. PyTorch handles this beautifully with the Embedding Layers.

Finally we do some scaling on our float features with a simple RobustScaler. This can all happen inside our Torch Dataset class which then gets dumped into a DataLoader for training. The trick here is to separate out the different identifiers that will get past into the forward() call for PyTorch. This [article](https://towardsdatascience.com/deep-learning-using-pytorch-for-tabular-data-c68017d8b480) by Offir Inbar really saved me some time by doing just that!

```python

class DictDataset(Dataset):
    def __init__(self, data_dict, norm_target=1, scaler=None):
        self.norm_target = norm_target
        self.data_df = build_pandas_ranking(data_dict)
        self.scaler = scaler

        # Build out the features that are continuous variables
        self.float_features = []

        for feat in self.data_df.columns:
            valid = True
            for non_float_keyword in [
                "user_id",
                "product_id",
                "combined_score",
                "target",
            ]:
                if non_float_keyword in feat:
                    valid = False
                    continue
            if valid:
                self.float_features.append(feat)

        self.u_cats = [
            _feat for _feat in self.data_df.columns if _feat.startswith("user_id")
        ]
        self.i_cats = [
            _feat for _feat in self.data_df.columns if _feat.startswith("product_id")
        ]
        self.targets = [
            _feat
            for _feat in self.data_df.columns
            if _feat.startswith("combined_score")
        ]

        logging.debug(
            "Float features: %s",
            [_ for _ in self.float_features if "embedding" not in _],
        )

        if not self.scaler:
            # If we haven't trained a scaler yet, do so here
            self.scaler = RobustScaler().fit(self.data_df[self.float_features])
        # Scale the data from the float features into its own dataframe
        self.float_df = pd.DataFrame(
            self.scaler.transform(self.data_df[self.float_features]),
            columns=self.float_features,
        )
        # Drop these from the normal DF as they are un-scaled
        self.data_df = self.data_df.drop(self.float_features, axis=1)

    def __getitem__(self, index):

        return (
            torch.Tensor(self.float_df.iloc[index].values),
            torch.Tensor(self.data_df[self.u_cats].iloc[index].values),
            torch.Tensor(self.data_df[self.i_cats].iloc[index].values),
            torch.Tensor(
                self.data_df[self.targets].iloc[index].values / self.norm_target
            ),
        )
```
## Model Building and Training
The only interesting thing about the Torch training was ensuring that the 2 user embeddings (one for each rater) and the k coffees in the list for training had the correct embeddings and dimensions to pass through our neural network. With a few tweaks, I was able to get something out:


This forward pushes each training example into a single concatenated list with all of the features.

With so few data points (only 16 coffees were rated), it can be difficult to train a robust NN model. I often build a simple sklearn model side by side so that I can compare the results. Are we really learning anything?

Using the same data preparation techniques, I built a LogisticRegression multi-class classifier model, and then dumped out the .predict_proba() scores to be used as our rankings. What could our metrics say about the performance of these two models?
```python
    def forward(self, x, u_cats, i_cats):
        """
        Forward pass
        :param x: Float Tensor
        :param u_cats: User index tensor
        :param i_cats: Item index tensor
        :return: Predictions for this batch
        """
        curr_batch_size = len(u_cats)
        # Take User and Item embeddings for each value
        u_embs = self.user_embedding(u_cats.long())
        i_embs = self.item_embedding(i_cats.long())
        # Orient along the correct axis
        u_embs = u_embs.view(curr_batch_size, -1)
        i_embs = i_embs.view(curr_batch_size, -1)
        # Concat float values and embeddings together
        x = torch.cat([u_embs, i_embs, x], 1)
        return self.layers(x)
```
## Results
For the metrics, I chose to track two:

* Top (`k=1`) accuracy
* NDCG
The goal, of course, is to get the ranking correct for these coffees. NDCG will fit the bill nicely here. However, I suspected that the LogReg model might struggle with the ranking aspect, so I thought I might throw a simple accuracy in there as well. Sometimes you only want one really good cup of coffee and don’t need a ranking!

Without any significant investment in parameter tuning on my part, I achieved very similar results between the two models. SKLearn had slightly worse NDCG on the (tiny) validation set (0.9581 vs 0.950), but similar accuracy. I believe with some hyper-parameter tuning on both the PyTorch model and the LogReg model, the results could be very similar with so little data. But at least they broadly agree!

Future Work
I have a new batch of 16 pounds of coffee to start ranking to add to the model, and I deliberately added some lesser-known varietals to the mix. I hope to clean up the repo a bit and make it less of a hack-job. Also I need to add a prediction function for unseen coffees so that I can figure out what to buy next order!

One thing to note is that if you are building a recommender for production, it is often a good idea to use a real library built for ranking. TensorFlow Ranking, XGBoost, LambdaRank, etc. are accepted in the industry and have lots of the pain points ironed out.

Please check out the repo [here](https://github.com/franckjay/UserPairRecommendationEngine) and let me know if you catch any bugs! I hope you are inspired to train your own User-Pair model for ranking.