# Ask Reddit Score Prediction

Sherman ALINE & Le Anh DUNG (Kaggle: Sherman and Dung (TSE))

## Archive 

* `project_reddit.ipynb` - this file contains all code steps for running
* `something.csv` - the results of running our model.

## Rerunning & Dependencies

### Rerunning
In order to rerun our code, include a csv file in the same directory as the notebook with the title `comments_students.csv` with the following index names, you can check the index names by running 
```
sed 1q comments_students.csv
```
which should return 
```
"created_utc","ups","subreddit_id","link_id","name","subreddit","id","author","body","parent_id"
```
Because the computation is intensive, we save the resulted dataframes each time and re-use it later. You have to change the path to where you want to save them. For example

```
path = r'E:\\M2 EconStat\\Web Mining\\Project\\comments_students.csv'
path = r'E:\\M2 EconStat\\Web Mining\\root_included.csv'
```

### Dependencies

* pandas
* numpy
* scipy
* matplotlib
* networkx
* dask
* nltk
* multiprocessing
* sklearn

## Features

In implementing features, we followed two main approaches. First, building graphs and creating metrics for them, and second using text mining techniques.
 
### Graphs

In the process of research past work on feature extraction from reddit comments we came upon [Conversation Modeling on Reddit Using a Graph-Structured LSTM by Victoria Zayats and Mari Ostendorf](https://www.aclweb.org/anthology/Q18-1009.pdf) which gave us some context for what graph-related features work well.

We build a separate graph characterized by its `root` in our dataset and the comments replying to it. This approah was best as each subgraph is unconnected anyways, and this approach allows for parallization. 

In fact, because our dataset does not contain textual data for the original post, 1st level replies (replies that are to the original post, not another comment) do not have an edge leading to the original post. By building these subgraphs we can keep track of which 1st level replies correspond to which post. In fact, these subgraph allowed us to extract some features in simpler ways in the next step.

Below is the list of features we extracted.

### Motivation for the choice of tree-related features

In this section,

- I use the words "node" and "comment" interchangably.

- If not stated otherwise, I meant by "the tree" the one characterized by the **root** of the comment.

Then

- `depth`: the shortest path from its root to the comment.

If the `depth` is too large, the comments are less likely to be seen by other users, and thus less likely to be upvoted or downvoted.

- `num_siblings`: the number of nodes within the tree that share the same parent.

- `num_comments_author`: the number of comments made by the author within the tree. If the author made more than $1$ comments within the tree, then these comments have the same value of `num_comments_author`.

- `num_previous_comments`: the number of nodes whose posting time is stricly sooner than that of the comment.

- `num_later_comments`: the number of nodes whose posting time is stricly later than that of the comment.

I add `num_siblings`, `num_comments_author`, `num_previous_comments`, and `num_later_comments` because I have seen them in the paper [Conversation Modeling on Reddit Using a Graph-Structured LSTM](https://www.aclweb.org/anthology/Q18-1009.pdf).

- `time_since_root`: the interval of posting time between the comment and its **root**.

- `time_since_parent`: the interval of posting time between the comment and its direct parent.

If `time_since_root` or `time_since_parent` is small, then the comment are more likely to be seen by later users, and thus more likely to be upvoted or downvoted.

- `num_comments_subtree`: the number of nodes in the tree.

- `height_subtree`: the longest (directed) path in the tree.

- `num_children`: the number of **direct** replies to the comment.

`num_comments_subtree`, `height_subtree`, and `num_children` are size proxies of the tree rooted from the comment. It seems that interesting comments attracts more replies and thus its induced tree has bigger size.

We could not find any package containing functions to compute all features, except for *depth of the comment*, *height of the subtree*, and *size of that subtree*. Luckily, our trees are of a special kind, [arborescence](https://www.wikiwand.com/en/Tree_(graph_theory)#/Rooted_tree). This allows us to utilize package `dask` and highly optimized functions from package `pandas` to speed up computation. First, we compute features related to the tree characterized by column `root`. Comments with the same `root` are in the same tree.


### Texts

For text features, we used domain knowledge about certain specific features of the text which may have value. After extracitng these features, we use text-cleaning and stemming to create the majority of the features.

#### Domain Knowledge Features

* Emojis - emoticons may be used to detect sentiment. Additionally, they may signal a overly-casual post which may not be recieved well.

* Sentiment in post - funny comments, as well as serious and informative posts will get upvotes. In long comment threads, a hostile tone may get upvotes (for one party, while the other party with hostile tone will get many downvotes).

* Length - long comments may be an indication of high-information or high-effort post.

* Punctuation - this is another indicator that a post is likely to be well-written, correlated with useful information and understandable writing etc. All things which will get upvotes on AskReddit. Extremely large amounts of punctuation may also indicate visual text jokes being made using markdown.

* Paragraphs - indicator of high amiount of content/contribution or high effort. We detect paragraphs by the number of double spaces. Doublespaces are removed on reddit. Through manual investigation, I have determined that double spaces in our corpora are new lines.

* Capital letters - indicator of high quality post, or very low quality post if THE NUMBER OF CAPITALS IS WAY TOO HIGH!!!!

* Number of links in a post - citations generally get upvotes, an indicator of sharing information.
	* Markdown Links - these are properly formatted and the text referring to the link must be extracted
	* Pasted Link - these contain no formatting and indicate a lower effort reply. They are also simpler to clean.

* Refer to other users `/u/` - indicates a discussion is occuring.

* Refer to subreddit `/r/` - posts referring to another subreddit are often of two types - humorous jokes or pointing someone to useful resources. In either case it is going to behave differently than a simple text post.

* Quotes - quotes are another indicator of a high-effort contribution, as well as evidence of detailed discussion with another user. If it is necessary to use quotes, it is likely the user is replying to specific parts of the text which they wish to specify. Additionally, quoted text must be removed because it will introduce false covariance with other users at the TF-IDF step.

* Comment Length - this could be an indicator of the level of detail in a post, longer posts may get more upvotes. They are both more visible and contain more information.

#### Cleaning & Stemming

After the initial building of features, and some cleaning along the way, we prepare the text for TF-IDF. This involves removing punctuation and capitals, and then using stemming.

For stemming, we used Lancaster stemmer. This stemmer is known to be quite fast, as well as the most agressive common form of stemming. We used this stemmer because our dataset is so large, with so many different authors using different writing styles and different typing mistakes. Before stemming we found the highest document frequency was 0.2. We need an aggressive stemmer to be able to combine many variations of words.

We decided to use stemming instead of lemmatization due to computing and time contraints.

#### TF-IDF and Feature Selection

When using TFIDFVectorizer from sklearn, we have the option of limiting our feature generation to terms which have a certain document frequency. In order to investigate the optimal min_df and max_df we examined what features correspond to different document frequency levels.

We determined optimal constraints for our case to be max_df = 0.018 and min_df = 0.00011. Terms with a higher document frequency than our max tend to be very ubiquitous and lack a signficant meaning, such as 
```
kind yeah play best talk thought
```
Terms with a lower document frequency than our minimum tended to be proper nouns, and semi-uncommon slang such as 
```
bieber, meow, morrowind, obama, yooooo, hubby, runescape
```

 These are too uncommon provide little information.
 
 Finally, the last step is to narrow down our number which is manageable for our computing resources. We use VarianceThreshold from sklearn to select features with the highest variance. We choose the threshhold arbtrarily so that it gives us a small enough number of features.
 
 Variance Threshholing is used in place of more advanced techniques as it is the least resource intensive method. Other methods are model-based approaches which will take longer to execute. 
 
 We believe it is rational to believe low-variance features will provide little information to our model. If a feature is uniform across observations it is unlikely to be significant in the model.

## Model

### Subsampling

Because training set contains over $3$ millions rows and thus takes too much time to run, I create a subsample whose size is $10\%$ of the original training set. To make this subsample more representative of the original distribution, I divide the training set into $100$ categories. The $i$-th category contains comments whose percentile of their `ups` belong to the interval $[i, i+1]$. For each category, we randomly select $10\%$ of the total comments in that category.

We use function `pandas.qcut` to get the percentile (w.r.t `ups`) of each comment. It's mentioned in the [documentation](https://pandas.pydata.org/docs/reference/api/pandas.qcut.html):

>q: Number of quantiles. 10 for deciles, 4 for quartiles, etc. Alternately array of quantiles, e.g. [0, .25, .5, .75, 1.] for quartiles.

So to get percentile, we should set `q` equal to $100$. However, setting `q = 100` does not do what we expect. We still don't know why. With trial and error, we found that `q = 1710` produces $100$ percentile as expected.

At the end, we use 3 models

- Random forest model
- XGBoost
- lightGBM