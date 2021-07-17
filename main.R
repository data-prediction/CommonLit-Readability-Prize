# %% [markdown]
# # CommonLit Readability - Building Regression Models
# 

# %% [markdown]
# # 0. Importing Libraries and Data

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T09:53:34.610935Z","iopub.execute_input":"2021-07-17T09:53:34.613942Z","iopub.status.idle":"2021-07-17T09:53:40.068068Z"}}
## Importing packages
library(tidyverse)
library(stringi)
library(tm)
library(irlba)
library(RColorBrewer)
library(wordcloud)
library(gridExtra)
library(caret)
library(doParallel)
library(tidytext) # tidy implimentation of NLP methods
library(topicmodels) # for LDA topic modelling 
library(tm) # general text mining functions, making document term matrixes
library(SnowballC) # for stemming
library(ggwordcloud)
library(tokenizers)
library(stopwords)
library(ggExtra)
library(RWeka)
library(data.table)
library(randomForest) # for random forests
library(word2vec)

library(patchwork)
library(ggwordcloud)
library(tidytext)
library(ggtext)
library(sentimentr)


options(warn=-1)



# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T20:23:51.176384Z","iopub.execute_input":"2021-07-13T20:23:51.209919Z","iopub.status.idle":"2021-07-13T20:23:51.334908Z"}}
## Reading in files
train <- read.csv("../input/commonlitreadabilityprize/train.csv", stringsAsFactor = F, na.strings = c(""))
test <- read.csv("../input/commonlitreadabilityprize/test.csv", stringsAsFactor = F, na.strings = c(""))

# %% [markdown]
# **Now we can examine numeric variables like target and standard error.
# We start our EDA from the ‘target’ variable. Next, we’ll look at excerpts’ sources and then will make NLP analysis for the main variable - ‘excerpt’**

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T20:38:24.146921Z","iopub.execute_input":"2021-07-13T20:38:24.148636Z","iopub.status.idle":"2021-07-13T20:38:24.224978Z"}}

## Dimensions of data
dim(train)
dim(test)

# Top rows for training data
head(train, 3)

# Top rows for testing data
test

summary(train)
summary(test)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T20:44:51.133905Z","iopub.execute_input":"2021-07-13T20:44:51.136012Z","iopub.status.idle":"2021-07-13T20:44:52.075363Z"}}
theme_set(theme_minimal())
my_theme <- theme(plot.title = element_text(hjust = 0.5, face = 'bold', size = 18),
        plot.subtitle = element_text(hjust = 0.5, size = 13),
        axis.title = element_text(face = 'bold', size = 15),
        axis.text = element_text(size = 13))


p1 <- train %>% 
  count(license) %>% 
  ggplot(aes(n, reorder(license, n))) +
  geom_col(fill = '#00b3b3') +
  geom_label(aes(label = n)) + 
  labs(x = 'Count',
       y = 'License',
       subtitle = '"train.csv"') +
  my_theme

p2 <- test %>% 
  count(license) %>% 
  ggplot(aes(n, reorder(license, n))) +
  geom_col(fill = '#00b3b3') +
  geom_label(aes(label = n)) + 
  labs(x = 'Count',
       y = 'License',
       subtitle = '"test.csv"') +
  my_theme

annot = theme(plot.title = element_text(hjust = 0.6, face = 'bold', size = 17))
p1 + p2 + plot_layout(design = 'AAABB') + plot_annotation(title = 'License frequency', 
                                                          theme = annot)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T20:53:13.121945Z","iopub.execute_input":"2021-07-13T20:53:13.123792Z","iopub.status.idle":"2021-07-13T20:53:57.719957Z"},"jupyter":{"source_hidden":true}}
p1 <- train %>% 
  ggplot(aes(target)) +
  geom_density(fill = '#00b3b3', size = 1) +
  geom_vline(aes(xintercept = mean(train$target)), col = 'red', 
             size = 1.5, linetype = 2) +
  geom_label(aes(x = mean(train$target), y = 0.4,
                 label = paste0('Mean: ', round(mean(train$target), 3))),
             col = 'red', size = 5.5) +
  labs(x = '',
       y = 'Density',
       title = 'Target',
       subtitle = '"train.csv"') +
  my_theme

mean <- round(mean(train$standard_error), 3)
median <- round(median(train$standard_error), 3)
p2 <- train %>% 
  ggplot(aes(standard_error)) +
  geom_density(fill = '#00b3b3', size = 1) +
  geom_vline(aes(xintercept = mean), col = 'red', 
             size = 1.5, linetype = 2) +
  geom_vline(aes(xintercept = median), col = 'blue', 
             size = 1.5, linetype = 2) +
  geom_label(aes(x = mean + 0.1, y = 17,
                 label = paste0('Mean: ', mean)),
             col = 'red', size = 5.5) +
  geom_label(aes(x = median - 0.11, y = 17,
                 label = paste0('Median: ', median)),
             col = 'blue', size = 5.5) +
  labs(x = '',
       y = '',
       title = 'Standard error',
       subtitle = '"train.csv"') +
  my_theme

p3 <- train %>% 
  ggplot(aes(x = target)) + 
  geom_boxplot(fill = '#00b3b3', size = 1) +
  labs(x = '') +
  theme(axis.text = element_blank())

p4 <- train %>% 
  ggplot(aes(x = standard_error)) + 
  geom_boxplot(fill = '#00b3b3', size = 1) +
  labs(x = '') +
  theme(axis.text = element_blank())

design <- 'AABB
AABB
AABB
AABB
CCDD'

p1 + p2 + p3 + p4 + plot_layout(design = design)


# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T21:02:02.863252Z","iopub.execute_input":"2021-07-13T21:02:02.864744Z","iopub.status.idle":"2021-07-13T21:02:03.204685Z"},"jupyter":{"source_hidden":true}}
p1 <- train %>%
  ggplot( aes(x=target)) +
    geom_density(fill="#000000", alpha=0.8)

p2 <- train %>%
  ggplot( aes(x=standard_error)) +
    geom_density(fill="#000000", alpha=0.8)

grid.arrange(p1,p2,ncol=2)

summary(train$target)

p <- ggplot(train, aes(x=target, y=standard_error)) +
      geom_point() +
      theme(legend.position="none")
 

# %% [markdown]
# The description of target is ease of reading:  higher numbers mean easier to read passages.
# 
# The distribution appears normal.
# 
# Standard error has a more skewed distribution.

# %% [markdown]
# **Relationship between target and standard error.**

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T21:01:41.898228Z","iopub.execute_input":"2021-07-13T21:01:41.899925Z","iopub.status.idle":"2021-07-13T21:01:42.234311Z"},"jupyter":{"source_hidden":true}}
train %>% ggplot(aes(x=standard_error, y = target)) + geom_point()

# %% [markdown]
# Passages with higher standard error are either rated very high or very low in terms of ease of reading.

# %% [markdown]
# # 1. Building Regression Models Using TD-IDF Frequencies as features

# %% [markdown]
# What we did last time:
# - cleaned and preprossed dataset
# - converted text excerpts to numeric feature vectors with TD-IDF frequencies

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T10:03:44.933845Z","iopub.execute_input":"2021-07-12T10:03:44.935481Z","iopub.status.idle":"2021-07-12T10:03:44.955764Z"}}
preprocess <- function(string){

  corpus <- VCorpus(VectorSource(string))
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeNumbers) 
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, content_transformer(gsub), pattern = "\\W", replacement = " ")
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, stemDocument)
  corpus <- tm_map(corpus, removeWords, stopwords("english")) 
  corpus
}

dtm.generate <- function(corpus, ng){

#   options(mc.cores=8) 
  tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = ng, max = ng)) # create n-grams
  dtm <- DocumentTermMatrix(corpus, control = list(tokenize = tokenizer, wordLengths= c(4, Inf),
                                                   weighting = function(x) weightTfIdf(x, normalize = TRUE)))  %>% 
                                                   as.matrix() %>% 
                                                   as.data.table()
  dtm
}
                                                   
 dtm.generate.with.dictionary <- function(corpus, ng, dict){

#   options(mc.cores=8) 
  tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = ng, max = ng)) # create n-grams
  dtm <- DocumentTermMatrix(corpus, control = list(dictionary = dict, tokenize = tokenizer, wordLengths= c(4, Inf),
                                                   weighting = function(x) weightTfIdf(x, normalize = TRUE))) %>% 
                                                   as.matrix() %>%
                                                   as.data.table()
  dtm
}
                                                   

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T10:04:42.589967Z","iopub.execute_input":"2021-07-12T10:04:42.591572Z","iopub.status.idle":"2021-07-12T10:04:42.613382Z"}}
# split train in train and validations sets
## 80% of the sample size
smp_size <- floor(0.8 * nrow(train))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(train)), size = smp_size)

train <- train[train_ind, ]
val <- train[-train_ind, ]

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T10:04:44.748541Z","iopub.execute_input":"2021-07-12T10:04:44.750077Z","iopub.status.idle":"2021-07-12T10:04:55.014121Z"}}
dtm_train <- preprocess(train$excerpt) %>% dtm.generate(ng = 1)

train.dictionary <- c(names(dtm_train))
dtm_val <- preprocess(val$excerpt) %>% dtm.generate.with.dictionary(ng = 1, train.dictionary)
dtm_test <- preprocess(test$excerpt) %>% dtm.generate.with.dictionary(ng = 1, train.dictionary)
dim(dtm_train)
dim(dtm_val)
dim(dtm_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T10:05:00.558421Z","iopub.execute_input":"2021-07-12T10:05:00.559925Z","iopub.status.idle":"2021-07-12T10:05:04.354747Z"}}
# Adding cleaned excerpt as column
train <- cbind(train, data.frame(text=unlist(sapply(preprocess(train$excerpt), `[`, "content")), stringsAsFactors=F)[,c('text'), drop = FALSE])
val <- cbind(val, data.frame(text=unlist(sapply(preprocess(val$excerpt), `[`, "content")), stringsAsFactors=F)[,c('text'), drop = FALSE])
test <- cbind(test, data.frame(text=unlist(sapply(preprocess(test$excerpt), `[`, "content")), stringsAsFactors=F)$text)
colnames(train)[ncol(train)] <- 'cleaned_excerpt'
colnames(val)[ncol(val)] <- 'cleaned_excerpt'
colnames(test)[ncol(test)] <- 'cleaned_excerpt'

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T10:05:04.357268Z","iopub.execute_input":"2021-07-12T10:05:04.358651Z","iopub.status.idle":"2021-07-12T10:05:04.389688Z"}}
head(test, 3)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T10:05:08.024307Z","iopub.execute_input":"2021-07-12T10:05:08.025932Z","iopub.status.idle":"2021-07-12T10:05:08.58131Z"}}
# filter by deviation
dtm_train <- dtm_train %>% select(all_of(which(sapply(dtm_train, sd) >= 0.01)))
dim(dtm_train)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T10:05:09.458336Z","iopub.execute_input":"2021-07-12T10:05:09.459905Z","iopub.status.idle":"2021-07-12T10:05:10.781814Z"}}
train.dictionary <- c(names(dtm_train))
dtm_val <- preprocess(val$excerpt) %>% dtm.generate.with.dictionary(ng = 1, train.dictionary)
dtm_test <- preprocess(test$excerpt) %>% dtm.generate.with.dictionary(ng = 1, train.dictionary)
dim(dtm_train)
dim(dtm_val)
dim(dtm_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T10:05:28.87217Z","iopub.execute_input":"2021-07-12T10:05:28.873877Z","iopub.status.idle":"2021-07-12T10:05:35.492014Z"}}
# PCA
pca.model <- prcomp(dtm_train, center = TRUE,scale. = TRUE, rank = 50)
# apply PCA in val, test
train.pca <- pca.model$x
val.pca <- predict(pca.model, newdata = dtm_val)
test.pca <- predict(pca.model, newdata = dtm_test)

dim(train.pca)
dim(val.pca)
dim(test.pca)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T10:05:40.168982Z","iopub.execute_input":"2021-07-12T10:05:40.170606Z","iopub.status.idle":"2021-07-12T10:05:40.224148Z"}}
# join back with target
train.df <- cbind(train[,c('target'), drop = FALSE], train.pca)
val.df <- cbind(val[,c('target'), drop = FALSE], val.pca)
test.df <- test.pca

dim(train.df)
head(train.df)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T10:05:44.044325Z","iopub.execute_input":"2021-07-12T10:05:44.045898Z","iopub.status.idle":"2021-07-12T10:06:06.836418Z"}}
# fit a random forest model to our training set
fitRandomForest <- randomForest(target ~ ., data = train.df, ntrees = 20)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T10:06:06.840115Z","iopub.execute_input":"2021-07-12T10:06:06.841723Z","iopub.status.idle":"2021-07-12T10:06:07.045863Z"}}
# package with the rmse function
library(modelr)

# get the root mean square error for our new model, based on our train data
rmse(model = fitRandomForest, data = train.df)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T10:06:07.049353Z","iopub.execute_input":"2021-07-12T10:06:07.050916Z","iopub.status.idle":"2021-07-12T10:06:07.124807Z"}}
# # get the root mean square error for our new model, based on our validation data
rmse(model = fitRandomForest, data = val.df)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T10:06:25.708236Z","iopub.execute_input":"2021-07-12T10:06:25.71188Z","iopub.status.idle":"2021-07-12T10:06:25.829582Z"}}
pred <- predict(fitRandomForest, val.df)
head(cbind(pred, val[, c('target', 'excerpt')]))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T10:06:39.533671Z","iopub.execute_input":"2021-07-12T10:06:39.53541Z","iopub.status.idle":"2021-07-12T10:06:39.588074Z"}}
# predict on test 
pred.test <- predict(fitRandomForest, test.df)
cbind(pred.test, test[, c('id','excerpt')])

# %% [markdown]
# # 2. Word Embeddings
# ______

# %% [markdown]
# What was the problem so far?
# 
# **Problem 1**: Too many columns compared to number of observations in training data.
# 
# Solutions: 
# - Dimensionality reduction
#     - PCA
#     - **Word Embeddings**
# - Extending training data
# 
# **Problem 2**: The respresentation does not incorporate the semantic similarity of words. 
# 
# Solutions:
#    - **Word Embeddings**

# %% [markdown]
# > **Word Embeddings:** the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers.
# 
# Word embeddings are a type of word representation that allows words with similar meaning to have a similar representation.
# 
# Key features: 
# - dense respresentation
# - words with similar meaning have similar vectors

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T11:00:03.707776Z","iopub.execute_input":"2021-07-12T11:00:03.709428Z","iopub.status.idle":"2021-07-12T11:00:03.736207Z"}}
text <- c("banana", "apple", "book")
my_corpus <- VCorpus(VectorSource(text))
dtm1 <- DocumentTermMatrix(my_corpus) %>% as.matrix()
dtm1

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T11:00:05.960758Z","iopub.execute_input":"2021-07-12T11:00:05.962453Z","iopub.status.idle":"2021-07-12T11:00:05.987138Z"}}
library(philentropy)

# calculating the distance between documents
distance(dtm1, method = "cosine")

# %% [markdown]
# Bananas and apples the same similar as bananas and books.

# %% [markdown]
# ## Word2Vec 
# ## “Tell me who your friends are, and I will tell you who you are.”
# 
# Word2Vec is a method to construct a word embedding based on its context developed by Google.
# 
# **The Core idea is that words that occur in the same contexts tend to have similar meanings.**
# 
# Two different learning models were introduced that can be used as part of the word2vec approach to learn the word embedding; they are:
# - Continuous Bag-of-Words, or CBOW model
# - Skip-Gram Model
# 
# The **CBOW model** learns the embedding by predicting the **current word** based on **its context**. The continuous **skip-gram model** learns by predicting the **surrounding words given a current word.**
# 
# Example: "Italy is a champion of Euro 2021"
# - **CBOW**: given context words in a given window predict the target word  
# - **Skip-Gram**: given target word "champion" predict the neighbouring words in a given window
# 
# ![image.png](attachment:87a86c26-806e-4e35-9609-869247829cea.png)

# %% [markdown]
# ## But where are the word embeddings?
# ![image.png](attachment:8552b1e7-c499-479a-811f-ae4bfc543904.png)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T10:35:40.574939Z","iopub.execute_input":"2021-07-12T10:35:40.577426Z","iopub.status.idle":"2021-07-12T10:36:06.809835Z"}}
pretrained_model <- read.word2vec(file = "../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin", normalize = TRUE)

# %% [markdown]
# ## What is the distance between bananas and apples Vs bananas and books now?

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T11:00:13.022872Z","iopub.execute_input":"2021-07-12T11:00:13.024869Z","iopub.status.idle":"2021-07-12T11:00:13.065209Z"}}
wv <- predict(pretrained_model, newdata = c("banana", "apple", "book"), type = "embedding")
distance(wv, method = "cosine")

# %% [markdown]
# ## king - man + woman = ?

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T11:03:41.178996Z","iopub.execute_input":"2021-07-12T11:03:41.180597Z","iopub.status.idle":"2021-07-12T11:03:43.657033Z"}}
wv <- predict(pretrained_model, newdata = c("king", "man", "woman"), type = "embedding")
wv <- wv["king", ] - wv["man", ] + wv["woman", ]
predict(pretrained_model, newdata = wv, type = "nearest", top_n = 3)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T10:25:25.542417Z","iopub.execute_input":"2021-07-12T10:25:25.545076Z","iopub.status.idle":"2021-07-12T10:25:30.179863Z"}}
set.seed(123)
model <- word2vec(x = train$cleaned_excerpt, type = "cbow", dim = 15, iter = 20)
embedding <- as.matrix(model) 
embedding <- cbind(rownames(embedding), data.table(embedding))
colnames(embedding)[1] <- 'word'
head(embedding)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T08:13:10.714286Z","iopub.execute_input":"2021-07-12T08:13:10.716036Z","iopub.status.idle":"2021-07-12T08:13:10.749766Z"}}
lookslike <- predict(model, c("king"), type = "nearest", top_n = 5)
lookslike

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T08:08:46.006023Z","iopub.execute_input":"2021-07-12T08:08:46.008122Z","iopub.status.idle":"2021-07-12T08:08:46.254557Z"}}
trainwords <- train %>% 
              select(id, target, cleaned_excerpt) %>% 
              unnest_tokens(word, cleaned_excerpt) %>% 
              filter(nchar(word) > 3)
valwords <- val %>% 
              select(id, target, cleaned_excerpt) %>% 
              unnest_tokens(word, cleaned_excerpt) %>% 
              filter(nchar(word) > 3)
testwords <- test %>% 
              select(id, cleaned_excerpt) %>% 
              unnest_tokens(word, cleaned_excerpt) %>% 
              filter(nchar(word) > 3)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T08:08:13.991938Z","iopub.execute_input":"2021-07-12T08:08:13.994113Z","iopub.status.idle":"2021-07-12T08:08:14.025607Z"}}
head(valwords)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T08:08:59.306846Z","iopub.execute_input":"2021-07-12T08:08:59.308636Z","iopub.status.idle":"2021-07-12T08:09:01.238256Z"}}
trainemd.df <- merge(trainwords, embedding, by = "word") %>% 
                    group_by(id, target) %>%
                    select(-'word') %>%
                    summarise_all("mean")
valemd.df <- merge(valwords, embedding, by = "word") %>% 
                    group_by(id, target) %>%
                    select(-'word') %>%
                    summarise_all("mean")
testemd.df <- merge(testwords, embedding, by = "word") %>% 
                    group_by(id) %>%
                    select(-'word') %>%
                    summarise_all("mean")

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T08:09:06.865092Z","iopub.execute_input":"2021-07-12T08:09:06.866738Z","iopub.status.idle":"2021-07-12T08:09:16.077907Z"}}
# fit a random forest model to our training set
fitRandomForest2 <- randomForest(target ~ ., data = trainemd.df, ntrees = 20)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T08:09:20.05939Z","iopub.execute_input":"2021-07-12T08:09:20.061271Z","iopub.status.idle":"2021-07-12T08:09:20.343137Z"}}
# get the root mean square error for our new model, based on our train data
rmse(model = fitRandomForest2, data = trainemd.df)

# # get the root mean square error for our new model, based on our validation data
rmse(model = fitRandomForest2, data = valemd.df)

pred <- predict(fitRandomForest2, valemd.df)
head(cbind(pred, val[, c('target', 'excerpt')]))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T08:11:48.171021Z","iopub.execute_input":"2021-07-12T08:11:48.172686Z","iopub.status.idle":"2021-07-12T08:11:48.218464Z"}}
cbind(predict(fitRandomForest2, testemd.df), test[, c('excerpt')])

# %% [markdown]
# # To Do Untill 15 July
# 
# - Try additional Feature engineering (e.g. excerpt length, average sentence length)
# - Try using a pretrained embeddings (word2vec, GloVe, doc2vec) 
# - Try Including External Publicly Available Data
# - Build 3 models:
#     - train on train set, validate performance and choose hyperparameters on validation set
#     - get predictions on test data, compare RMSE and choose 1 final model 
#     - retrain model on whole dataset and prepare the submission file 

# %% [code] {"execution":{"iopub.status.busy":"2021-07-12T07:26:38.494166Z","iopub.execute_input":"2021-07-12T07:26:38.49569Z","iopub.status.idle":"2021-07-12T07:26:38.535206Z"}}
# write.table(data.table(train)[,":="(excerpt=gsub("\n", "", excerpt))], "train.csv", sep="\t", eol = "\r", row.names = FALSE)
# write.table(test, "test.csv", sep="\t", eol = "\r", row.names = FALSE)
