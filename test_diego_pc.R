# %% [markdown]
# # CommonLit Readability - Building Regression Models
# 

# %% [markdown]
# # 0. Importing Libraries and Data

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:09:09.713332Z","iopub.execute_input":"2021-07-18T16:09:09.715674Z","iopub.status.idle":"2021-07-18T16:09:15.395509Z"}}
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



# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:09:15.397795Z","iopub.execute_input":"2021-07-18T16:09:15.429748Z","iopub.status.idle":"2021-07-18T16:09:15.545991Z"}}
## Reading in files
train <- read.csv("../input/commonlitreadabilityprize/train.csv", stringsAsFactor = F, na.strings = c(""))
test <- read.csv("../input/commonlitreadabilityprize/test.csv", stringsAsFactor = F, na.strings = c(""))

# %% [markdown]
# **Now we can examine numeric variables like target and standard error.
# We start our EDA from the ‘target’ variable. Next, we’ll look at excerpts’ sources and then will make NLP analysis for the main variable - ‘excerpt’**

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:09:15.549390Z","iopub.execute_input":"2021-07-18T16:09:15.550748Z","iopub.status.idle":"2021-07-18T16:09:15.644541Z"}}

## Dimensions of data
dim(train)
dim(test)

# Top rows for training data
head(train, 3)

# Top rows for testing data
test

summary(train)
summary(test)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:09:15.647158Z","iopub.execute_input":"2021-07-18T16:09:15.648540Z","iopub.status.idle":"2021-07-18T16:09:16.511479Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:09:16.513972Z","iopub.execute_input":"2021-07-18T16:09:16.515457Z","iopub.status.idle":"2021-07-18T16:10:02.829164Z"}}
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


# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:10:02.831426Z","iopub.execute_input":"2021-07-18T16:10:02.832850Z","iopub.status.idle":"2021-07-18T16:10:03.182808Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:10:03.185147Z","iopub.execute_input":"2021-07-18T16:10:03.186514Z","iopub.status.idle":"2021-07-18T16:10:03.523652Z"}}
train %>% ggplot(aes(x=standard_error, y = target)) + geom_point()

# %% [markdown]
# Passages with higher standard error are either rated very high or very low in terms of ease of reading.

# %% [markdown]
# # 1. Building Regression Models Using TD-IDF Frequencies as features

# %% [markdown]
# What we did last time:
# - cleaned and preprossed dataset
# - converted text excerpts to numeric feature vectors with TD-IDF frequencies

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:10:03.526015Z","iopub.execute_input":"2021-07-18T16:10:03.527422Z","iopub.status.idle":"2021-07-18T16:10:03.544994Z"}}
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


# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:10:03.547366Z","iopub.execute_input":"2021-07-18T16:10:03.548760Z","iopub.status.idle":"2021-07-18T16:10:03.569833Z"}}
# split train in train and validations sets
## 80% of the sample size
smp_size <- floor(0.8 * nrow(train))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(train)), size = smp_size)

train <- train[train_ind, ]
val <- train[-train_ind, ]

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:10:03.572332Z","iopub.execute_input":"2021-07-18T16:10:03.573702Z","iopub.status.idle":"2021-07-18T16:10:14.911735Z"}}
dtm_train <- preprocess(train$excerpt) %>% dtm.generate(ng = 1)

train.dictionary <- c(names(dtm_train))
dtm_val <- preprocess(val$excerpt) %>% dtm.generate.with.dictionary(ng = 1, train.dictionary)
dtm_test <- preprocess(test$excerpt) %>% dtm.generate.with.dictionary(ng = 1, train.dictionary)
dim(dtm_train)
dim(dtm_val)
dim(dtm_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:10:14.913903Z","iopub.execute_input":"2021-07-18T16:10:14.915271Z","iopub.status.idle":"2021-07-18T16:10:18.849719Z"}}
# Adding cleaned excerpt as column
train <- cbind(train, data.frame(text=unlist(sapply(preprocess(train$excerpt), `[`, "content")), stringsAsFactors=F)[,c('text'), drop = FALSE])
val <- cbind(val, data.frame(text=unlist(sapply(preprocess(val$excerpt), `[`, "content")), stringsAsFactors=F)[,c('text'), drop = FALSE])
test <- cbind(test, data.frame(text=unlist(sapply(preprocess(test$excerpt), `[`, "content")), stringsAsFactors=F)$text)
colnames(train)[ncol(train)] <- 'cleaned_excerpt'
colnames(val)[ncol(val)] <- 'cleaned_excerpt'
colnames(test)[ncol(test)] <- 'cleaned_excerpt'

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:10:18.852145Z","iopub.execute_input":"2021-07-18T16:10:18.853545Z","iopub.status.idle":"2021-07-18T16:10:18.882004Z"}}
head(test, 3)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:10:18.884408Z","iopub.execute_input":"2021-07-18T16:10:18.885823Z","iopub.status.idle":"2021-07-18T16:10:19.424882Z"}}
# filter by deviation
dtm_train <- dtm_train %>% select(all_of(which(sapply(dtm_train, sd) >= 0.01)))
dim(dtm_train)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T17:27:09.525729Z","iopub.execute_input":"2021-07-18T17:27:09.529650Z","iopub.status.idle":"2021-07-18T17:27:11.140586Z"}}
train.dictionary <- c(names(dtm_train))
dtm_val <- preprocess(val$excerpt) %>% dtm.generate.with.dictionary(ng = 1, train.dictionary)
dtm_test <- preprocess(test$excerpt) %>% dtm.generate.with.dictionary(ng = 1, train.dictionary)
dim(dtm_train)
dim(dtm_val)
dim(dtm_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T17:27:35.313708Z","iopub.execute_input":"2021-07-18T17:27:35.315290Z","iopub.status.idle":"2021-07-18T17:27:43.987310Z"}}
# PCA
pca.model <- prcomp(dtm_train, center = TRUE,scale. = TRUE, rank = 50)
# apply PCA in val, test
train.pca <- pca.model$x
val.pca <- predict(pca.model, newdata = dtm_val)
test.pca <- predict(pca.model, newdata = dtm_test)
dim(train.pca)
dim(val.pca)
dim(test.pca)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T17:05:16.809538Z","iopub.execute_input":"2021-07-18T17:05:16.811489Z","iopub.status.idle":"2021-07-18T17:05:16.868519Z"}}
# join back with target
train.df <- cbind(train[,c('target','id'), drop = FALSE], train.pca)
val.df <- cbind(val[,c('target','id'), drop = FALSE], val.pca)
test.df <- test.pca

dim(train.df)
head(train.df)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:10:28.642732Z","iopub.execute_input":"2021-07-18T16:10:28.644267Z","iopub.status.idle":"2021-07-18T16:10:50.646365Z"}}
# fit a random forest model to our training set
fitRandomForest <- randomForest(target ~ ., data = train.df, ntrees = 20)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:10:50.649106Z","iopub.execute_input":"2021-07-18T16:10:50.650742Z","iopub.status.idle":"2021-07-18T16:10:50.852358Z"}}
# package with the rmse function
library(modelr)

# get the root mean square error for our new model, based on our train data
rmse(model = fitRandomForest, data = train.df)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:10:50.854942Z","iopub.execute_input":"2021-07-18T16:10:50.856433Z","iopub.status.idle":"2021-07-18T16:10:50.942288Z"}}
# # get the root mean square error for our new model, based on our validation data
rmse(model = fitRandomForest, data = val.df)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:10:50.944718Z","iopub.execute_input":"2021-07-18T16:10:50.946139Z","iopub.status.idle":"2021-07-18T16:10:51.026141Z"}}
pred <- predict(fitRandomForest, val.df)
head(cbind(pred, val[, c('target', 'excerpt')]))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:10:51.028492Z","iopub.execute_input":"2021-07-18T16:10:51.029909Z","iopub.status.idle":"2021-07-18T16:10:51.083603Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:10:51.087131Z","iopub.execute_input":"2021-07-18T16:10:51.088783Z","iopub.status.idle":"2021-07-18T16:10:51.117694Z"}}
text <- c("banana", "apple", "book")
my_corpus <- VCorpus(VectorSource(text))
dtm1 <- DocumentTermMatrix(my_corpus) %>% as.matrix()
dtm1

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:10:51.120009Z","iopub.execute_input":"2021-07-18T16:10:51.121406Z","iopub.status.idle":"2021-07-18T16:10:51.179231Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:10:51.181540Z","iopub.execute_input":"2021-07-18T16:10:51.183035Z","iopub.status.idle":"2021-07-18T16:11:42.809479Z"}}
pretrained_model <- read.word2vec(file = "../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin", normalize = TRUE)

# %% [markdown]
# ## What is the distance between bananas and apples Vs bananas and books now?

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:11:42.813538Z","iopub.execute_input":"2021-07-18T16:11:42.815250Z","iopub.status.idle":"2021-07-18T16:11:42.841426Z"}}
wv <- predict(pretrained_model, newdata = c("banana", "apple", "book"), type = "embedding")
distance(wv, method = "cosine")

# %% [markdown]
# ## king - man + woman = ?

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:11:42.844640Z","iopub.execute_input":"2021-07-18T16:11:42.846171Z","iopub.status.idle":"2021-07-18T16:11:45.196190Z"}}
wv <- predict(pretrained_model, newdata = c("king", "man", "woman"), type = "embedding")
wv <- wv["king", ] - wv["man", ] + wv["woman", ]
predict(pretrained_model, newdata = wv, type = "nearest", top_n = 3)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:11:45.199491Z","iopub.execute_input":"2021-07-18T16:11:45.201115Z","iopub.status.idle":"2021-07-18T16:11:49.642650Z"}}
set.seed(123)
model <- word2vec(x = train$cleaned_excerpt, type = "cbow", dim = 15, iter = 20)
embedding <- as.matrix(model) 
embedding <- cbind(rownames(embedding), data.table(embedding))
colnames(embedding)[1] <- 'word'
head(embedding)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:11:49.644901Z","iopub.execute_input":"2021-07-18T16:11:49.646266Z","iopub.status.idle":"2021-07-18T16:11:49.676101Z"}}
lookslike <- predict(model, c("king"), type = "nearest", top_n = 5)
lookslike

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:11:49.679096Z","iopub.execute_input":"2021-07-18T16:11:49.680641Z","iopub.status.idle":"2021-07-18T16:11:49.954534Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:11:49.957216Z","iopub.execute_input":"2021-07-18T16:11:49.959287Z","iopub.status.idle":"2021-07-18T16:11:49.984014Z"}}
head(valwords)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T17:28:25.167862Z","iopub.execute_input":"2021-07-18T17:28:25.169462Z","iopub.status.idle":"2021-07-18T17:28:27.375415Z"}}
valwords['test'] <- 1

#valwords

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


#head(trainemd.df)
#dim(trainemd.df)
#head(trainemd.df)
#dim(trainemd.df)

train.df$target <- NULL
trainemd.df <- inner_join(trainemd.df, train.df, by = c('id'), copy = TRUE)

val.df$target <- NULL
valemd.df <- inner_join(valemd.df, val.df, by = c('id'), copy = TRUE)

#test.df$target <- NULL
#testemd.df <- inner_join(testemd.df, test.df, by = c('id'), copy = TRUE)

trainemd.df
valemd.df
testemd.df
#train.df
#cbind(train.df, val[, c('target', 'excerpt')])

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T17:40:37.485306Z","iopub.execute_input":"2021-07-18T17:40:37.487298Z","iopub.status.idle":"2021-07-18T17:41:05.227665Z"}}
# fit a random forest model to our training set
fitRandomForest2 <- randomForest(target ~ ., data = trainemd.df, ntrees = 10)

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T17:40:15.709096Z","iopub.execute_input":"2021-07-18T17:40:15.710845Z","iopub.status.idle":"2021-07-18T17:40:16.016355Z"}}
# get the root mean square error for our new model, based on our train data
rmse(model = fitRandomForest2, data = trainemd.df)

# # get the root mean square error for our new model, based on our validation data
rmse(model = fitRandomForest2, data = valemd.df)

pred <- predict(fitRandomForest2, valemd.df)
head(cbind(pred, val[, c('target', 'excerpt')]))



# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:11:59.878871Z","iopub.execute_input":"2021-07-18T16:11:59.880313Z","iopub.status.idle":"2021-07-18T16:11:59.943485Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T16:11:59.945966Z","iopub.execute_input":"2021-07-18T16:11:59.947459Z","iopub.status.idle":"2021-07-18T16:11:59.958964Z"}}
# write.table(data.table(train)[,":="(excerpt=gsub("\n", "", excerpt))], "train.csv", sep="\t", eol = "\r", row.names = FALSE)
# write.table(test, "test.csv", sep="\t", eol = "\r", row.names = FALSE)