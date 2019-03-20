#install.packages(tidyverse)
#install.packages(sentimentr)
#install.packages(caret)
#install.packages(sentimentr)
#install.packages(quanteda)
#install.packages(e1071)

library(tidyverse)
library(sentimentr)
library(caret)
library(sentimentr)
library(quanteda)
library(e1071)


reviews <- read_csv('data/Reviews.csv')

# Check the dimensions
reviews %>% 
  dim

# Take a look at the first entries
reviews %>% 
  head()

# Since the names are all in upper case, I'll turn it to lower. Also, I'm only interested in the score 
# and text variables.

# Turn the dataset names to lowercase
names(reviews) <- reviews %>% 
  names %>% 
  tolower

# Filter
reviews <- reviews %>% 
  select(text, score)

reviews %>% 
  head()

# I won't bother in cleaning the text. i.e. mispellings, emojis and everything else. I'll deal with the
# text reviews 'as it is'.

# Plot our target variable to see its distribution
reviews %>% 
  pull(score) %>% 
  qplot(binwidth=1, 
        main='Number of Scores', xlab='Scores', ylab='N')

# It's a very unbalanced dataset. There's a lot of 5 when compared to the other existing scores. 
# However, I'll no be dealing with it here either.

# Check how the number of sentences are distributed.
n_sentences <- reviews %>% 
  pull(text) %>% 
  nsentence()

# Check the range
n_sentences %>% 
  range

#Plot the number of sentences distribution
n_sentences %>% 
  qplot(binwidth=20, 
        main='Number of Sentences per Text', xlab='Number of Sentences', ylab='N')

# The majority of the texts reviews have less than 20 sentences. So I'll narrow it down to the 
# maximum of 10 sentences per review. # Since it'll, later, become a sparse dataset, the sentences 
# reduction will be an important step.

index <- n_sentences<=10
reviews <- reviews[index, ]
reviews %>% 
  dim()

# For the sake of simplicity, and to improve calculations time in this project, I'll remove the scores 
# equals 3, i.e. neutral score, and reduce the dataset to 150.000 entries

reviews <- reviews %>% 
  filter(score!=3)

reviews %>% 
  dim()

# Turn scores into binary terms: Good (4 and 5) and Bad (1 and 2). Also, turn it into factors
reviews <- reviews %>% 
  mutate(score=ifelse(score>2, 'Good', 'Bad')) %>% 
  mutate(score=as.factor(score))

reviews %>% 
  pull(score) %>% 
  qplot(main='Binary Scores', ylab='N')

###########################################################

#                         Modelling

###########################################################

# Separate the sentences of each text review. This improves the RAM usage and the speed of the 
# 'sentiment extraction' by the sentimentr package

reviews <- reviews %>% 
  get_sentences(text)

# Now, extract the sentiment of each sentence
sentiments <- sentiment(reviews)

# The sentimentr output, is a data.frame with the text, the score, the element id, which is the review, 
# the sentence id, which is the sentence of the review, a word count and a sentiment value. I'll remove the 
# columns that aren't necessary for the analysis.

# Also, since quanteda and sentimentr count the number of sentences differently, I'll filter the number of 
# sentences again, to keep on 10 sentences per review.

# Remove reviews with more than 10 sentences

#Select the elements ids with 10 or less sentences.
elements_10 <- sentiments %>% 
  mutate(count=1) %>% 
  group_by(element_id) %>% 
  summarize(S=sum(count)) %>% 
  filter(S<=10) %>% 
  pull(element_id)

sentiments <- sentiments %>% 
  filter(element_id %in% elements_10) %>% 
  select(-word_count, -text) # These two columns are not necessary any longer.

# Now, we turn this sentiments dataset into a sparse matrix to consider each sentence sentiment as a variable 
# for that review. 

sparse_sentiments <- sentiments %>% 
  spread(sentence_id, sentiment, fill=0)

set.seed(221)
index <- createDataPartition(y=sparse_sentiments$score, times=1, p=0.20, list=FALSE)
train_set <- sparse_sentiments[-index, ]
test_set <- sparse_sentiments[index, ]

# I'll fit a SVM model with radial kernel
# I don't recommend to run this. I left it overnight so I have no idea of how long it took.
fit <- svm(score ~., train_set, kernel='radial')
y_hat <- predict(fit, test_set)
mean(y_hat==test_set$score)

# Naive bayes algorithm can also be used. It provides a small different accuracy than SVM. However, it runs much, much faster.
#fit_bayes <- naiveBayes(score ~., train_set)
#y_bayes <- predict(fit_bayes, test_set)




