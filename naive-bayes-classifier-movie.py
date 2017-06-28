# Open files and put reviews into lists read by paragraph
f = open('rt-polarity.neg', 'r')
data = f.read()
neg_reviews = data.split("\n")
neg_reviews[:] = (value for value in neg_reviews if value != '\t')

f = open('rt-polarity.pos', 'r')
data = f.read()
pos_reviews = data.split("\n")
pos_reviews[:] = (value for value in pos_reviews if value != '\t')

# Split the dataset into train/dev/test
pos_trainSize = int(len(pos_reviews) * 0.7)
pos_trainSet = []
pos_devSize = int(len(pos_reviews) * 0.15)
pos_devSet = []
pos_testSize = int(len(pos_reviews) * 0.15)
pos_testSet = []

neg_trainSize = int(len(neg_reviews) * 0.7)
neg_trainSet = []
neg_devSize = int(len(neg_reviews) * 0.15)
neg_devSet = []
neg_testSize = int(len(neg_reviews) * 0.15)
neg_testSet = []

import random
while len(pos_trainSet) < pos_trainSize:
  index = random.randrange(len(pos_reviews))
  pos_trainSet.append(pos_reviews.pop(index))
while len(pos_devSet) < pos_devSize:
  index = random.randrange(len(pos_reviews))
  pos_devSet.append(pos_reviews.pop(index))
while len(pos_testSet) < pos_testSize:
  index = random.randrange(len(pos_reviews))
  pos_testSet.append(pos_reviews.pop(index))

while len(neg_trainSet) < neg_trainSize:
  index = random.randrange(len(neg_reviews))
  neg_trainSet.append(neg_reviews.pop(index))
while len(neg_devSet) < neg_devSize:
  index = random.randrange(len(neg_reviews))
  neg_devSet.append(neg_reviews.pop(index))
while len(neg_testSet) < neg_testSize:
  index = random.randrange(len(neg_reviews))
  neg_testSet.append(neg_reviews.pop(index))
  

# Function to convert a raw string to a string consisting of only meaningful words
# The input is a single string and 
# the output is a single string with only lower-cased words
# without punctuations and articles
import re
from nltk.corpus import stopwords
def convert_to_meaningful_words(raw_string):
    # 1. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", str(raw_string)) 
    #
    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 3. Convert the imported stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 4. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 5. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))

# Convert all reviews into strings of meaningful words
pos_train_clean = []
pos_dev_clean = []
pos_test_clean = []
neg_train_clean = []
neg_dev_clean = []
neg_test_clean = []

for i in xrange( 0, pos_trainSize ):
  string_to_append = ""
  string_to_append += convert_to_meaningful_words(pos_trainSet[i])
  pos_train_clean.append( 
    string_to_append
  )
for i in xrange( 0, pos_devSize ):
  string_to_append = ""
  string_to_append += convert_to_meaningful_words(pos_devSet[i])
  pos_dev_clean.append( 
    string_to_append
  )
for i in xrange( 0, pos_testSize ):
  string_to_append = ""
  string_to_append += convert_to_meaningful_words(pos_testSet[i])
  pos_test_clean.append( 
    string_to_append
  )
for i in xrange( 0, neg_trainSize ):
  string_to_append = ""
  string_to_append += convert_to_meaningful_words(neg_trainSet[i])
  neg_train_clean.append( 
    string_to_append
  )
for i in xrange( 0, neg_devSize ):
  string_to_append = ""
  string_to_append += convert_to_meaningful_words(neg_devSet[i])
  neg_dev_clean.append( 
    string_to_append
  )
for i in xrange( 0, neg_testSize ):
  string_to_append = ""
  string_to_append += convert_to_meaningful_words(neg_testSet[i])
  neg_test_clean.append( 
    string_to_append
  )

#Convert the reviews into vectors
from sklearn.feature_extraction.text import CountVectorizer

# # Initialize the "CountVectorizer" object, which is scikit-learn's
# # bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 50) 

pos_train_vect = vectorizer.fit_transform(pos_train_clean)
pos_train_vect = pos_train_vect.toarray()
pos_dev_vect = vectorizer.fit_transform(pos_dev_clean)
pos_dev_vect = pos_dev_vect.toarray()
pos_test_vect = vectorizer.fit_transform(pos_test_clean)
pos_test_vect = pos_test_vect.toarray()

neg_train_vect = vectorizer.fit_transform(neg_train_clean)
neg_train_vect = neg_train_vect.toarray()
neg_dev_vect = vectorizer.fit_transform(neg_dev_clean)
neg_dev_vect = neg_dev_vect.toarray()
neg_test_vect = vectorizer.fit_transform(neg_test_clean)
neg_test_vect = neg_test_vect.toarray()

import math
# Calculate the mean of each dimension
def mean(numbers):
  return sum(numbers)/float(len(numbers))

# Calculate the standard deviation of each dimension
def stdev(numbers):
  avg = mean(numbers)
  variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
  return math.sqrt(variance)

# Wrap the mean and standard deviation into one entry and do the same for all attributes
def summarize(dataset):
  summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
  return summaries

# Calculate the probability based on one attribute
def calculateProbability(x, mean, stdev):
  exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
  return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

# Calculate the probabilities of one vector entry belonging to each classification
def calculateClassProbabilities(summaries, inputVector):
  probabilities = {}
  for classValue, classSummaries in summaries.iteritems():
    probabilities[classValue] = 1
    for i in range(len(classSummaries)):
      mean, stdev = classSummaries[i]
      x = inputVector[i]
      probabilities[classValue] *= calculateProbability(x, mean, stdev)
  return probabilities

# Predict the classification of the intput vector by comparing the probability of it belonging to each
def predict(summaries, inputVector):
  probabilities = calculateClassProbabilities(summaries, inputVector)
  bestLabel, bestProb = None, -1
  for classValue, probability in probabilities.iteritems():
    if bestLabel is None or probability > bestProb:
      bestProb = probability
      bestLabel = classValue
  return bestLabel

# Do the same prediction from the above for all entries in the dataset
def getPredictions(summaries, testSet):
  predictions = []
  for i in range(len(testSet)):
    result = predict(summaries, testSet[i])
    predictions.append(result)
  return predictions

# Label for reviews: positive=1, negative=0. The x is plugged in for training purposes for prediction. 
x = {1: summarize(pos_train_vect), 0: summarize(neg_train_vect)}

# Get accuracy of the predictions based on the percentage of the right answer
def getAccuracy(predictions, expected_label):
  correct = 0
  for x in range(len(predictions)):
    if predictions[x] == expected_label:
      correct += 1
  return (correct/float(len(predictions))) * 100.0

# print getAccuracy(getPredictions(x, neg_dev_vect), 0)
# print getAccuracy(getPredictions(x, pos_dev_vect), 1)

# print "Divider for accuracy on test set"

print getAccuracy(getPredictions(x, neg_test_vect), 0)
print getAccuracy(getPredictions(x, pos_test_vect), 0)
