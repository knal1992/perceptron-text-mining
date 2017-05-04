There are four files in this directory: 
train.positive (positive train instances)
train.negative (negative train instances)
test.positive (positive test instances)
test.negative (negative test instances).

In each file, each line corresponds to the unigram and bigram features representing a single 
Amazon review. Features are separated by a space. Bigram features are denoted by two underscores.
For example, if you look at the first line in train.positive, you will see the
three features "atmosphere they you__unfamiliar". Here, "atmosphere" and "they" are two unigram
features, whereas "you__unfamiliar" is a bigram feature consisting of the two words "you" and "unfamiliar".
Feature vectors are binary. In other words each feature appears exactly once for each Amazon review.