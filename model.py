#Stochastic Gradient Descent

from tok import StanfordSentiment
import numpy as np
import random

dataset = StanfordSentiment()  # takes about 45sec
tokens = dataset.tokens
num_words = len(tokens)
context = dataset.get_random_context()

vector_dim = 10
word_vecs = np.concatenate(
    (
        (np.random.rand(num_words, vector_dim) - 0.5) / vector_dim,
        np.zeros(
            (num_words, vector_dim)
        ),  # for simplicity's sake, we will have a separate set of vectors for each context word as well as for each center word
    ),
    axis=0,
)

word_vecs.shape  # 2*num_words (one for the context vector and another for the center vector) x vector_dim
# initially random vectors

# getting center word vecs and context word vecs
# each word will have two word vectors: center and context, we will only care about the center word vectors
center_word_vecs = word_vecs[:num_words, :]
outside_word_vecs = word_vecs[num_words:, :]

block_size = 5
center_word, context = dataset.get_random_context(block_size)
center_word, context  # get a center word and context

# find index of center word
center_word_idx = dataset.tokens[center_word]
center_word_idx

# getting the random word vec for this index/word
center_word_vec = center_word_vecs[center_word_idx]
center_word_vec  # still random

# example with just one outside word
outside_word_idx = dataset.tokens[context[0]]
outside_word_idx

outside_word_vec = outside_word_vecs[outside_word_idx]
outside_word_vec  # start as zeros

dot_products = np.dot(
    outside_word_vecs, center_word_vec
)  # take the dot product between all outside words and the center word
dot_products.shape

# let's see what this dot product produces
import matplotlib.pyplot as plt

# plt.plot(dot_products)
# plt.show()  # it's all zeros because all of the outside word vectors are zero

def softmax(x):
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # Vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp

    assert x.shape == orig_shape
    return x

softmax_probs = softmax(dot_products)
softmax_probs.shape

softmax_probs[0]

# let's see what this looks like
plt.hist(softmax_probs)
# plt.show()  # probabilities are even, at zero, as you might expect

loss = -np.log(softmax_probs[outside_word_idx])  # nll in code
print(loss)
# this number represents how good our prediction is
# zero is the lowest number that we can predict

current_grad_center_vec = -outside_word_vecs[outside_word_idx] + np.dot(
    softmax_probs, outside_word_vecs
)  # derivative of dot product for the center word vec
print(current_grad_center_vec)

current_grad_outside_vecs = np.outer(
    softmax_probs, center_word_vec
)  # derivative of dot product for the outer word vecs
current_grad_outside_vecs[outside_word_idx] -= center_word_vec
print(current_grad_outside_vecs)

grad_center_vecs = np.zeros(center_word_vecs.shape)  # holder for our derivative values
grad_outside_vecs = np.zeros(
    outside_word_vecs.shape
)  # holder for our derivative values

grad_center_vecs[center_word_idx] += current_grad_center_vec
grad_outside_vecs += current_grad_outside_vecs

# now that we've calculated our derivatives we can take a step
step = 1
center_word_vecs -= step * grad_center_vecs
outside_word_vecs -= step * grad_outside_vecs

# and then run another forward pass
dot_products = np.dot(outside_word_vecs, center_word_vec)
dot_products  # longer zero!

softmax_probs = softmax(dot_products)


loss = -np.log(softmax_probs[outside_word_idx])


# new example copying from above
vector_dim = 10
word_vecs = np.concatenate(
    (
        (np.random.rand(num_words, vector_dim) - 0.5) / vector_dim,
        np.zeros(
            (num_words, vector_dim)
        ),  # for simplicity's sake, we will have a separate set of vectors for each context word as well as for each center word
    ),
    axis=0,
)

block_size = 5
center_word, context = dataset.get_random_context(block_size)

center_word_idx = dataset.tokens[center_word]
center_word_vec = word_vecs[center_word_idx]

outside_word_idxs = [dataset.tokens[w] for w in context]

center_word_vecs = word_vecs[:num_words, :]
outside_word_vecs = word_vecs[num_words:, :]

# first we need a data structure from which we can easily sample from
# we can use a table of values
table_size = 100


sample_table = dataset.sampleTable()

sample_table[random.randint(0, table_size - 1)]

negSampleWordIndices = [None] * 5
for k in range(5):
    newidx = sample_table[random.randint(0, table_size - 1)]
    print(newidx)
    negSampleWordIndices[k] = newidx
[int(n) for n in negSampleWordIndices]

# function version
# def get_negative_samples(outsideWordIdx, dataset, K):
#     negSampleWordIndices = [None] * K
#     for k in range(K):
#         newidx = dataset.sampleTokenIdx()
#         while newidx == outsideWordIdx:
#             newidx = dataset.sampleTokenIdx()
#         negSampleWordIndices[k] = newidx
#     return [int(n) for n in negSampleWordIndices]

