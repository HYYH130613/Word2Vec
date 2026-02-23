import numpy as np
import random
import re
import os


# path = "D:\Test_Projects\Word2Vec"


# def preparing_data(filename, path):

#     temp_filename = filename + ".tmp"
#     pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    
#     with open(f"{path}/{filename}", "r", encoding="utf-8") as read_file, \
#      open(f"{path}/{temp_filename}", "w", encoding="utf-8") as write_file:

#         for line in read_file:
#             modified_line = pattern.findall(line.lower())
#             for i in modified_line:
#                 write_file.write(i)
#                 print(i)

#     os.replace(temp_filename, filename)
#     return "Done!"

# path = "D:/Test_Projects/Word2Vec"
# filename = "datasetest.txt"
# preparing_data(filename, path)           
            

class Newname:
    """
    Class for reading and loading Stanford Sentiment Treebank. We ignore the sentiment component of the treebank and extract just the text.
    """

    def __init__(self, path=None, table_size=1000000):
        if not path:
            path = "D:/Test_Projects/Word2Vec"

        self.path = path
        self.table_size = table_size

        self.get_sentences()
        self.get_tokens()
        self.get_all_sentences()
        self.dataset_split()
        self.sampleTable()
        

    def get_tokens(self):
        if hasattr(self, "tokens") and self.tokens:
            return self.tokens

        tokens = {}
        tok_freq = {}
        word_count = 0
        rev_tokens = []
        idx = 0

        for sent in self.sentences:
            for w in sent:
                word_count += 1
                if w not in tokens:
                    tokens[w] = idx
                    rev_tokens += [w]
                    idx += 1
                    tok_freq[w] = 1
                else:
                    tok_freq[w] += 1

        tokens["UNK"] = idx
        rev_tokens += ["UNK"]
        tok_freq["UNK"] = 1
        word_count += 1

        self.tokens = tokens
        self.tok_freq = tok_freq
        self.rev_tokens = rev_tokens
        self.word_count = word_count
        return self.tokens

    def get_sentences(self):
        if hasattr(self, "sentences") and self.sentences:
            return self.sentences

        sentences = []
        with open(f"{self.path}/datasetest.txt", "r") as f:
            for line in f:
                split = line.strip().split()[0:]
                sentences += [[w.lower() for w in split]]
        sent_lens = np.array([len(s) for s in sentences])
        cum_sent_lens = np.cumsum(sent_lens)

        self.sentences = sentences
        self.sent_lens = sent_lens
        self.cum_sent_lens = cum_sent_lens
        return sentences

    def get_reject_prob(self):
        if hasattr(self, "reject_prob") and self.reject_prob:
            return self.reject_prob

        threshold = 1e-8 * self.word_count
        reject_prob = np.zeros((len(self.tokens),))
        n_tokens = len(self.tokens)
        for i in range(n_tokens):
            w = self.rev_tokens[i]
            freq = 1.0 * self.tok_freq[w]
            reject_prob[i] = max(0, 1 - np.sqrt(threshold / freq))
        self.reject_prob = reject_prob
        return reject_prob

    def get_all_sentences(self):
        if hasattr(self, "all_sentences") and self.all_sentences:
            return self.all_sentences

        sentences = self.get_sentences()
        reject_prob = self.get_reject_prob()
        tokens = self.get_tokens()
        all_sentences = [
            [
                w
                for w in s
                if 0 >= reject_prob[tokens[w]]
                or random.random() >= reject_prob[tokens[w]]
            ]
            for s in sentences * 30
        ]
        all_sentences = [s for s in all_sentences if len(s) > 1]
        self.all_sentences = all_sentences
        return all_sentences

    def get_random_context(self, C=5):
        sentences = self.get_all_sentences()
        sent_id = random.randint(0, len(sentences) - 1)
        sent = sentences[sent_id]
        word_id = random.randint(0, len(sent) - 1)

        context = sent[max(0, word_id - C) : word_id]
        if word_id + 1 < len(sent):
            context += sent[word_id + 1 : min(len(sent), word_id + C + 1)]

        center = sent[word_id]
        context = [w for w in context if w != center]

        if len(context) > 0:
            return center, context
        else:
            return self.get_random_context

    def dataset_split(self):
        if hasattr(self, "split") and self.split:
            return self.split

        split = [[] for _ in range(3)]
        with open(f"{self.path}/datasetSplit.txt", "r") as f:
            for line in f:
                split = line.strip().split(",")
                split[int(split[1]) - 1] += [int(split[0]) - 1]
        self.split = split
        return split

    def sampleTable(self):
        # if hasattr(self, "sample_table") and self.sample_table:
        #     return self.sample_table

        tokens_num = len(self.tokens)
        sampling_freq = np.zeros((tokens_num,))

        i = 0
        for w in range(tokens_num):
            w = self.rev_tokens[i]
            if w in self.tok_freq:
                freq = 1.0 * self.tok_freq[w]
                freq = freq**0.75
            else:
                freq = 0.0
            sampling_freq[i] = freq
            i += 1

        sampling_freq /= np.sum(sampling_freq)
        sampling_freq = np.cumsum(sampling_freq) * self.table_size

        self.sample_table = np.zeros((int(self.table_size),))

        j = 0
        for i in range(int(self.table_size)):
            while i > sampling_freq[j]:
                j += 1
            self.sample_table[i] = j

        return self.sample_table

    def get_random_train_sentence(self):
        split = self.dataset_split()
        sent_id = random.choice(split[0])
        return self.all_sentences[sent_id]

    def get_split_sentences(self, split=0):
        split = self.dataset_split()
        sentences = [self.all_sentences[i] for i in split[split]]
        return sentences

    def get_train_sentences(self):
        return self.get_split_sentences(0)

    def get_test_sentences(self):
        return self.get_split_sentences(1)

    def get_val_sentences(self):
        return self.get_split_sentences(2)

    def sampleTokenIdx(self):
        return self.sample_table[random.randint(0, self.table_size - 1)]
    
dataset = Newname()  # takes about 45sec
tokens = dataset.tokens
num_words = len(tokens)