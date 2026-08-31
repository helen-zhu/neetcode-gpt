import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    
    @staticmethod
    def get_unique_words(list_of_sentences):
        words = []
        sentences_list = []
        for sentence in list_of_sentences:
            sentence_split = sentence.split(" ")
            words.extend(sentence_split)
            sentences_list.append(sentence_split)
        return set(words), sentences_list

    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)

        positive_words, positive_sentences = self.get_unique_words(positive)
        negative_words, negative_sentences = self.get_unique_words(negative)
        words = positive_words.union(negative_words)
        words = list(words)
        words.sort()
        vocab = {word: i+1 for i, word in enumerate(words)}

        sentence_tensors = []
        for sent in positive_sentences:
            encoded_sentence = [
                vocab[i] for i in sent
            ]
            sentence_tensors.append(torch.tensor(encoded_sentence))
        
        for sent in negative_sentences:
            encoded_sentence = [
                vocab[i] for i in sent
            ]
            sentence_tensors.append(torch.tensor(encoded_sentence))
        
        return torch.nn.utils.rnn.pad_sequence(sentence_tensors, batch_first=True)


