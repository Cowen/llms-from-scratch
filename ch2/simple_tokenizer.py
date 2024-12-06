# coding: utf-8
import re
import urllib.request

# Fetch the sample text, The Verdict
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)
with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

class SimpleTokenizerV2:
    # Special tokens to indicate end of a text and unknown tokens
    # Other common ones:
    # - [BOS] Beginning of Sequence: Start of text
    # - [EOS] End of Sequence: End of text, same as this one
    # - [PAD] Padding: When batching smaller texts, this special character is used to fill texts up to BATCHSIZE. EOS can also be used.

    # GPT models do not use any of those tokens though, except for EOS.
    # GPT models also do not encode an unknown token, they do byte pair encoding to break words into subword units
    TOKEN_ENDOFTEXT = "<|endoftext|>"
    TOKEN_UNKNOWN = "<|unk|>"

    def __init__(self, vocab: dict):
        self.token_to_id = vocab
        self.id_to_token = { i:t for t, i in vocab.items() }

    @classmethod
    def from_text(cls, text: str):
        regex = r'([,.:;?_!"()\']|--|\s)'
        tokens_all = [t.strip() for t in  re.split(regex, text) if t.strip()]

        # Uniquify and add special tokens marking end of text and unknown words
        tokens_uniq = sorted(list(set(tokens_all)))
        tokens_uniq.extend([cls.TOKEN_ENDOFTEXT, cls.TOKEN_UNKNOWN])
        vocab = { token:idx for idx, token in enumerate(tokens_uniq) }

        return SimpleTokenizerV2(vocab)

    def encode(self, text: str) -> [int]:
        """Text to token IDs"""
        # Find tokens, remove whitespace
        tokens = re.split(r'([,.?_!"()\']|--|\s)', text)
        tokens = [ t.strip() for t in tokens if t.strip() ]
        # Replace unknown tokens
        tokens = [ t if t in self.token_to_id else self.TOKEN_UNKNOWN
                    for t in tokens ]
        return [ self.token_to_id[t] for t in tokens ]


    def decode(self, token_ids: [int]) -> str:
        """Token IDs to text"""
        text = " ".join([self.id_to_token[i] for i in token_ids])
        # Remove spaces before some punctuation
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text) 


