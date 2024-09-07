
**Background**: The task of Sanskrit Word Segmentation (SWS) is a challenging problem in natural language processing (NLP), especially because of the unique structure of Sanskrit texts. Traditional texts in Sanskrit often lack spaces between words which makes word segmentation a crucial task for processing these texts effectively. Moreover, Sanskrit also exhibits a linguistic phenomenon called sandhi, where compound words are commonly formed by combining multiple subwords, with specific rules governing changes at word boundaries. This process, known as sandhi, modifies the ending of one word and the beginning of another to form a compound. These sandhi rules vary depending on the characters at the word boundaries.

Our work focuses on developing a model for Sanskrit word segmentation using the ByT5 model, which leverages character-level tokenization to process texts. The primary motivation behind this work is to address the segmentation
of unspaced Sanskrit texts, which is a common format in ancient manuscripts. Although our focus is on unspaced texts, we believe the model could be extended
to handle sandhi with further adjustments and additional training.

Dataset:
The dataset used in our experiments is the Digital Corpus of Sanskrit (DCS), created by Oliver Hellwig (2010-2021). The DCS is a comprehensive, Sandhi- split corpus of Sanskrit texts. Each string in the corpus has been analyzed and verified by a single annotator.
The dataset’s overall size includes 670,479 lines and 5,989,632 words, and exhibits a mean string length of 50.69 characters and a median length of 46 characters, indicating a balanced mix of short and long strings in the corpus.
The DCS dataset is publicly available through its official website and its GitHub repository.
