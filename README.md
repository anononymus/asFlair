# AsFlair
[AsFlair](https://drive.google.com/file/d/1oZG5HoWnbPXM2a0_NLy4YRqAyvqZFeHP/view?usp=drive_link): Contextual Word Embedding for Assamese


## Sequence model trained using AsFlair embedding

This repository contains a pre-trained model for Assamese POS tagging based on BiLSTM-CRF architecture and AsFlair Embedding.
## How to run

Download the pre-trained model from the [link] (https://drive.google.com/file/d/1MC7mVOguotsPEnpLiL20ag97O7siMqvU/view?usp=sharing)

```
from flair.models import SequenceTagger
from flair.data import  Sentence, Token

# Load the tagger

model = SequenceTagger.load('AsFlair.pt')

#  create example sentence
sen='ভাৰতীয় পেচ বলাৰ জৱাগল শ্রীনাথে আক্রমণ কৰিবলৈ আৰম্ভ কৰি প্রথম বলটোতেই শ্রীলংকাপেনাৰ ৰমেশ কালুৱিথার্ণাক পেভিলিয়নলৈ পঠিয়াইছিল ৷'
sentence = Sentence(sen)

# predict tags and print
model.predict(sentence)
print(sentence.to_tagged_string())

```

<!-- Assamese monolingual corpus (23 Million) - [Download] (https://drive.google.com/file/d/1sA5qEwD-Fadh3dIB1ADcK3m1yfWb8i_l/view?usp=sharing) -->
