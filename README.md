# AsFlair
AsFlair: A Parts of Speech tagger for Assamse using BiLSTM-CRF architecture. The model achieves F1 score of 73.26% in Assamese POS tagging.

This repository contains pre-trained model for Assamese POS tagging based on BiLSTM-CRF architecture
## How to run

Download the pre-trained model from the link- https://drive.google.com/file/d/1MC7mVOguotsPEnpLiL20ag97O7siMqvU/view?usp=sharing

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
