# AsFlair
AsFlair: A Parts of Speech tagger for Assamse using BiLSTM-CRF architecture

This repository contains pre-trained model for Assamese POS tagging based on BiLSTM-CRF architecture using FLair framework
## How to run

Download the pre-trained model from the link- https://drive.google.com/file/d/1ONDEnVgjDkHWSJFSbsaelwKMN9WTIM1J/view?usp=sharing

```
from flair.models import SequenceTagger
from flair.data import  Sentence, Token

# Load the tagger

model = SequenceTagger.load('final-model.pt')

#  create example sentence
sen='ভাৰতীয় পেচ বলাৰ জৱাগল শ্রীনাথে আক্রমণ কৰিবলৈ আৰম্ভ কৰি প্রথম বলটোতেই শ্রীলংকাপেনাৰ ৰমেশ কালুৱিথার্ণাক পেভিলিয়নলৈ পঠিয়াইছিল ৷'
sentence = Sentence(sen)

# predict tags and print
model.predict(sentence)
print(sentence.to_tagged_string())
