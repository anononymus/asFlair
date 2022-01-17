from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

# are you training a forward or backward LM?
is_forward_lm = True

# load the default character dictionary
#dictionary: Dictionary = Dictionary.load('chars')
import pickle
dictionary = Dictionary.load_from_file('dict')

# get your corpus, process forward and at the character level
corpus = TextCorpus('corpus',
                    dictionary,
                    is_forward_lm
                    #is_backward_lm
                    #character_level
			)

# instantiate your language model, set hidden size and number of layers
language_model = LanguageModel(dictionary,
                               is_forward_lm,
                               hidden_size=1024,
                               nlayers=6)

# train your language model
trainer = LanguageModelTrainer(language_model, corpus)

trainer.train('language_model_new',
              sequence_length=128,
              mini_batch_size=32,
	      learning_rate=20,
              patience=10,
             # checkpoint=True,
              max_epochs=50)
