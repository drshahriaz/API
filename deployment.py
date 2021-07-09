"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import os
from keras.models import load_model
from imageio import imread
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class Deployment:

    
    def __init__(self, base_directory, context):

        print("Initialising deployment")


        weights = os.path.join(base_directory, "epochBIG.h5")
        self.model = load_model(weights)
        self.token_loc = os.path.join(base_directory, "token")

        print("Token = "+str(self.token_loc))

        
        tokenizer = Tokenizer()
        tok = pickle.load(open(self.token_loc, "rb"))
        tokenizer = tok
        
        print("tokenizer = "+str(tokenizer.index_word[1]))

        self.token = tokenizer
        

    

    def request(self,  data):


        print("Processing request")
        
       
        print("Seed_id"+str(data['seed_id']))

        print("Seed_text"+data['seed_text'])
        
        
        
        data['seed_id'] = data['seed_id']
       
        seq_len = 25
        num_gen_words=50

        seed_text = data['seed_text']
                    
        #gen_words = generate_text(self.model,self.token,seq_len,seed_text=seed_text,num_gen_words=50)
         # Final Output
        output_text = []
        
        # Intial Seed Sequence
        input_text = seed_text
        
        # Create num_gen_words
        for i in range(num_gen_words):
            
            # Take the input text string and encode it to a sequence
            encoded_text = self.token.texts_to_sequences([input_text])[0]
            
            # Pad sequences to our trained rate (50 words in the video)
            pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
            
            # Predict Class Probabilities for each word
            pred_word_ind = self.model.predict_classes(pad_encoded, verbose=0)[0]
            
            # Grab word
            pred_word = self.token.index_word[pred_word_ind] 
            
            # Update the sequence of input text (shifting one over with the new word)
            input_text += ' ' + pred_word
            
            output_text.append(pred_word)

        
        


    

        data['auto_generated_paragraph'] =' '.join(output_text) 
        
        return {'auto_generated_paragraph':str(data['auto_generated_paragraph']),'seed_id':int(data['seed_id']) }
