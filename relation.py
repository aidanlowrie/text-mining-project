import pandas as pd
import spacy 
from spacy.matcher import Matcher 
from tqdm import tqdm 
nlp = spacy.load('en_core_web_sm')

import matplotlib.pyplot as plt
from tqdm import tqdm
nlp = spacy.load('en_core_web_sm')

# This class provides data frames to retrieve
# information related relations between entities
# This file will be edited soon
class RelationExtractor:
    def __init__(self, sentences):
        self.sentences = sentences
        entity_pairs = []

        for i in tqdm(sentences):
            entity_pairs.append(self.extract_entities(i))
        self.entity_pairs = entity_pairs
        # maybe tqdm
        relations = [self.extract_relation(i) for i in sentences]
        self.relations = relations
        # extract subject
        source = [i[0] for i in entity_pairs]
        self.source = source
        # extract object
        target = [i[1] for i in entity_pairs]
        self.target = target

        df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
        self.df = df

        self.relations = self.get_relations(self.df)


# ENTITY PATTERN MIGHT CHANGE
# LEARN ABOUT MATCHER
# CHANGE INSTANTIATION DETAILS
# ADD GETTER METHODS
    
    def extract_relation(self, sent):

        doc = nlp(sent)

  # Matcher class object 
        matcher = Matcher(nlp.vocab)

  #define the pattern 
        pattern = [[{'DEP':'ROOT'}], 
            [{'DEP':'prep'}, {'OP':"?"}],
            [{'DEP':'agent'}, {'OP':"?"}],  
            [{'POS':'ADJ'}, {'OP':"?"}]] 

        matcher.add("matching_1", pattern) 

        matches = matcher(doc)
        k = len(matches) - 1

        span = doc[matches[k][1]:matches[k][2]] 

        return(span.text)
    
    def extract_entities(self, sent):
    
  
        ent1 = ""
        ent2 = ""

        prv_tok_dep = ""    # dependency tag of previous token in the sentence
        prv_tok_text = ""   # previous token in the sentence

        prefix = ""
        modifier = ""

  
        for tok in nlp(sent):

    # if token is a punctuation mark then move on to the next token
            if tok.dep_ != "punct":
      # check: token is a compound word or not
                if tok.dep_ == "compound":
                    prefix = tok.text
        # if the previous word was also a 'compound' then add the current word to it
                    if prv_tok_dep == "compound":
                        prefix = prv_tok_text + " "+ tok.text
      

    # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
        # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " "+ tok.text
      
            if tok.dep_.find("subj") == True:
                ent1 = modifier +" "+ prefix + " "+ tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""      
          
            if tok.dep_.find("obj") == True:
                ent2 = modifier +" "+ prefix +" "+ tok.text
        
      # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
    
        return [ent1.strip(), ent2.strip()]
    
    def get_relations(self, df):

        relations = {}
        for index, row in df.iterrows():
            phrase = row['edge']
            if phrase not in relations:
                relations[phrase] = 0
            relations[phrase] += 1
        return sorted(relations.items(), key=lambda x:x[1], reverse = True)        



    # get relations
    # get
    