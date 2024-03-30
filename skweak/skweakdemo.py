import spacy, re
from skweak import heuristics, gazetteers, generative, utils

# LF 1: heuristic to detect occurrences of MONEY entities
def money_detector(doc):
   for tok in doc[1:]:
      if tok.text[0].isdigit() and tok.nbor(-1).is_currency:
          yield tok.i-1, tok.i+1, "MONEY"
lf1 = heuristics.FunctionAnnotator("money", money_detector)

# LF 2: detection of years with a regex
lf2= heuristics.TokenConstraintAnnotator("years", lambda tok: re.match("(19|20)\d{2}$", 
                                                  tok.text), "DATE")

# LF 3: a gazetteer with a few names
NAMES = [("Barack", "Obama"), ("Donald", "Trump"), ("Joe", "Biden")]
trie = gazetteers.Trie(NAMES)
lf3 = gazetteers.GazetteerAnnotator("presidents", {"PERSON":trie})

# We create a corpus (here with a single text)
nlp = spacy.load("en_core_web_sm")
doc = nlp("Donald Trump paid $750 in federal income taxes in 2016")

# apply the labelling functions
doc = lf3(lf2(lf1(doc)))

# create and fit the HMM aggregation model
hmm = generative.HMM("hmm", ["PERSON", "DATE", "MONEY"])
hmm.fit([doc]*10)

# once fitted, we simply apply the model to aggregate all functions
doc = hmm(doc)

# we can then visualise the final result (in Jupyter)
utils.display_entities(doc, "hmm")