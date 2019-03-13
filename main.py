
"""
Yahia El Bsat
IEMS 308 - HW4
Q&A System

"""

import os
import io
import re
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize,RegexpTokenizer
from nltk.corpus import stopwords
import string
import pandas as pd
import numpy as np
import random
import enchant
import datetime
from geotext import GeoText
from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Search, query
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

ppath = '/Users/Yahia/Desktop/IEMS308/HW4/'

os.chdir(ppath)

### Converting day corpus to corpus of articles and reading into nltk

pre_dir = ppath + "articles_pre/" #directory with all 2013 and 2014 days as downloaded (needs cleaning)
days_dir = ppath + "days/" #empty directory to fill in the cleaned articles grouped by days
corpus_dir = ppath + "articles/" #empty directory that will include each article in a day
rfcomp_path = ppath + 'rf_comp.joblib'
rfceo_path = ppath + 'rf_ceo.joblib'
features_path = ppath + 'features.csv'

def get_documents(path):
    for name in os.listdir(path):
        if name.endswith('.txt'):
            yield os.path.join(path, name)
            

printable = set(string.printable)

def clean_corpus(corpus=days_dir):
    for path in get_documents(pre_dir):
        with io.open(path, 'r',errors='ignore') as f:
            text = f.read()
            text = filter(lambda x: x in printable, text)
        fname = os.path.splitext(os.path.basename(path))[0] + ".txt"
        outpath = os.path.join(corpus, fname)
        with io.open(outpath, 'w',encoding='utf8',errors='ignore') as f:
            f.write("".join(text))


def split_corpus(corpus=corpus_dir,days=days_dir):
    for path in get_documents(days):
        with io.open(path, 'r',errors='ignore') as f:
            text = f.read()
            art = text.split("\n")
        date = os.path.splitext(os.path.basename(path))[0]
        for i in range(len(art)):
            if (len(art[i]) == 0): continue
            ind = i + 1
            fname = date + '-' + str(ind) + '.txt'
            outpath = os.path.join(corpus, fname)
            with io.open(outpath, 'w',encoding='utf8',errors='ignore') as f:
                f.write("".join(art[i]))

clean_corpus()
split_corpus()

# Importing the Corpus
corpus = nltk.corpus.PlaintextCorpusReader(corpus_dir, '.*\.txt')
len(corpus.fileids()) # we have 35231 articles


### Setting up Elastic Search
es = Elasticsearch()

index_name = "article"
if not es.indices.exists(index = index_name):
    es.indices.create(index = index_name)
 
        
def es_insert(fileid, sentences):    
    #fileid of the form 'yyyy-mm-dd-i.txt'
    y = int(fileid[:4])
    m = int(fileid[5:7])
    d = int(fileid[8:10])
    _id = f"{fileid[:fileid.find('.')]}"
    _date = datetime.date(y, m, d)
    _body = sentences
    _doc = {"body": _body, "date": _date, "month": _date.month, "year": _date.year}
    return {"_index": index_name, "_type": "article", "_id": _id,"_source": _doc}   
    

cmds = [ ]
for fname in corpus.fileids():
    text = corpus.raw(fname)
    cmd = es_insert(fname,text)
    cmds.append(cmd)

helpers.bulk(es, cmds) #inserts the articles

#############################################
############ Question analysis ##############
#############################################
         
def classify_question(quest : str):
    """
    Types:
    1: Which companies went bankrupt in month x of year y?
    2: Who is the CEO of company X?
    3: What affects GDP?
    4: What percentage increase or decrease is associated with 
       this {property}?
    5: other   
    """
    quest = quest.lower()
    # Type 1:
    if any(w in quest for w in ['bankrupt','bankruptcy','bankrupted']):
        return 1
    # Type 2: includes CEO or chief or executive
    if any(w in quest for w in ['ceo','executive','chief','leader','leading','lead','leads']):
        return 2
    # Type 3: match the question
    if ("affect" in quest or "affects" in quest) and "gdp" in quest:
        return 3
    # Type 4: increase or decrease
    if any(w in quest for w in ['increase','decrease','associated']) and "percentage" in quest:
        return 4
    # Type 5: all else
    return 5


#############################################
############# Query Formation ###############
#############################################

stop = set(stopwords.words("english"))

def rm_sw(text : str):    
    new = [ ]
    for word in text.split():
        if word.lower() not in stop:
            new.append(word)
    return " ".join(new)

def rm_notalphanum(text : str):    
    text = re.sub(r"[^A-Za-z0-9 ]", " ", text)
    return text

def pos_tag_sent(text : str):
    tokenizer = RegexpTokenizer("\w+") 
    words = tokenizer.tokenize(text)
    pos = nltk.pos_tag(words,tagset='universal')
    return pos

def combine_with_AND(text : str):
    return " AND ".join(text.split())

months = ["january","february","march","april","may","june","july","august",
          "september","october","november","december"]


def query_bankrupt(quest : str):
    quest = rm_notalphanum(quest)    
    # Find the month in the query
    r = re.compile('|'.join([r'\b%s\b' % m for m in months]), flags=re.I)
    month = r.findall(quest.lower())
    if len(month) == 0:
        return [-1,[]]
    month_num = months.index(month[0])    
    # Find the year in the query
    r = re.compile("\d{4}")
    year = r.findall(quest)
    if len(year) == 0:
        return [-1,[]]
    year = int(year[0])
    # Queries
    q1 = query.Q("query_string", query = "bankrupt bankruptcy bankrupted")
    #q2 = query.Q("match", month = month_num)
    #q3 = query.Q("match", year = year)
    q4 = query.Q("query_string", query = str(year))
    q5 = query.Q("query_string", query = str(month[0]))
    quer = q1 + q4 + q5
    return [quer,[str(year).lower(),str(month[0]).lower()]]

def query_ceo(quest : str):    
    quest = rm_sw(quest)
    quest = rm_notalphanum(quest)
    l = ['ceo','executive','chief','leader','leading','lead','leads']
    s = quest.lower().split(' ')
    s = set(s) - set(l)
    s = ' '.join(s)
    quer = query.Q("query_string", query = s) + query.Q("query_string",query = 'CEO executive chief leader lead leads leading')
    return [quer, s.split(' ')]


def query_gdp(quest : str):        
    q1 = "GDP"
    q2 = "affect affects effect effects increase decrease rise drop growth drag"  
    quer = query.Q("query_string", query = q1) + query.Q("query_string", query = q2)
    return [quer,[]]


def query_percent(quest : str):
    quest = rm_notalphanum(quest)
    pos = pos_tag_sent(quest)    
    # Find the words after the last preposition
    terms = [ ]
    for w, t in pos[::-1]:
        if t != "ADP":
            terms.append(w)
        else:
            break
    # Form query to match GDP and terms in quest
    toks = terms
    terms.append("GDP")
    quest = " ".join(terms)
    quest = combine_with_AND(quest)
    quer = (query.Q("query_string", query = quest) + 
            query.Q("query_string", query = "affect affects effect effects increase decrease rise drop growth drag % percent"))
    return [quer,toks]

def query_empty(quest: str):
    return [-1,[]]

def get_query(q : str):    
    qtype = classify_question(q)
    mapping = {
        1 : query_bankrupt,
        2 : query_ceo,
        3 : query_gdp,
        4 : query_percent,
        5 : query_empty
    }
    query_fun = mapping[qtype]
    [q,tok] = query_fun(q)
    if tok.count('') > 0:
        tok.remove('')
    return [q,tok]


def search_articles(es_query):    
    s = Search(using = es, index = index_name)
    s = s.query(es_query)
    s = s[:50]
    result = s.execute()
    return result


#############################################
############ NER Classifier #################
#############################################
rf_ceo = load(rfceo_path)
rf_comp = load(rfcomp_path)
comp_keywords = {"Co", "Corp", "Corporation", "Company", "Group", "Inc",
            "Ltd", "Capital", "Financial", "Management"}
ceo_keywords = {"CEO", "chief",  "executive", "officer", "Chief" ,"Executive","Officer"}
dico = enchant.Dict("en_US")
exp2 = "([A-Z][\w-]*(\s+[A-Z][\w-]*)+)"
pattern2 = re.compile(exp2)
tokenizer = RegexpTokenizer("\w+") 
features = pd.read_csv(features_path)
pos_before_feat = features.iloc[10:47,:]
pos_before_feat['pos'] = pos_before_feat.cols.apply(lambda col: col[11:])
before_feat = pos_before_feat.pos.values.tolist()
pos_after_feat = features[47:]
pos_after_feat['pos'] = pos_after_feat.cols.apply(lambda col: col[10:])
after_feat = pos_after_feat.pos.values.tolist()

def first_occur(sub,lst):
    if len(sub) == 1:
        return lst.index(sub[0])
    return lst.index(sub[1]) - 1

def process_sentence(sent,word_l):
    word = ' '.join(word_l)
    start = first_occur(word_l,sent)
    leng = len(word_l)
    pos = nltk.pos_tag(sent,tagset = 'universal')
    pos_before = pos[start-1][1] if start != 0 else ''
    pos_after = pos[start+leng][1] if start + leng < len(sent) else ''    
    is_ceo_keyword = 1 if len(set(sent).intersection(ceo_keywords)) > 0 else 0 
    is_comp_keyword = 1 if len(set(sent).intersection(comp_keywords)) > 0 else 0
    is_english = all(dico.check(n) for n in word_l)
    is_english = 1 if is_english else 0
    places = GeoText(word)
    is_location = (len(places.cities) + len(places.countries)) != 0
    is_location = 1 if is_location else 0
    num_capital = sum(1 for c in word if c.isupper())
    beg_sent = 1 if start == 0 else 0
    end_sent = 1 if start + leng == len(sent) else 0
    char_len = len(word)
    r =  {'sent': sent, 'word':word, 'pos_before':pos_before, 'pos_after':pos_after, 'body':{
            'start':[start], 'len':[leng], 'is_ceo_keyword':[is_ceo_keyword],
            'is_comp_keyword':[is_comp_keyword],'is_english': [is_english], 'is_location': [is_location], 'num_capital': [num_capital],
            'beg_sent': [beg_sent], 'end_sent': [end_sent], 'char_len':[char_len]}}
    return r

def find_names(sents,typ):
    names = []
    for sent in sents:
        pats = re.findall(pattern2, " ".join(sent))
        for p in pats:
            tok = tokenizer.tokenize(p[0]) 
            tok = [x for x in tok if x not in list(ceo_keywords)]
            if len(tok) == 0:
                continue
            r = process_sentence(sent,tok)
            #build pos_before featires
            vec_before = pd.DataFrame(columns = pos_before_feat['cols'])
            vec_before.loc[0] = [0 for n in range(len(pos_before_feat['pos']))]
            if before_feat.count(r.get('pos_before')) > 0:
                ind = before_feat.index(r.get('pos_before'))
            else:
                ind = -1    
            if ind > -1:
                vec_before.iloc[0,ind] = 1
            
            #build pos_after features
            vec_after = pd.DataFrame(columns = pos_after_feat['cols'])
            vec_after.loc[0] = [0 for n in range(len(pos_after_feat['pos']))]
            if after_feat.count(r.get('pos_after')) > 0:
                ind = after_feat.index(r.get('pos_after'))
            else:
                ind = -1    
            if ind > -1:
                vec_after.iloc[0,ind] = 1                   
            
            #build full vector and predicting
            vec_body = pd.DataFrame.from_dict(r.get('body'))
            vec = pd.concat([vec_body,vec_before,vec_after],axis = 1)
            is_name = rf_comp.predict(vec)[0]
            is_name2 = rf_ceo.predict(vec)[0]
            if is_name or is_name2:
                names.append(r.get('word'))    
    return names


def find_company(sents):
    companies = find_names(sents,'comp')
    if len(companies) == 0:
        return 'Could not find any company'
    else:
        return max(set(companies), key=companies.count)    


def find_ceo(sents):
    ceos = find_names(sents,'ceo')
    if len(ceos) == 0:
        return 'Could not find any CEO'
    else:
        return max(set(ceos), key=ceos.count)


#############################################
############ Answer Analysis ################
#############################################

bankrupt_set = set(["bankrupt", "bankruptcy", "bankrupted", "chapter"])

def get_bankruptcy_answer(results,quest_tokens):   
    # Get all sentences that have bankrupcy terms
    sentences = [ ] 
    for hit in results.hits:
        _id = hit.meta.id
        for sent in corpus.sents(_id + '.txt'):
            sent2 = [item.lower() for item in sent]
            if len(set(sent2).intersection(bankrupt_set)) > 0: #sentence includes bankruptcy term
                  if len(set(sent2).intersection(set(quest_tokens))) > 0: #must include the year and month
                      sentences.append(sent)
    company = find_company(sentences)
    return company

ceo_set = set(['ceo','executive','chief','leader','leading','lead','leads'])


def get_ceo_answer(results,quest_tokens):
    sentences = [ ]
    for hit in results.hits:
        _id = hit.meta.id
        for sent in corpus.sents(_id + '.txt'):
            sent2 = [item.lower() for item in sent]
            if len(set(sent2).intersection(ceo_set)) > 0: #Sentence includes a CEO term
                  if len(set(sent2).intersection(set(quest_tokens))) > 0: #Sentence must include the company name
                      sentences.append(sent)
    ceo = find_ceo(sentences)
    return ceo



def get_gdp_factors(results,quest_tokens,run = False):    
    #to save on compute during question, as the answer will always be the same given this corpus
    if not run:
        factors = [
        "interest rates", "credit", "investments", "inflation",
        "government", "growth", "debt",
        "labor", "markets", "economy"
        ]
        return ", ".join(factors)    
    else:
        # Get doc contents from search results
        bodies = [hit.body for hit in results.hits]
        bodies = map(rm_notalphanum, bodies)
        bodies = map(rm_sw, bodies)
        # Compute tf-idf scores
        tf = TfidfVectorizer(ngram_range = (1, 2))
        tfidf = tf.fit_transform(bodies)
        # Get top terms throughout docs
        term_freqs = np.sum(tfidf, axis = 0)
        inds = np.argsort(term_freqs)[:, -50:]
        words = np.array(tf.get_feature_names())[inds]
        return ", ".join(words)
    

change_set = set(['affect', 'affects', 'effect', 'effects', 'increase', 'decrease', 'rise', 'drop','growth','drag','%','percent','perc.'])
gdp_set = set(['gdp','GDP'])

def get_percent_answer(results,quest_tokens):    
    sentences = [ ]
    for hit in results.hits:
        _id = hit.meta.id
        for sent in corpus.sents(_id + '.txt'):
            sent2 = [item.lower() for item in sent]
            if len(set(sent2).intersection(change_set)) > 0: #Sentence includes a gdp change term
                if len(set(sent2).intersection(set(quest_tokens))) > 0: #Sentence must include the key effect of the question
                    if len(set(sent2).intersection(gdp_set)) > 0: #Sentence should include the GDP word
                        sentences.append(sent)
    percents = [ ]
    pattern1 = re.compile("\d+(?:\.\d+)?(?:%| percent(?:age points)?)")
    digits = "(?:one|two|three|four|five|six|seven|eight|nine)"
    teens = "(?:eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)"
    tens = "(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    exp = f"(?:{digits}|{teens}|{tens}|(?:{tens}-{digits})) percent(?:age points)?"
    pattern2 = re.compile(exp)
    
    for sent in sentences:
        matches = re.findall(pattern1, " ".join(sent))
        percents.extend(matches)
        matches = re.findall(pattern2, " ".join(sent))
        percents.extend(matches)
    percents = [x for x in percents if '0' not in x]
    
    if len(percents) == 0:
        return 'Sorry could not find an answer.'
    else:
        return max(set(percents),key=percents.count)

def get_no_answer(results,quest_tokens):
    return 'Sorry could not find an answer.'


def results_to_answer(results, qtype,quest_tokens):    
    mapping = {
        1 : get_bankruptcy_answer,
        2 : get_ceo_answer,
        3 : get_gdp_factors,
        4 : get_percent_answer,
        5 : get_no_answer
    }
    answerer = mapping[qtype]
    return answerer(results,quest_tokens)

#############################################
############### Pipeline ####################
#############################################
def answer_question(q : str):
    q_type = classify_question(q)
    if q_type == 5:
        return 'Cannot answer this question form.'
    [es_query,toks] = get_query(q)
    if es_query == -1:
        return 'Cannot answer this question, information missing'
    results = search_articles(es_query)
    answer = results_to_answer(results, q_type,toks)
    return answer 


