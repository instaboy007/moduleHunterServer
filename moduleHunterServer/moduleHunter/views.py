from django.shortcuts import render
from django.core import serializers
from django.http import HttpResponse, JsonResponse
from moduleHunter.models import ModuleInfo, ModulesInvertedIndex, ModulesDocumentTermFrequency
from rest_framework.decorators import api_view
from rest_framework.response import Response

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import math
import json

# Create your views here.

def getModulesInfo(request):
    # Retrieve all objects from the MyModel table
    all_objects = ModuleInfo.objects.all()
    # Serialize the queryset to JSON
    serialized_data = serializers.serialize('json', all_objects)
    # Return the JSON data as a response
    return JsonResponse(serialized_data, safe=False)

def getModuleInfo(doc_id):
    try:
        module_data = ModuleInfo.objects.filter(doc_id = str(doc_id))
        module_data = module_data[0]
        module_data = {
                'doc_id': module_data.doc_id,
                'name': module_data.name,
                'install': module_data.install,
                'repository': module_data.repository,
                'homepage': module_data.homepage,
                'weekly_downloads': module_data.weekly_downloads,
                'version': module_data.version,
                'license': module_data.license,
                'unpacked_size': module_data.unpacked_size,
                'total_files': module_data.total_files,
                'issues': module_data.issues,
                'pull_requests': module_data.pull_requests,
                'last_publish': module_data.last_publish,
                'url': module_data.url
        }
        return module_data
    except ModuleInfo.DoesNotExist:
        # Handle the case where no object is found
        print(f"No object found for document id '{doc_id}'")
        return None

def getModuleInvertedIndex(queryTerm):
    try:
        # Retrieve the object that matches the query term
        obj = ModulesInvertedIndex.objects.filter(term=queryTerm)
        return obj
    except ModulesInvertedIndex.DoesNotExist:
        # Handle the case where no object is found
        print(f"No object found for term '{queryTerm}'")
        return None
    
def getDocumentTermFrequency():
    try:
        # Retrieve the object that matches the query term
        obj = ModulesDocumentTermFrequency.objects.all()
        return len(obj)
    except ModulesDocumentTermFrequency.DoesNotExist:
        # Handle the case where no object is found
        print(f"No Modules found in the ModulesDocumentTermFrequency Table")
        return None
    
def PreProcessor(documents, key):

    # Tokenization
    tokens = [word_tokenize(document[key]) for document in documents]

    # Converting to Lowercase
    tokens = [[token.lower() for token in document] for document in tokens]

    # Removing Stop Words
    stop_words = set(stopwords.words("english"))
    tokens = [[token for token in document if token not in stop_words] for document in tokens]

    #Lemmatization
    word_net = WordNetLemmatizer()
    tokens =  [[word_net.lemmatize(token) for token in document] for document in tokens]

    # Joining tokens back into documents
    return [" ".join(document) for document in tokens]    

def retrieveTermFrequency(doc_id):
    # Retrieve all objects from the MyModel table
    all_objects = ModulesDocumentTermFrequency.objects.all()

    # Retrieve objects that match a certain condition
    filtered_objects = ModulesDocumentTermFrequency.objects.filter(doc_id=doc_id)

    return filtered_objects

def calculateTf_Idf(docs,terms,query):

    # Compute the document frequency (DF) for each term
    doc_freq = {}
    for term in terms:
        doc_freq[str(term['term_id'])] = len(term['doc_ids'])

    # Compute the IDF score for each term
    idf = {}
    num_docs = getDocumentTermFrequency()
    for term_id in doc_freq:
        df = doc_freq[term_id]
        idf[term_id] = math.log(num_docs / df)

    # Compute the TF-IDF score for each term in each document
    tfidf_scores = {}
    for doc_id in docs:
        tfidf_scores[doc_id] = {}
        for term_id in docs[doc_id]:
            tf = docs[doc_id][term_id]
            idf_score = idf[term_id]
            tfidf_scores[doc_id][term_id] = tf * idf_score

    # Compute the TF-IDF score for each query term
    query_tfidf_scores = {}
    for term in query:
        if term not in query_tfidf_scores:
            query_tfidf_scores[term] = 0.0
        tf = query.count(term)
        idf_score = idf[str(next(t['term_id'] for t in terms if t['term'] == term))]
        query_tfidf_scores[term] = tf * idf_score

    return tfidf_scores,query_tfidf_scores

def cosine_similarity(query_tfidf, doc_tfidf,terms):
    """
    Computes the cosine similarity between a query and a document.
    """
    dot_product = 0
    query_norm = 0
    doc_norm = 0
    
    for term, tfidf in query_tfidf.items():
        term_id = str(next(t['term_id'] for t in terms if t['term'] == term))
        if term_id in doc_tfidf.keys():
            dot_product += tfidf * doc_tfidf[term_id]
        query_norm += tfidf ** 2
    
    for tfidf in doc_tfidf.values():
        doc_norm += tfidf ** 2
    
    if query_norm == 0 or doc_norm == 0:
        return 0
    return dot_product / (math.sqrt(query_norm) * math.sqrt(doc_norm))


@api_view(['GET'])
def modulesSearch(request):
    query = [{'query': request.GET.get('query', '')}]
    query_terms = PreProcessor(query,'query')
    query_terms = query_terms[0].split(' ')
    terms=[]
    for query in query_terms:
        matching_terms = getModuleInvertedIndex(query)
        if matching_terms:
            terms.append({'term_id':matching_terms[0].term_id,'term':matching_terms[0].term,'doc_ids':list(map(int, matching_terms[0].doc_ids.strip('{}').split(',')))})
    documents = {}
    for term in terms:
        for doc_id in term['doc_ids']:
            document_term_frequencies = retrieveTermFrequency(doc_id)
            term_frequencies = {k: v for k, v in document_term_frequencies[0].term_frequency.items() if k in [str(t['term_id']) for t in terms]}
            documents[document_term_frequencies[0].doc_id] = term_frequencies

    tf_idf_scores,query_tf_idf_scores = calculateTf_Idf(documents,terms,[t['term'] for t in terms])
    scores = {}
    for doc_id, doc_terms in tf_idf_scores.items():
        score = cosine_similarity(query_tf_idf_scores, doc_terms,terms)
        scores[doc_id] = score
    ranked_documents = sorted(scores, key=scores.get, reverse=True)
    ranked_documents_data = []
    for ranked_doc_id in ranked_documents:
        module_data = getModuleInfo(ranked_doc_id)
        if module_data:
            ranked_documents_data.append(module_data)
    print(scores)
    ranked_documents_data = json.dumps(ranked_documents_data)
    return JsonResponse(ranked_documents_data, safe=False)
