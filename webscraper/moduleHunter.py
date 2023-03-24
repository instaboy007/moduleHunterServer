import requests
import json
from multiprocessing import Pool
from bs4 import BeautifulSoup
import nltk
import string
import math
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import defaultdict
from sqlalchemy import create_engine, text
from IPython.core.getipython import get_ipython
import psycopg2
import os
import socks
import socket
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

def load_packages():
    # Opening JSON file
    package_names_file = open('E:\SEM VIII\Information Retrieval\Package\webscraper\package_names.json')
    
    # returns JSON object as 
    # a dictionary
    data = json.load(package_names_file)

    # # Define the URL of the npm registry API to retrieve all public packages
    # url = 'https://raw.githubusercontent.com/nice-registry/all-the-package-names/master/names.json'
    # headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'}

    # # Send a GET request to the API endpoint
    # response = requests.get(url, headers=headers).content

    # # Get the JSON data from the response
    # data = response.json()

    # Extract the names of all public packages from the JSON data
    return [name for name in data]

def web_scrappe(package):

    # # Set up the socket to use Tor as a proxy
    # socks.set_default_proxy(socks.SOCKS5, "localhost", 9050)
    # socket.socket = socks.socksocket

    print(f'Scraping {package}...')

    try:
        
        # Make a request to the Node.js modules website
        url = f'https://www.npmjs.com/package/{package}'
        response = requests.get(url, headers=headers, verify=False)
        # print(response.status_code, url)
        # print(f'Done Web Scraping {package}')
        # os.system("net stop tor && net start tor")
        return (response, url)
    
    except Exception as e:
        print(f'Error scraping {url}: {e}')
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

def create_inverted_index(documents):
    inverted_index = defaultdict(list)

    for idx, document in enumerate(documents):
        for word in set(re.findall(r'\b\w+\b', document)):
            inverted_index[word].append(idx)
    
    inverted_index = remove_invalid_keys(inverted_index)

    temp = []
    for id, (key,val) in enumerate(inverted_index.items()):
        temp.append((id,key,val))
    inverted_index = temp

    return inverted_index

def create_document_term_frequency(documents, terms):
    document_term_frequency = []  
    for i, document in enumerate(documents):
        term_frequency = {}
        for term in terms:
            if term[1] in document:
                term_frequency[term[0]] = document.count(term[1])
        if term_frequency:
            document_term_frequency.append((i, json.dumps(term_frequency)))
    return document_term_frequency

def compute_tf_idf_index(documents):
    inverted_index = defaultdict(list)
    document_frequency = defaultdict(int)
    test_dict = defaultdict(list)

    # Step 1: Compute the term frequency (TF) of each term in a document.
    for idx, document in enumerate(documents):
        term_frequency = defaultdict(int)
        maxf = 0
        for word in re.findall(r'\b\w+\b', document):
            term_frequency[word] += 1
            maxf = max(maxf, term_frequency[word])

        # Step 2: Compute the inverse document frequency (IDF) of each term in the corpus.
        for word, freq in term_frequency.items():
            document_frequency[word] += 1
            inverted_index[word].append((idx, freq / maxf))

    num_documents = len(documents)

    # Compute IDF values
    idf_values = {term: math.log(num_documents / freq) for term, freq in document_frequency.items()}

    # Step 3: Multiply the TF and IDF values to get the TF-IDF values.
    tf_idf_index = defaultdict(list)
    for term, document_list in inverted_index.items():
        for document, freq in document_list:
            tf_idf = freq * idf_values[term]
            test_dict[document].append({'term': term, 'tf_idf': tf_idf})
            tf_idf_index[term].append((document, tf_idf))
    print(test_dict)
    return tf_idf_index

def remove_invalid_keys(d):
    # Regular expression pattern to match non-ASCII characters
    non_ascii_pattern = re.compile('[^\x00-\x7F]+')
    
    # Loop through each key in the dictionary
    for key in list(d.keys()):
        if isinstance(key, (int, float)):
            # Remove key if it is a number
            del d[key]
        elif non_ascii_pattern.search(key):
            # Remove key if it contains non-ASCII characters
            del d[key]
    
    return d

def createQuery(query):
    try:
        connection = psycopg2.connect(user="postgres",
                                    password="prashanth",
                                    host="127.0.0.1",
                                    port="5432",
                                    database="modules")
        cursor = connection.cursor()
        cursor.execute(query)

        connection.commit()
        print("Creation Successfull...")

    except (Exception, psycopg2.Error) as error:
        print("Creation Failed...", error)

    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
    

def insertAllQuery(query, records, table_name):
    try:
        connection = psycopg2.connect(user="postgres",
                                    password="prashanth",
                                    host="127.0.0.1",
                                    port="5432",
                                    database="modules")
        cursor = connection.cursor()
        cursor.executemany(query,records)

        connection.commit()
        count = cursor.rowcount
        print(count, f"Records inserted successfully into {table_name} table")

    except (Exception, psycopg2.Error) as error:
        print(f"Failed to insert records into {table_name} table", error)

    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

def prepare_modules(responses):
    print("Preparing Modules...")
    if responses is not None:
        for response,url in responses:
            if response and response.content:
                # Parse the HTML content using BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract module data from the search results
                readme = soup.find('div', id="readme")
                name = soup.find('span', class_="_50685029 truncate")
                details = soup.find('div', class_="fdbf4038 w-third-l mt3 w-100 ph3 ph4-m pv3 pv0-l")

                if not readme or not name or not details:
                    continue
                if details:
                    details = details.text
                    transformed_details = {}
                useful_data_list = ['Install', 'Repository', 'Homepage', 'Weekly Downloads', 'Version', 'License', 'Unpacked Size', 'Total Files', 'Issues', 'Pull Requests', 'Last publish']
                for i in range(len(useful_data_list)):
                    data = useful_data_list[i]
                    if i != len(useful_data_list)-1:
                        for useful_data in useful_data_list[i+1:]:
                            if useful_data in details:
                                next_data = useful_data
                                break
                        if data in details:
                            start_index = details.find(data) + len(data)
                            end_index = details.find(next_data)
                            transformed_details[data] = details[start_index:end_index]
                    else:
                        if data in details:
                            start_index = details.find(data) + len(data)
                            transformed_details[data] = details[start_index::]
                if 'Install' in transformed_details:
                    transformed_details['Install'] = transformed_details['Install'].replace('Downloads', '')
                if 'Repository' in transformed_details:
                    transformed_details['Repository'] = transformed_details['Repository'].replace('Git', '')
                if 'Last publish' in transformed_details:
                    transformed_details['Last publish'] = transformed_details['Last publish'].replace('CollaboratorsTry on RunKitReport malware', '')

                module = {
                    'data': readme.text,
                    'name': name.text,
                    'details': transformed_details,
                    'url': url
                }

                modules.append(module)
    else: 
        print('Responses is None')
package_names = load_packages()

headers= {
    "Accept": "text/html,application/xhtml+xml",
    "Connection": "keep-alive",
    "Host": "www.npmjs.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/110.0"
}

modules = []

if __name__ == '__main__':

    # Create a multiprocessing pool with 20 worker processes
    pool = Pool(processes=20)

    responses = pool.map(web_scrappe, package_names[737803:737903])
    print(len(responses))
    
    prepare_modules(responses)

    preprocessed_modules = PreProcessor(modules, 'data')

    engine = create_engine('postgresql://postgres:prashanth@localhost:5432/modules')
    
    # Get the IPython instance
    ipython = get_ipython()

    # Define the SQL query to create the table
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS module_info (
        doc_id INTEGER PRIMARY KEY,
        name VARCHAR(255),
        install VARCHAR(255),
        repository VARCHAR(255),
        homepage VARCHAR(255),
        weekly_downloads INTEGER,
        version VARCHAR(20),
        license VARCHAR(50),
        unpacked_size VARCHAR(20),
        total_files INTEGER,
        issues INTEGER,
        pull_requests INTEGER,
        last_publish VARCHAR(50),
        url VARCHAR(255)
    )
    '''

    createQuery(create_table_query)
    
    # Define the SQL query to insert the data
    insert_modules_query = '''
    INSERT INTO module_info (doc_id, name, install, repository, homepage, weekly_downloads, version, license, unpacked_size, total_files, issues, pull_requests, last_publish, url)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    '''

    # Extract the data from the dictionary
    module_info = [(
        index,
        module['name'], 
        module['details'].get('Install', 'N/A'), 
        module['details'].get('Repository', 'N/A'), 
        module['details'].get('Homepage', 'N/A'), 
        int(module['details'].get('Weekly Downloads', 'N/A').replace(',', '') if module['details'].get('Weekly Downloads', 'N/A') != 'N/A' else '0'), 
        module['details'].get('Version', 'N/A'), 
        module['details'].get('License', 'N/A'), 
        module['details'].get('Unpacked Size', 'N/A'),
        int(module['details'].get('Total Files', 'N/A').replace(',', '') if module['details'].get('Total Files', 'N/A') != 'N/A' else '0'), 
        int(module['details'].get('Issues', 'N/A').replace(',', '') if module['details'].get('Issues', 'N/A') != 'N/A' else '0'), 
        int(module['details'].get('Pull Requests', 'N/A').replace(',', '') if module['details'].get('Pull Requests', 'N/A') != 'N/A' else '0'), 
        module['details'].get('Last publish', 'N/A'), 
        module['url']
    ) for index, module in enumerate(modules)]

    insertAllQuery(insert_modules_query, module_info,'module_info')

    inverted_index = create_inverted_index(preprocessed_modules)

    # Define the SQL query to create the table
    create_modules_inverted_index_query = '''
    CREATE TABLE IF NOT EXISTS modules_inverted_index (
        term_id INTEGER PRIMARY KEY,
        term VARCHAR(255),
        doc_id INTEGER[]
    )
    '''
    createQuery(create_modules_inverted_index_query)

    insert_inverted_index = """ INSERT INTO modules_inverted_index (term_id, term, doc_ids) VALUES (%s, %s, %s)"""
    
    insertAllQuery(insert_inverted_index, inverted_index, 'modules_inverted_index')
      
    document_term_frequency = create_document_term_frequency(preprocessed_modules,inverted_index)
    
    create_modules_document_term_frequency_table_query = '''
    CREATE TABLE IF NOT EXISTS modules_document_term_frequency (
        doc_id INTEGER PRIMARY KEY,
        term_frequency JSONB
    )
    '''

    createQuery(create_modules_document_term_frequency_table_query)

    insert_document_term_frequency = """ INSERT INTO modules_document_term_frequency (doc_id, term_frequency) VALUES (%s, %s)"""

    insertAllQuery(insert_document_term_frequency, document_term_frequency, 'modules_document_term_frequency')

    pool.close()
    pool.join()



