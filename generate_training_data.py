import lmdb
import xml.etree.ElementTree as ET
import networkx as nx
from tqdm import tqdm
import numpy as np
import re
import shutil
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import random

os.chdir('/Users/bebr1814/projects/wikipedia')



class WikipediaGame:
	def __init__(self, wiki_xml_path=None, graph_pkl=None):
		"""
		Initialize the Wikipedia Game class with either an XML path or an existing graph.
		
		Args:
			wiki_xml_path (str, optional): Path to the Wikipedia XML dump.
			graph (nx.DiGraph, optional): Existing NetworkX graph.
		"""


		self.exclude_prefixes = ['Module:', 'Template:', 'Wikipedia:', 'MediaWiki:', 'Category:', 'File:', 'Help:', 'Portal:', 'Draft:', 'Book:', 'User:', 'Special:', 'TimedText:', 'Talk:', 'Wikipedia talk:', 'MediaWiki talk:', 'Category talk:', 'File talk:', 'Template talk:', 'Module talk:']

		# DB to store text
		self.lmdb_path = "wikipedia_db"
		if os.path.exists(self.lmdb_path) and graph_pkl is None:
			shutil.rmtree(self.lmdb_path, ignore_errors=True) # delete LMDB database if it exists
		self.lmdb_env = lmdb.open(self.lmdb_path, map_size=20**9)  # 2GB max size

		if graph_pkl is not None:
			self.G = pickle.load(open(graph_pkl, "rb"))
			self.article_titles = set(self.G.nodes())
		elif wiki_xml_path is not None:
			self.wiki_xml = wiki_xml_path
			self.build_graph()
		else:
			raise ValueError("Either wiki_xml_path or graph must be provided")
		


	def store_article(self,title, content):
		with self.lmdb_env.begin(write=True) as txn:
			txn.put(title.encode(), content.encode())  # Store title → full text

	def get_article(self,title):
		with self.lmdb_env.begin() as txn:
			data = txn.get(title.encode())
			return data.decode() if data else None  # Retrieve full text by title
	
	def vectorize_articles(self, titles):
		texts = {title: self.clean_wikipedia_text(self.get_article(title)) for title in titles}
		vectorizer = TfidfVectorizer(stop_words='english',max_features=1000)
		vectors = {title: list(vector) for title, vector in zip(titles, vectorizer.fit_transform(texts.values()).toarray())}
		# pad all vectors to be exactly 1000 features
		for title in titles:
			if len(vectors[title]) < 1000:
				vectors[title] += [0] * (1000 - len(vectors[title]))
		return vectors

	def build_graph(self):
		"""Build the directed graph from Wikipedia XML."""
		print("Parsing XML...")
		tree = ET.parse(self.wiki_xml)
		root = tree.getroot()
		
		# Initialize a directed graph
		self.G = nx.DiGraph()
		self.article_titles = set()
		
		print("Building graph and extracting text content...")

		# get all article titles
		for page in tqdm(root[1:]):
			title = page.find('{http://www.mediawiki.org/xml/export-0.11/}title').text
			if any(title.startswith(prefix) for prefix in self.exclude_prefixes) or title in ['Global']:
				continue
			self.article_titles.add(title)
		
		
		# Iterate over the XML tree
		for page in tqdm(root[1:]):
			# Get the title of the page
			title = page.find('{http://www.mediawiki.org/xml/export-0.11/}title').text
			if any(title.startswith(prefix) for prefix in self.exclude_prefixes):
				continue

			# Get the text of the page
			text_element = page.find('{http://www.mediawiki.org/xml/export-0.11/}revision').find('{http://www.mediawiki.org/xml/export-0.11/}text')
			
			if text_element is not None:

				text = text_element.text
				if text is None or text == '':
					self.article_titles.remove(title)
					# Remove any links to this article from the graph
					if self.G.has_node(title):
						self.G.remove_node(title)
					continue

				# Store the text content in the LMDB database
				self.store_article(title, self.clean_wikipedia_text(text))

				# Find all links in the text using regex
				try:
					links = re.findall(r'\[\[([^|\]]+)(?:\|[^\]]*)?\]\]', text)
					# Exclude links that aren't valid articles
					links = list(self.article_titles.intersection(links))
					
					# Add the links to the graph
					if len(links) > 0:
						for link in links:
							self.G.add_edge(title, link)
					else:
						# Add the node even if it has no outgoing links
						self.G.add_node(title)
				except Exception as e:
					print(f"{title} processing error: {e}")
			else:
				print(f"{title} has no text content")
		
		# print("Vectorizing all articles...")
		# self.vectorize_all()
		
		print(f"Graph built with {len(self.G.nodes())} nodes and {len(self.G.edges())} edges")
	
	def clean_wikipedia_text(self,text):
		
		# Remove {{templates}}
		text = re.sub(r"\{\{.*?\}\}", "", text)

		# Remove [[File:...]] and similar media references
		text = re.sub(r"\[\[File:.*?\]\]", "", text)

		# Remove section headers (== Title ==)
		text = re.sub(r"==+.*?==+", "", text)

		# Remove [[Category:...]] tags (if present in raw text)
		text = re.sub(r"\[\[Category:.*?\]\]", "", text)

		# Replace [[linked text|display text]] with just "display text"
		text = re.sub(r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2", text)

		# Remove all remaining [[brackets]] (if any)
		text = re.sub(r"\[\[|\]\]", "", text)

		# Remove extra whitespace and newlines
		text = re.sub(r"\s+", " ", text).strip()

		# Remove \'
		text = text.replace("\'", "")
		
		return text


wiki_game = WikipediaGame(wiki_xml_path='/Users/bebr1814/projects/wikipedia/data/simplewiki-20250301-pages-articles-multistream.xml', graph_pkl='/Users/bebr1814/projects/wikipedia/pickles/simplewiki_graph.3.21.25.pkl')


def get_current_node_info(wiki_game, source=None, target=None, god_mode=False, negative_example_limit=None):
    G = wiki_game.G
    
    # Select random source and target if not provided
    if source is None or target is None:
        nodes = list(G.nodes)
        while True:
            source, target = random.sample(nodes, 2)
            if nx.has_path(G, source, target):
                break

    # Get all valid next steps if god_mode is enabled
    if god_mode:
        try:
            shortest_paths = nx.all_shortest_paths(G, source=source, target=target)
            correct_next_steps = {path[1] for path in shortest_paths}  # Unique valid next steps
        except nx.NetworkXNoPath:
            return []
    else:
        correct_next_steps = {None}  # Placeholder for unknown

    # Get source article's neighbors
    neighbors = list(G.successors(source))
    if not neighbors:
        return []  # Skip if no outgoing links

    # Limit negative examples to 5
    negative_examples = list(set(neighbors) - correct_next_steps)
    if negative_example_limit:
        if len(negative_examples) > negative_example_limit:
            negative_examples = random.sample(negative_examples, 5)

    # Combine positive and limited negative examples
    selected_neighbors = list(correct_next_steps) + negative_examples

    # Get TF-IDF vectors for source, target, and selected neighbors in one batch call
    articles_to_vectorize = {source, target} | set(selected_neighbors)  # Use a set to avoid duplicates
    vectors = wiki_game.vectorize_articles(articles_to_vectorize)

    # Precompute node degrees in a single pass
    degrees = G.in_degree(articles_to_vectorize)  # Returns a dict-like view
    out_degrees = G.out_degree(articles_to_vectorize)

    # Extract precomputed values for source and target
    source_vector = vectors[source]
    target_vector = vectors[target]
    source_indegree, source_outdegree = degrees[source], out_degrees[source]
    target_indegree, target_outdegree = degrees[target], out_degrees[target]

    # Construct training examples efficiently
    training_examples = [
        {
            "node_name": n,
            "source_vector": source_vector,
            "target_vector": target_vector,
            "neighbor_vector": vectors[n],  # Direct access without repeated lookups
            "source_indegree": source_indegree,
            "source_outdegree": source_outdegree,
            "target_indegree": target_indegree,
            "target_outdegree": target_outdegree,
            "neighbor_indegree": degrees[n],  # Precomputed
            "neighbor_outdegree": out_degrees[n],  # Precomputed
            "correct_next": int(n in correct_next_steps) if god_mode else None,
        }
        for n in selected_neighbors
    ]

    return training_examples



import json
import pandas as pd
import numpy as np
import h5py
import uuid

def compress_vector(vector):
	"""Convert a dense TF-IDF vector into a sparse dictionary of nonzero values."""
	return {int(i): v for i, v in enumerate(vector) if v != 0}

def decompress_vector(sparse_vector):
	"""Converts a sparse TF-IDF representation (dict {index: value}) back to a dense vector."""
	sparse_dict = json.loads(sparse_vector)
	dense_vector = np.zeros(1000)
	for index, value in sparse_dict.items():
		dense_vector[int(index)] = value  # Ensure index is int
	return dense_vector

def save_training_examples_hdf5(examples, filename):
    """Saves training examples to an HDF5 file, storing sparse TF-IDF vectors."""
    if not examples:
        return  # Skip empty input

    with h5py.File(filename, 'a') as f:
        for ex in examples:
            # Use a unique identifier for each example
            unique_id = str(uuid.uuid4())
            grp = f.create_group(unique_id)
            grp.create_dataset("source_vector", data=list(compress_vector(ex["source_vector"]).items()))
            grp.create_dataset("target_vector", data=list(compress_vector(ex["target_vector"]).items()))
            grp.create_dataset("neighbor_vector", data=list(compress_vector(ex["neighbor_vector"]).items()))
            grp.attrs["node_name"] = ex["node_name"]  # Store the node name as an attribute
            grp.attrs["source_indegree"] = ex["source_indegree"]
            grp.attrs["source_outdegree"] = ex["source_outdegree"]
            grp.attrs["target_indegree"] = ex["target_indegree"]
            grp.attrs["target_outdegree"] = ex["target_outdegree"]
            grp.attrs["neighbor_indegree"] = ex["neighbor_indegree"]
            grp.attrs["neighbor_outdegree"] = ex["neighbor_outdegree"]
            grp.attrs["correct_next"] = ex["correct_next"]

train_set = set()
while len(train_set) < 500:
	source, target = random.sample(list(wiki_game.G.nodes()), 2)

	if source != target and nx.has_path(wiki_game.G, source, target):
		shortest_path = nx.shortest_path(wiki_game.G, source, target)
		if len(shortest_path) > 2 and len(shortest_path) < 10:
			train_set.add((source, target))


for i, (source, target) in enumerate(train_set):
	
	path_examples = []
	current_node = source
	# print(f'{source} -> {target}')

	# print(f'{source} ({wiki_game.G.out_degree(source)}) -> ',end='')

	while current_node != target:
		# Get the next node info
		examples = get_current_node_info(wiki_game, source=current_node, target=target, god_mode=True, negative_example_limit=5)
		
		path_examples.extend(examples)

		# current_node = path_examples[-1]['correct_next']
		current_node = [e for e in examples if e['correct_next'] == 1][0]['node_name']

		# if current_node == target:
		# 	print(f'{current_node} ({wiki_game.G.out_degree(current_node)})')
		# else:
		# 	print(f'{current_node} ({wiki_game.G.out_degree(current_node)}) -> ',end='')
	
	# print(f'Path: {source} -> {target}')
	print(i)
		
	
	# Save the examples to a file
	# save_training_examples(path_examples, '/Users/bebr1814/projects/wikipedia/training_set/mini_set.tsv')
	save_training_examples_hdf5(path_examples, '/Users/bebr1814/projects/wikipedia/training_set/mini_set.hdf5')



