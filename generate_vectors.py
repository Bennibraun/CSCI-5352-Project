import os
import pickle
import torch
import lmdb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import xml.etree.ElementTree as ET
import re
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import shutil
import numpy as np
import sys



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


game = WikipediaGame(graph_pkl='/Users/bebr1814/projects/wikipedia/pickles/simplewiki_full_graph.4.15.25.pkl')
assert game.get_article('Aaron Clauset') is not None

# Check for CUDA
print("Checking for CUDA")
print(torch.version.cuda)
if torch.cuda.is_available():
    print("CUDA is available (GPU)")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("MPS is available (M1 Mac)")
    device = torch.device("mps")
else:
    print("CUDA is not available (CPU)")
    device = torch.device("cpu")


# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)


# Set up LMDB to store vectors
vec_lmdb_path = "vector_db"
if os.path.exists(vec_lmdb_path):
	print("Removing existing vector LMDB...")
	shutil.rmtree(vec_lmdb_path)

vec_env = lmdb.open(vec_lmdb_path, map_size=3*(10**9))  # 3GB

# Process in batches
titles = list(game.article_titles)
batch_size = 512

with vec_env.begin(write=True) as txn:
	# for i in tqdm(range(0, len(titles), batch_size)):
	for i in range(0, len(titles), batch_size):
		if i % 1000 == 0:
			print(f"Processed {i} articles")
		batch_titles = titles[i:i+batch_size]
		texts = [game.get_article(t) for t in batch_titles]
		texts = [game.clean_wikipedia_text(text) for text in texts if text is not None]
		embeddings = model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
		for title, vec in zip(batch_titles, embeddings):
			txn.put(title.encode(), vec.astype('float32').tobytes())


