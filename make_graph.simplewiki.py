import lmdb
import xml.etree.ElementTree as ET
import networkx as nx
from tqdm import tqdm
import numpy as np
import re
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

os.chdir('/Users/bebr1814/projects/wikipedia')


class WikipediaGame:
	def __init__(self, wiki_xml_path=None, graph_pkl=None):

		self.exclude_prefixes = ['Module:', 'Template:', 'Wikipedia:', 'MediaWiki:', 'Category:', 'File:', 'Help:', 'Portal:', 'Draft:', 'Book:', 'User:', 'Special:', 'TimedText:', 'Talk:', 'Wikipedia talk:', 'MediaWiki talk:', 'Category talk:', 'File talk:', 'Template talk:', 'Module talk:']

		# DB to store text
		self.lmdb_path = "wikipedia_db"
		if os.path.exists(self.lmdb_path) and graph_pkl is None:
			shutil.rmtree(self.lmdb_path, ignore_errors=True) # delete LMDB database if it exists
		self.lmdb_env = lmdb.open(self.lmdb_path, map_size=20**9)  # 2GB max size

		# DB to store vectors
		# This should be generated beforehand
		self.vec_lmdb_path = "vector_db"
		if os.path.exists(self.vec_lmdb_path):
			self.vec_env = lmdb.open(self.vec_lmdb_path, readonly=True, lock=False)
		else:
			self.vec_env = None


		if graph_pkl is not None:
			self.G = pickle.load(open(graph_pkl, "rb"))
			self.article_titles = set(self.G.nodes())
		elif wiki_xml_path is not None:
			self.wiki_xml = wiki_xml_path
			print('Building Graph...')
			self.build_graph()
		else:
			raise ValueError("Either wiki_xml_path or graph must be provided")

		print('Calculating PageRank and Betweenness...')
		self.pagerank = nx.pagerank(self.G, alpha=0.85)
		self.betweenness = nx.betweenness_centrality(self.G, normalized=True, k=1000)

	def store_article(self,title, content):
		with self.lmdb_env.begin(write=True) as txn:
			txn.put(title.encode(), content.encode())  # Store title → full text

	def get_article(self,title):
		with self.lmdb_env.begin() as txn:
			data = txn.get(title.encode())
			return data.decode() if data else None  # Retrieve full text by title
	
	def get_vector(self, title):
		if self.vec_env is None:
			raise RuntimeError("Vector LMDB is not initialized. Make sure vector_db exists.")

		with self.vec_env.begin() as txn:
			data = txn.get(title.encode())
			if data is None:
				return None
			return np.frombuffer(data, dtype=np.float32)

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
		for page in root[1:]:
			title = page.find('{http://www.mediawiki.org/xml/export-0.11/}title').text
			if any(title.startswith(prefix) for prefix in self.exclude_prefixes) or title in ['Global']:
				continue
			self.article_titles.add(title)
		
		
		# Iterate over the XML tree
		for page in root[1:]:
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


wiki_game = WikipediaGame(wiki_xml_path='/Users/bebr1814/projects/wikipedia/data/simplewiki-20250301-pages-articles-multistream.xml')
print('Graph built')

with open('/Users/bebr1814/projects/wikipedia/pickles/simplewiki_full_graph.4.15.25.pkl', 'wb') as f:
	pickle.dump(wiki_game.G, f)

with open('/Users/bebr1814/projects/wikipedia/pickles/simplewiki_full_graph.betweenness.pkl', 'wb') as f:
	pickle.dump(wiki_game.betweenness, f)

print('Graph saved to /Users/bebr1814/projects/wikipedia/pickles/simplewiki_full_graph.4.10.25.pkl')

