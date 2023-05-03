import glob
import os
#remove images in /static/images
images = glob.glob("static/images/*.png")
for image in images:
    os.remove(image)

#remove db
#check if file exists
if os.path.exists("wiki_articles.db"):
    os.remove("wiki_articles.db")

#remove the directory chromadb/index and its contents
import shutil
#check if directory exists
if os.path.exists("chromadb/index"):
    shutil.rmtree("chromadb/index")
if os.path.exists("chromadb/chroma-collections.parquet"):
    os.remove("chromadb/chroma-collections.parquet")
if os.path.exists("chromadb/chroma-embeddings.parquet"):
    os.remove("chromadb/chroma-embeddings.parquet")
