import glob
import os
#remove images in /static/images
images = glob.glob("static/images/*.png")
for image in images:
    os.remove(image)

#remove db
os.remove("wiki_articles.db")

#remove the directory chromadb/index and its contents
import shutil
shutil.rmtree("chromadb/index")
