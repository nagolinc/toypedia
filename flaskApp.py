import os
import openai
from flask import Flask, render_template, request, redirect, url_for
import dataset
from diffusers import DiffusionPipeline, UniPCMultistepScheduler, DPMSolverMultistepScheduler
import tomesd
import re
from io import BytesIO
from flask import jsonify
import hashlib
from exampleArticles import exampleArticles
import torch
import argparse


import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

import signal
import sys

from urllib.parse import quote

#will this make chroma save?
def sigint_handler(signal, frame):
    global chroma_client, collection
    print("CTRL+C detected. Saving database and exiting...")
    chroma_client.persist()
    del collection
    del chroma_client
    sys.exit(0)

# Register the signal handler for SIGINT (CTRL+C)
signal.signal(signal.SIGINT, sigint_handler)

#read openai key from system environment variable
open_ai_key = os.environ["OPENAI_API_KEY"]
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=open_ai_key,
                model_name="text-embedding-ada-002"
            )

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "chromadb")

chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                         persist_directory=DB_DIR,
                                         anonymized_telemetry=False))
chroma_client.persist()
collection = chroma_client.get_or_create_collection(name="toypedia",embedding_function=openai_ef)
#collection.create_index()
thisCount=collection.count()
print("Found this many articles in the database:",thisCount)

# Set up OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Set up Flask
app = Flask(__name__)

# Set up a SQLite database using dataset
db = dataset.connect("sqlite:///wiki_articles.db")
table = db["articles"]




def generate_image_filename(prompt):
    hashed_prompt = hashlib.sha1(prompt.encode()).hexdigest()
    return f"static/images/{hashed_prompt}.png"

def generate_and_save_image(prompt,width=512,height=512):
    image = pipe(prompt+args.prompt_suffix,num_inference_steps=20,width=width,height=height).images[0]
    image_path = generate_image_filename(prompt)
    image.save(image_path, 'PNG', quality=90)

def generate_anchor_tag(match, source_title=None):
    link_title = match.group(1).strip()
    link_url = url_for('article', title=link_title)
    if source_title:
        encoded_source_title = quote(source_title)
        link_url += f'?source_title={encoded_source_title}'
    output = f'<a href="{link_url}">{link_title}</a>'

    print("what",link_title,source_title,output)

    return output

def generate_article(title, n=1,n_related=3,source=None):
    print("getting related articles")
    related_articles = collection.query(
        query_texts=[title],
        n_results=min(n_related,collection.count())
    )
    related_article_messages=[]
    related_article_titles=[]
    #go in reverse order so that the most related article is first
    for id in related_articles["ids"][0][::-1]:
        related_article=table.find_one(id=id)
        related_title=related_article["title"]
        #skip if it's the same as the source
        if related_title==source:
            continue
        related_content=related_article["content"]
        related_article_messages+=[{"role": "user", "content": f"write an article about {related_title}"},
                                   {"role":"assistant","content":related_content}]
        related_article_titles+=[related_title]
    #add source if it exists
    if source:
        source_article=table.find_one(title=source)
        source_content=source_article["content"]
        related_article_messages+=[{"role": "user", "content": f"write an article about {source_article}"},
                                   {"role":"assistant","content":source_content}]
        related_article_titles+=[source]

    print("got related articles",related_article_titles)

    print("Generating article for", title)

    systemPrompt="""You are a fictional Wikipedia article generator.

You are generating articles about a fictional world. These articles should include frequent nonsensical details
such as pigs that fly and people who can teleport.
    
articles can include titles for subheadings by using the ==title== syntax

articles can link to other articles by using the tag [link:some article title]
    
articles can include images by using the tag [image:some image prompt]

make sure to include multiple images, links and sections in each article!

make sure to include a [link:link to another article] around each proper noun in the article!

        
    """

    messages = [{"role": "system", "content": systemPrompt}]
    if len(related_article_messages)>0:
        messages+=related_article_messages
    else:
        messages+=exampleArticles
    messages+=[{"role": "user", "content": f"Write an article about {title}"}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        timeout=10,
        n=n,
    )

    print(response)

    article_text = response.choices[0].message.content

    #let's see if we can systematically add links around proper nouns
    #first, let's get all the proper nouns
    #we'll assume that any word that is Capital Case and not in the first sentence is a proper noun
    properNouns = []
    article_text_without_titles=re.sub(r'==.*==', '', article_text)
    #split on "."s and new lines and "?"s !s
    sentences = re.split(r'[.?!:\n]', article_text_without_titles)
    lowercase_words = ["the", "of", "in", "and","for"]
    for sentence in sentences:
        #strip whitespace
        sentence = sentence.strip()
        #skip if sentence is empty
        if len(sentence)==0:
            continue        
        #change first letter to lowercase
        sentence = sentence[0].lower() + sentence[1:]
        #regex that matches a set of words that are capital case such as 'Bob Smith' in 'hello Bob Smith'
        #properNounRegex = r'(?<!\w)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?!\w)'
        #properNounRegex = r'(?<!\w)(?:[A-Z][a-z]+(?:\s+(?:' + '|'.join(lowercase_words) + r'))*\s+)*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?!\w)'
        properNounRegex = r'(?<!\w)(?:[A-Z][a-z]*\b(?:\s+(?:' + '|'.join(lowercase_words) + r'))*\s+)*[A-Z][a-z]*\b(?:\s+[A-Z][a-z]*)*(?!\w)'
        matches = re.finditer(properNounRegex, sentence)
        for match in matches:
            properNoun=match.group(0)
            properNouns.append(properNoun)
            #print("What happened?",sentence,properNoun)

    #remove duplicates
    properNouns = list(set(properNouns))
    #now let's add links around first instance of each proper noun (should only do one substitution per proper noun)
    for properNoun in properNouns:
        #find first instance of proper noun
        firstIndex = article_text.find(properNoun)
        #make sure it isn't already a link (must not be inside of brackets)
        #find the first [ before firstIndex
        firstOpenBracketIndex = article_text.rfind("[", 0, firstIndex)
        if firstOpenBracketIndex==-1:
            #we're good to go
            pass
        else:
            #find the first ] after firstOpenBracketIndex
            firstCloseBracketIndex = article_text.find("]", firstOpenBracketIndex)
            if firstCloseBracketIndex==-1:
                #we're good to go
                pass
            else:
                #if if firstclosebracketindex is after firstindex, then we skip this proper noun
                if firstCloseBracketIndex>firstIndex:
                    continue

        article_text = article_text.replace(properNoun, f'[link:{properNoun}]',1)


    #make sure there's at least one image, if not, add one to the top
    if not re.search(r'\[image:.*\]', article_text):
        article_text = f'[image:{title}]\n\n' + article_text

    return article_text

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        title = request.form["title"]
        if not table.find_one(title=title):
            content = generate_article(title,n_related=args.related_articles)
            table.insert({"title": title, "content": content})
            #get id of article
            article_id=table.find_one(title=title)["id"]
            #add to chroma as well
            collection.add(
                documents=[content],
                metadatas=[{"title": title}],
                ids=[str(article_id)]
            )
            chroma_client.persist()

        return redirect(url_for("article", title=title))
    
    #get most recent articles
    most_recent_articles=table.find(order_by='-id', _limit=10)
    #format them for display
    most_recent_html=""
    for article in most_recent_articles:
        #create a link
        link=url_for("article", title=article["title"])
        #add to html
        most_recent_html+=f'<a href="{link}">{article["title"]}</a><br>'

    return render_template("index.html",most_recent_html=most_recent_html)

@app.route("/article/<title>")
def article(title):
    article = table.find_one(title=title)

    #get source_title from url parameters
    source_title = request.args.get('source_title')

    print("source_title",source_title)

    if not article:
        content = generate_article(title,source=source_title,n_related=args.related_articles)
        table.insert({"title": title, "content": content})
        article = table.find_one(title=title)
        article_id=table.find_one(title=title)["id"]
        #add to chroma as well
        collection.add(
            documents=[content],
            metadatas=[{"title": title}],
            ids=[str(article_id)]
        )
        chroma_client.persist()

    content = article["content"]

    # Replace line breaks with <br>
    content = content.replace('\n', '<br>')

    # Replace ==title== with <h2>title</h2>
    content = re.sub(r'==(.+?)==', r'<h2>\1</h2>', content)

    # Handle image tags
    image_pattern = r'\[image:(.+?)\]'
    matches = re.finditer(image_pattern, content,flags=re.IGNORECASE)
    content_with_links_and_images = content

    for match in matches:
        image_prompt = match.group(1).strip()
        image_path = generate_image_filename(image_prompt)

        if not os.path.exists(image_path):
            generate_and_save_image(image_prompt,width=args.width,height=args.height)

        img_tag = f'<img src="/{image_path}" alt="{image_prompt}"  title="{image_prompt}" />'
        content_with_links_and_images = content_with_links_and_images.replace(match.group(0), img_tag)

    # Handle link tags
    link_pattern = r'\[link:(.+?)\]'
    generate_anchor_tag_with_source = lambda match: generate_anchor_tag(match,source_title=title)
    content_with_links_and_images = re.sub(link_pattern, generate_anchor_tag_with_source, content_with_links_and_images)

    #get related articles with chroma
    results = collection.query(
        query_texts=[article["content"]],
        n_results=min(args.related_articles,collection.count())
    )

    related_links=""
    print(results['metadatas'])
    for id in results["ids"][0]:
        related_article=table.find_one(id=id)
        related_title=related_article["title"]
        #skip if it's the same article
        if related_title==title:
            continue
        #create link to article
        link=url_for("article", title=related_title)
        #add to html
        related_links+=f'<a href="{link}">{related_title}</a><br>'


    return render_template("article.html", title=article["title"], content=content_with_links_and_images,related=related_links)


@app.route("/update-article/<title>", methods=["POST"])
def update_article(title):
    data = request.json
    updated_content = data.get("content", "").strip()
    if not updated_content:
        return jsonify({"success": False, "message": "No content provided."}), 400

    article = table.find_one(title=title)
    if not article:
        return jsonify({"success": False, "message": "Article not found."}), 404

    table.update({"id": article["id"], "title": title, "content": updated_content}, ["id"])
    return jsonify({"success": True})

@app.route("/get-article-content/<title>")
def get_article_content(title):
    article = table.find_one(title=title)
    if not article:
        return jsonify({"success": False, "message": "Article not found."}), 404
    return jsonify({"success": True, "content": article["content"]})

if __name__ == "__main__":

    #argparse for diffusion-model, width, and height
    parser = argparse.ArgumentParser(description='Generate Wikipedia articles.')
    parser.add_argument('--diffusion-model', type=str, default="andite/anything-v4.0")
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    #prompt suffix
    parser.add_argument('--prompt-suffix', type=str, default=", masterpiece, best quality")
    #number of related articles to fetch
    parser.add_argument('--related-articles', type=int, default=5)
    
    
    args = parser.parse_args()

    print(args)

    pipe = DiffusionPipeline.from_pretrained(args.diffusion_model,
                                            torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.safety_checker = None
    tomesd.apply_patch(pipe, ratio=0.5)

    app.run(debug=True, use_reloader=False)