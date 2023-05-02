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

def generate_anchor_tag(match):
    link_title = match.group(1).strip()
    link_url = url_for('article', title=link_title)
    return f'<a href="{link_url}">{link_title}</a>'




def generate_article(title, n=1):

    print("Generating article for", title)

    systemPrompt="""You are a Wikipedia article generator.

You are generating articles about a fictional world. These articles should include frequent nonsensical details
such as pigs that fly and people who can teleport.
    
articles can include titles for subheadings by using the ==title== syntax

articles can link to other articles by using the tag [link:some article title]
    
articles can include images by using the tag [image:some image prompt]

make sure to include multiple images, links and sections in each article!

make sure to include a [link:link to another article] around each proper noun in the article!

        
    """

    messages = [{"role": "system", "content": systemPrompt}]
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
    #split on "."s and new lines

    article_text_without_titles=re.sub(r'==.*==', '', article_text)
    sentences = re.split(r'\.|\n', article_text_without_titles)
    lowercase_words = ["the", "of", "in", "and"]
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
        properNounRegex = r'(?<!\w)(?:[A-Z][a-z]+(?:\s+(?:' + '|'.join(lowercase_words) + r'))*\s+)*[A-Z][a-z]+(?!\w)'
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

    return article_text

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        title = request.form["title"]
        if not table.find_one(title=title):
            content = generate_article(title)
            table.insert({"title": title, "content": content})
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
    if not article:
        content = generate_article(title)
        table.insert({"title": title, "content": content})
        article = table.find_one(title=title)

    content = article["content"]

    # Replace line breaks with <br>
    content = content.replace('\n', '<br>')

    # Replace ==title== with <h2>title</h2>
    content = re.sub(r'==(.+?)==', r'<h2>\1</h2>', content)

    # Handle image tags
    image_pattern = r'\[image:(.+?)\]'
    matches = re.finditer(image_pattern, content)
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
    content_with_links_and_images = re.sub(link_pattern, generate_anchor_tag, content_with_links_and_images)

    return render_template("article.html", title=article["title"], content=content_with_links_and_images)


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