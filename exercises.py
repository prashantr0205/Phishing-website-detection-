import os.path

import requests as re
from bs4 import BeautifulSoup

URL="https://www.kaggle.com"
response=re.get(URL)
print("response-->", response, "\ntype -->", type(response))
print("text -->", response.text, "\ncontent -->", response.content, "\nstatus_code -->", response.status_code)

if response.status_code != 200:
    print("HTTP connection is not succesful. Try again please")
else:
    print("HTTP connection is succesful.")

soup= BeautifulSoup(response.content, "html.parser")

print("title with tags --> ", soup.title, "\ntitle without tags -->", soup.title.text)

for link in soup.findAll("link"):
    print(link.get("href"))
# In this code, soup is an object of the BeautifulSoup class, which is created by parsing an HTML or XML document. The findAll() method is used to find all occurrences of a particular tag, in this case, <link>. The loop then iterates over each <link> tag found, and the get() method is used to retrieve the value of the href attribute for each tag. Finally, the value of the href attribute is printed to the console.
print(soup.get_text())

# 1 CREATE A FOLDER TO SAVE HTML FILES
folder="mini_dataset"

if not os.path.exists(folder):
    os.mkdir(folder)
# 2 DEFINE A FUNCTION THAT SCRAPES AND RETURNS DATA
def scrape_content(URL):
    response=re.get(URL)
    if response.status_code ==200:
        print("HTTP connection is succesful for the URL:", URL)
        return response
    else:
        print("HTTP connection is not succesful for the URL:", URL)
        return none




# 3 DEFINE A FUNCTION TO SAVE HTNL FILES OF THE SCRAPED WEBPAGE IN A DIRECTORY
path = os.getcwd() + "/" + folder

def save_html(to_where, text, name):
    file_name= name + ".html"
    with open(os.path.join(to_where, file_name), "w", encoding='utf-8') as f: # One common encoding that supports a wide range of characters is UTF-8. 
        f.write(text)

test_text= response.text
save_html(path, test_text, "example")



# 4 DEFINE A URL LIST VARIABLE

URL_list = [
    "https://www.kaggle.com",
    "https://stackoverflow.com",
    "https://www.researchgate.net",
    "https://www.python.org",
    "https://www.w3schools.com",
    "https://wwwen.uni.lu",
    "https://github.com",
    "https://scholar.google.com",
    "https://www.mendeley.com",
    "https://www.overleaf.com"
]

# 5 DEFINE A FUNCTION WHICH TAKES URL LIST AND RUN STEP 2 AND 3 FOR EACH URL
def create_mini_dataset(to_where, URL_list):
    for i in range(0, len(URL_list)):
        content = scrape_content(URL_list[i])
        if content is not None:
            save_html(to_where, content.text, str(i))
        else:
            pass
    print("Mini Dataset is created!")

create_mini_dataset(path,URL_list)

# 6 CHECK IF YOU HAVE TEN DIFFERENT HTML FILES








