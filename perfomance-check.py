from flair.data import Sentence
from flair.models import SequenceTagger
tagger = SequenceTagger.load("flair/ner-english-large")
from bs4 import BeautifulSoup
import re
def convert(html):
    description= []
    empty = [""]
    empty[-1]
    empty_list=[]
    soup = BeautifulSoup(html, "lxml")
    tags_list = list(set([tag.name for tag in soup.find_all()]))
    tags_list = ["/"+i for i in tags_list]+tags_list
    tags_list.append("br/")
    tags_list.append("p/")
    for tag in soup.find_all():
        if (str(tag) in str(empty[-1])):
            continue
        else:
            empty.append(tag)
    for i in empty[1:]:
        empty_list.append(re.findall(r'<>[^>]*>.*?</[^>]*>(?:<[^>]*/>)?|[^<>]+', str(i)))
    for i in range(len(empty_list)):
            for j in range(len(empty_list[i])):
                if  not(empty_list[i][j].isspace()):
                    if empty_list[i][j] in tags_list:
                        continue
                    else:
                        description.append(empty_list[i][j]) 
    return description

from sentence_transformers import SentenceTransformer, util
def paracheck(orginal_input,modified_input):
    orginal_input.strip()
    modified_input.strip()
    if orginal_input.__eq__(modified_input):
        return 1
    else:
        similarity_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
        embedding_1=similarity_model.encode(orginal_input,convert_to_tensor=False)
        embedding_2=similarity_model.encode(modified_input, convert_to_tensor=False)
        similarity=util.pytorch_cos_sim(embedding_1, embedding_2).numpy()[0][0]
        return similarity

def sentence_length(org_senten,mod_senten):
    org_word = len(org_senten.split())
    mod_word = len(mod_senten.split())
    if org_word == mod_word:
        return [False,org_word,mod_word]
    else:
        return [True,org_word,mod_word]

def pres_asterisk(org_desc,mod_desc):
    if "*" in org_desc:
        if org_desc.__eq__(mod_desc):
            return False
        elif "*" in org_desc and "*" in mod_desc:
            if org_desc.count("*") == mod_desc.count("*"):
                at_first=0
                at_last=0
                at_middle=0
                c=org_desc.count("*")
                c1=mod_desc.count("*")
                i=0
                j=-1
                at_first1=0
                at_last1=0
                at_middle1=0
                while True:
                    if org_desc[i]=="*":
                        at_first+=1
                        i+=1
                    if org_desc[j]=="*" :
                        at_last+=1
                        j-=1
                    if org_desc[i]!="*" and org_desc[j]!="*":
                        break
                at_middle=c-at_first-at_last
                i=0
                j=-1
                while True:
                    if mod_desc[i]=="*":
                        at_first1+=1
                        i+=1
                    if mod_desc[j]=="*" :
                        at_last1+=1
                        j-=1
                    if mod_desc[i]!="*" and mod_desc[j]!="*":
                        break
                at_middle1=c1-at_first1-at_last1
                if at_middle>0:
                    indexes = [x for x, v in enumerate(org_desc) if v == '*']
                    index_weight=sum(indexes)
                    if (index_weight)//len(indexes) <len(org_desc)//2:
                        if at_first1==(at_middle+at_first):
                            return False
                        else:
                            return True
                    else:
                        if at_last1==(at_middle+at_last):
                            return False
                        else:
                            return True
                elif at_first!=at_first1:
                    return True
                elif at_last>=1 and (at_last!=at_last1):
                    return True
                else:
                    return False
            else:
                return True
    elif "*" in org_desc and "*" not in mod_desc or "*" not in org_desc and "*"  in mod_desc:
        return True
    else:
        return False

def presence_colon(org_sen,mod_sen):
    if ":" in org_sen and ":" in mod_sen:
        org_ind = org_sen.index(':')
        mod_ind = mod_sen.index(':')
        original_col_data = org_sen[:org_ind]
        modified_col_data = mod_sen[:mod_ind]
        org_count = org_sen.count(':')
        mod_count = mod_sen.count(':')
        if original_col_data == modified_col_data and org_count == mod_count:
            return False
        else:
            return True
    elif (":" in org_sen and ":" not in mod_sen) or (":" not in org_sen and ":" in mod_sen):
        return True
    else:
        return False

def Cname_Loc_compare(Original_desc,Modified_desc):
    ner_model = Sentence(Original_desc)
    tagger.predict(ner_model)
    l=ner_model.get_spans('ner')
    company_list_original=[i.text for i in l if i.score>0.99 and i.tag=="ORG"]
    Locations_original=[i.text for i in l if i.score>0.99 and i.tag=="LOC"]
    ner_model = Sentence(Modified_desc)
    tagger.predict(ner_model)
    l=ner_model.get_spans('ner')
    company_list_modified=[i.text for i in l if i.score>0.99 and i.tag=="ORG"]
    Locations_modified=[i.text for i in l if i.score>0.99 and i.tag=="LOC"]
    result=[]
    if len(company_list_original)!=len(company_list_modified):
        result.append(True)
    else:
        f=0
        for i in range(len(company_list_modified)):
            if company_list_original[i]!=company_list_modified[i]:
                f=1
                break
        if f==0:
            result.append(False)
        else:
            result.append(True)
    if len(Locations_original)!=len(Locations_modified):
        result.append(True)
    else:
        f=0
        for i in range(len(Locations_original)):
            if Locations_original[i]!=Locations_modified[i]:
                f=1
                break
        if f==0:
            result.append(False)
        else:
            result.append(True)
    return result

def fullstop(orginial_desc,modified_desc):
    original_count=orginial_desc.count('.')
    modified_count=modified_desc.count('.')
    difference=original_count-modified_count
    if original_count==modified_count:
        return [False,difference]
    else:
        return [True,difference]

html = input("Enter the orginal description")
org_data = convert(html)
html1 = input("Enter the modified description")
mod_data = convert(html1)
org_desc = [j for i in org_data for j in i.split(". ")]
mod_desc = [j for i in mod_data for j in i.split(". ")]

para_list = [round(paracheck(i,j), 3) for i,j in zip(org_desc,mod_desc)]
paraphased_list = [i for i in para_list if i!=1]
paraphased_count = len(paraphased_list)
average = sum(paraphased_list)*100/len(paraphased_list)
print("Average paraphased : ",average)

length_match = [(sentence_length(i,j)) for i,j in zip(org_desc,mod_desc)]
print("Length_match : ",length_match)

asterisk_list = [(pres_asterisk(i,j)) for i,j in zip(org_desc,mod_desc)]
print("asterisk_list : ",asterisk_list)

cname_loc_list = [(Cname_Loc_compare(i,j)) for i,j in zip(org_desc,mod_desc)]
print("cname_loc_list : ",cname_loc_list)

colon_list = [(presence_colon(i,j)) for i,j in zip(org_desc,mod_desc)]
print("colon_list : ",colon_list)

fullstop_list = [(fullstop(i,j)) for i,j in zip(org_desc,mod_desc)]
print("fullstop_list : ",fullstop_list)

