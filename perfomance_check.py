
import subprocess
import sys

def install_mypackage(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
install_mypackage('sentence_transformers')    
 
install_mypackage('flair')  
from gettext import install
import sentence_transformers
from sentence_transformers import SentenceTransformer, util


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

html = """<b>About this role:</b><br/><br/>Wells Fargo is seeking a Customer Service Representative...<br/><br/><b><b>In this role, you will:</b></b><ul><li>Support financial products and services</li><li>Identify opportunities to improve customer service experience and offer ideas to resolve internal and external customer issues to provide first call resolution</li><li>Perform moderately complex customer support tasks</li><li>Receive direction from customer service supervisor and escalate non-routine questions</li><li>Interact with customer service team on complex customer issues</li><li>Provide training on policies and procedures, and provide guidance to less experienced individuals, as well as internal and external customers</li></ul><b><b>Required Qualifications, US:</b></b><ul><li>2+ years of Customer Service, Financial Services or Contact Center experience, or equivalent demonstrated through one or a combination of the following: work experience, training, military experience, education</li></ul><b><b>Desired Qualifications: </b></b><ul><li>A BS/BA degree or higher (DQO0000001) <br/>Brokerage industry experience (DQO0003825) <br/>Contact center experience (DQO0003897) <br/>Military experience in personnel benefits management, processing military personnel orders or transitions, wartime readiness operations, human resources or military recruiting (DQO0012289)</li><li>Military intelligence or analytics experience including operational management, project management, mission evolution management and finance management (DQO0012290)</li><li>Experience working with military protocol and instructions, enlisted evaluations, officer/leadership reporting, and assistance with keeping military personnel combat-ready and effective (DQO0012291)</li></ul><b> @RWF22 </b> <br/><br/><b>We Value Diversity</b> <br/><br/>At Wells Fargo, we believe in diversity, equity and inclusion in the workplace; accordingly, we welcome applications for employment from all qualified candidates, regardless of race, color, gender, national origin, religion, age, sexual orientation, gender identity, gender expression, genetic information, individuals with disabilities, pregnancy, marital status, status as a protected veteran or any other status protected by applicable law. <br/><br/>Employees support our focus on building strong customer relationships balanced with a strong risk mitigating and compliance-driven culture which firmly establishes those disciplines as critical to the success of our customers and company. They are accountable for execution of all applicable risk programs (Credit, Market, Financial Crimes, Operational, Regulatory Compliance), which includes effectively following and adhering to applicable Wells Fargo policies and procedures, appropriately fulfilling risk and compliance obligations, timely and effective escalation and remediation of issues, and making sound risk decisions. There is emphasis on proactive monitoring, governance, risk identification and escalation, as well as making sound risk decisions commensurate with the business unit&apos;s risk appetite and all risk and compliance program requirements. <br/><br/>Candidates applying to job openings posted in US: All qualified applicants will receive consideration for employment without regard to race, color, religion, sex, sexual orientation, gender identity, or national origin. <br/><br/>Candidates applying to job openings posted in Canada: Applications for employment are encouraged from all qualified candidates, including women, persons with disabilities, aboriginal peoples and visible minorities. Accommodation for applicants with disabilities is available upon request in connection with the recruitment process. <br/><br/>"""
org_data = convert(html)
html1 = """<html><body><b>About this role: </b><br/><br/>Wells Fargo is seeking a Customer Service Representative.<br/><br/><b><b>In this role, you will: </b></b><ul><li>Support financial products and services</li><li>Identifying opportunities to improve customer service experience and offering ideas to resolve internal and external issues will give first call resolution</li><li>Perform moderately complex customer support tasks</li><li>Receive direction from customer service supervisor and escalate non-routine questions</li><li>On complex customer issues, interact with customer service team</li><li>Provide training on policies and procedures and provide guidance for less experienced people as well as internal and external customers</li></ul><b><b>Required Qualifications, US: </b></b><ul><li>2  years of Customer Service, Financial Services or Contact Center experience, or equivalent demonstrated through one or a combination of the following:  work experience, training, military experience, education</li></ul><b><b>Desired Qualifications: </b></b><ul><li>A BS/BA degree or higher (DQO0000001) <br/>Brokerage industry experience (DQO0003825) <br/>Contact center experience (DQO0003897) <br/>Military experience in personnel benefits management, processing military personnel orders or transitions, wartime readiness operations, human resources or military recruiting (DQO0012289)</li><li>Military intelligence or analytics experience including operational management, project management, mission evolution management and finance management (DQO0012290)</li><li>Experience working with military protocol and instructions, enlisted evaluations, officer/leadership reporting, and assistance with keeping military personnel combat-ready and effective (DQO0012291)</li></ul><b> @RWF22 </b> <br/><br/><b>We Value Diversity</b> <br/><br/>At Wells Fargo, we believe in diversity, equity and inclusion in the workplace; accordingly, we welcome applications for employment from all qualified candidates, regardless of race, color, gender, national origin, religion, age, sexual orientation, gender identity, gender expression, genetic information, individuals with disabilities, pregnancy, marital status, status as a protected veteran or any other status protected by applicable law. <br/><br/>The employees of the company support our focus on building strong customer relationships while also emphasizing risk mitigated and compliance-driven culture that's critical to the success of our customers and company. They are accountable for execution of all applicable risk programs (Credit, Market, Financial Crimes, Operational, Regulatory Compliance), which includes effectively following and adhering to applicable Wells Fargo policies and procedures, appropriately fulfilling risk and compliance obligations, timely and effective escalation and remediation of issues, and making sound risk decisions. It requires proactive monitoring and governance, risk identification and escalate, as well as making sound risk decisions that correlate with the business unit's risk appetite and all risk and compliance program requirements. <br/><br/>Candidates applying to job openings posted in US:  All qualified applicants will receive consideration for employment without regard to race, color, religion, sex, sexual orientation, gender identity, or national origin. <br/><br/>Candidates applying to job openings posted in Canada:  Applications for employment are encouraged from all qualified candidates, including women, persons with disabilities, aboriginal peoples and visible minorities. Accommodation for applicants with disabilities is available upon request in connection with the recruitment process. <br/><br/></body></html>"""
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

