#Version 4
import os
#import psycopg2
from flask import Flask, render_template,jsonify, request, g
import pandas as pd
import numpy as np
from simple_salesforce import Salesforce, SalesforceLogin
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
import json
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'XYZ')


#def connect_db():
#    return psycopg2.connect(os.environ.get('DATABASE_URL'))


#@app.before_request
#def before_request():
  #  g.db_conn = connect_db()


#@app.route('/')
#def index():
 #   cur.execute("SELECT * FROM country;")
  #  return render_template('index.html', countries=cur.fetchall())
 #      return "Hello Krsna"   
     
     
'''cred = [{'username':'B'}]
@app.route('/', methods=['GET'])
def index():
    return 'Test'
    
'''

class CaseOperation():
    def __init__(self, sf, request):
        print('<------Case------>')
        inputCaseId = request.json['CaseId']
        #loginData = AuthAndRetrieveData(request.json['userName'], request.json['password'], request.json['token'], request.json['isSandbox'])
        #sf = loginData.authentication("Test")
        retrieveDataObj = RetrieveCaseData()
        queryCase = retrieveDataObj.retrieveData(sf, inputCaseId)
        df = pd.DataFrame(queryCase) 
        dfOriginal = df
        print(' 51-->', df)
        print('columns -->', df.columns)
        preprocessingObj = PreProcessing()
        df = preprocessingObj.preProcessing(df)  
        print("df 55----->",df.head(1))
        #input = "Issue: After the customer refreshed his sanbox he is getting the folllowing error when the user goes to the Remedyforce CMDB   Unexpected end-of-input: was expecting closing '\"\"' for name at [line:1, column:511] An unexpected error has occurred. Your solution provider has been notified. (System)"
        input = request.json['CaseDescription']
        print(input)
        inputMod = preprocessingObj.identifyModule(input)
        inputToken = preprocessingObj.inputTokenizing(input)
        cosinLogicObj = CosinLogic()
        df = cosinLogicObj.cosin(dfOriginal, inputToken,inputCaseId)
        print('columns 65-->', df.columns)
        df = cosinLogicObj.maxSimilarityLogic(df, inputMod)
        print('columns 67-->', df.columns)
        #print(df)
        dfOriginal[['Module']] = df[['Module']]
        predictionModelObj = PredictionModel();
        predictionValue = predictionModelObj.getPrediction(dfOriginal, inputCaseId)
        df = df[['Id','Module', 'Similarity_Index', 'Case_Owner__c']]
        #print(df)
        df = df.rename(columns = {"Id": "Suggested_Case__c", "Module":"Module__c", "Similarity_Index":"Similarity_Index__c", "Case_Owner":"Case_Owner__c"}) 
        df['Case_Name__c'] = inputCaseId
        df['SLA_Prediction__c'] = predictionValue
        #print("DF--->",df)
        data = df.to_json(orient='records')
        data = json.loads(data)
        print("Json--->",data)
        #data = [{"Module__c":"CMDB,","Similarity_Index__c":0.87055},{"Module__c":"CMDB,","Similarity_Index__c":0.84265},{"Module__c":"Upgrade, CMDB,","Similarity_Index__c":0.70825}] 
         #      [{"Module__c":"CMDB,","Similarity_Index__c":0.87055},{"Module__c":"CMDB,","Similarity_Index__c":0.84265},{"Module__c":"Upgrade, CMDB,","Similarity_Index__c":0.70825}]
        sf.bulk.Suggested_Case__c.insert(data) 
        #sf.bulk.Suggested_Case__c.insert(df)
        #return df #jsonify({'cred': c
        # reds})

class CosinLogic:
    def cosin(self, df, inputToken,inputCaseId):
        df1 = df[df.Id != inputCaseId]
        clean = []
        clean.append(0)
        for texts in df1.cleaned_text:
            l1 = []
            lst = []
            text_list = word_tokenize(texts)
            sw = stopwords.words('english')
            text_set = {w for w in text_list if not w in sw} 
            rvector = inputToken.union(text_set)
            
            for w in rvector: 
                if w in inputToken: l1.append(1) # create a vector 
                else: l1.append(0) 
                if w in text_set: lst.append(1) 
                else: lst.append(0) 
            c = 0
            
            for i in range(len(rvector)): 
                c+= l1[i]*lst[i] 
            cosine = 0.0
            if (sum(l1)*sum(lst)) != 0:
                cosine = c / float((sum(l1)*sum(lst))**0.5)
            clean.append((np.round(cosine,4)))
            #clean.append(' '.join(str(np.round(cosine,4))))
            #print("similarity: in new Case n",name, ' is ', cosine) 
        df['Similarity_Index'] = clean
        return df
    
    def maxSimilarityLogic(self, df, inputMod):
        suggestedDF = df.sort_values('Similarity_Index', ascending=False).head(20)
        suggestedDF = suggestedDF[['Id','CaseNumber', 'Description','cleaned_text','Status','Module', 'Similarity_Index', 'Case_Owner__c', 'CASE_AGGRAVATION__c' ]]
        i = -1
        max_sim = 1
        for item in suggestedDF.Module:
            i += 1
            if len(inputMod) > 0:
                if inputMod[0] in item:
                    max_sim = suggestedDF.iloc[i]['Similarity_Index']
                    break
        adjusted_param1 = (1-max_sim)*.7
        #print(adjusted_param1)
        adjusted_param2 = (1-max_sim)*.4
        #print(adjusted_param2)
        suggestedDF = suggestedDF.reset_index()
        i = -1
        for item in suggestedDF.Module:
            i += 1
            if len(inputMod) > 0:
                if inputMod[0] in item:
                    suggestedDF.at[i,'Similarity_Index'] = suggestedDF.at[i,'Similarity_Index']+ adjusted_param1
                else:
                    suggestedDF.at[i,'Similarity_Index'] = suggestedDF.at[i,'Similarity_Index']+ adjusted_param2
        suggestedDF = suggestedDF.sort_values('Similarity_Index', ascending=False).head(20)
        df4 = self.suggestedOwner(suggestedDF)
        df5 = df4.to_json(orient='records')
        suggestedDF['json']= ''
        suggestedDF['json'][0] = df5
        suggestedDF = suggestedDF.sort_values('Similarity_Index', ascending=False).head(5)
        return suggestedDF

    def suggestedOwner(self, suggestedDF):
        t1 = suggestedDF.Case_Owner__c.value_counts()
        t2 = t1.to_dict()
        df2 = pd.DataFrame.from_dict(t2,orient='index', columns=["Case_Count"])
        avg_counts = []
		res_Cases = []
        for item in t2:
			res_Case = []
            loc_avg = 0
            loc_avg = suggestedDF[suggestedDF['Case_Owner__c'] == item]['CASE_AGGRAVATION__c'].mean()
			res_Case = suggestedDF[suggestedDF['Case_Owner__c'] == item]['CASENUMBER'].tolist()
            avg_counts.append(loc_avg)
			res_Cases.append(res_Case)
            print('average --> ', loc_avg)
        df2['Avg_Days'] = avg_counts
		df2['Case_Number'] = res_RFAs
        df3 = df2.sort_values('Avg_Days', ascending=True).head(20)
        return df3

class PreProcessing:
    def inputTokenizing(self, input):
        sw = stopwords.words('english') 
        input_token = word_tokenize(input) 
        input_set = {w for w in input_token if not w in sw} 
        return input_set
        
    def identifyModule(self, input):
        modules = ['CMDB', 'Console', 'Smart Sync', 'SLA', 'Service Request', 'Request Definition', 'Self Service', 'Pentaho',
          'Template', 'Category', 'Usage Metric', 'SRD', 'Service Target', 'Task', 'Broadcast', 'Change Request', 
          'Incident', 'Approval', 'Lookup', 'CPU Time', 'KA', 'Knowledge Article', 'Workflow', 'DmlException', 'BCM',
          'Integration', 'Activity Feed', 'Smart Suggestion', 'Upgrade', 'REST API', 'Discovery', 'Scheduled Job', 
          'System.LimitException:', 'Tasks Closed Controller', 'Email Listener', 'lightning', 'Delegated Approver', 
          'time based workflow', 'report', 'FIELD_CUSTOM_VALIDATION_EXCEPTION', 'Task Template', 'System.DmlException:',
          'Primary Client','SSO']
        input_mod = set()
        print("input-->",input)
        for item in modules:
            item1 = ' ' +item.lower() + ' '
            sentence1 = input.lower()
            if item1 in sentence1:
                input_mod.add(item)  
        input_mod = list(input_mod)
        print("Module-->", input_mod)
        return input_mod
        
    def preProcessing(self, df):
        #print(df.columns)
        #print(df['Problem_Description_and_Definition__c'])
        modules = ['CMDB', 'Console', 'Smart Sync', 'SLA', 'Service Request', 'Request Definition', 'Self Service', 'Pentaho',
          'Template', 'Category', 'Usage Metric', 'SRD', 'Service Target', 'Task', 'Broadcast', 'Change Request', 
          'Incident', 'Approval', 'Lookup', 'CPU Time', 'KA', 'Knowledge Article', 'Workflow', 'DmlException', 'BCM',
          'Integration', 'Activity Feed', 'Smart Suggestion', 'Upgrade', 'REST API', 'Discovery', 'Scheduled Job', 
          'System.LimitException:', 'Tasks Closed Controller', 'Email Listener', 'lightning', 'Delegated Approver', 
          'time based workflow', 'report', 'FIELD_CUSTOM_VALIDATION_EXCEPTION', 'Task Template', 'System.DmlException:',
          'Primary Client','SSO']           
        stopwords_EN  = set(stopwords.words('english'))
        cleaned_text = []
        clean = []
        url = 'http, www.'
        spl_char = '\', --, -, [, ], \n, (, ), \, ,?' #!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
        module = []
        for sentence in df.Description:
            if sentence != None:
                temp_mod = set()
                #sentence = str(sentence.encode("latin-1"), "windows-1252")
                print('sentence--->',sentence)
                str2 = re.findall('.*Error:.*',sentence)
                if 'Steps to Reproduce' in sentence:
                    sentence = sentence.split('Steps to Reproduce')[0]
                if 'steps to reproduce' in sentence:
                    sentence = sentence.split('steps to reproduce')[0]
                if 'Steps to reproduce' in sentence:
                    sentence = sentence.split('Steps to reproduce')[0]
                #sentence = sentence.split('Steps to reproduce')[0]
                sentence = sentence.split('Apex Class:')[0]
                sentence = sentence.replace('Remedyforce Version:','')
                sentence = sentence.replace('Summary/Error of issue:','')
                sentence = sentence.replace('\'','')
                sentence = sentence.replace(',','')
                str4 = re.findall('(\w*00D\w*)|(\w*00d\w*)]', sentence)
                #print(str4, 'size ::', len(str4))
                if len(str4) > 0:
                    if str4[0][0] == '':
                        sentence = sentence.replace(str4[0][1], '')
                    elif str4[0][1] == '':
                        sentence = sentence.replace(str4[0][0], '')
                sentence = sentence + str(str2)
                #cleaned_text = re.sub(r'\d+', '', sentence)
                for item in modules:
                    item1 = ' ' +item.lower() + ' '
                    sentence1 = sentence.lower()
                    if item1 in sentence1:
                        temp_mod.add(item+'$$')
                cleaned_text = [word for word in sentence.split() if word not in punctuation]
                cleaned_text = [word for word in cleaned_text if word not in url]
                cleaned_text = [word for word in cleaned_text if word not in stopwords_EN]
                cleaned_text = [word for word in cleaned_text if word not in spl_char]
                
                module.append(' '.join(temp_mod))
                clean.append(' '.join(cleaned_text))
            else:
                module.append(' '.join(''))
                clean.append(' '.join(''))
        Case_Owner = []
        for item in df.Case_Owner__c:
            if item :
            print('item-->', item)
            Case_Owner.append(item)
        df['Module'] = module
        df['cleaned_text'] = clean
        df['Case_Owner'] = Case_Owner
        df.Module = df.Module.apply(lambda x:x.replace('$$ ',',').replace('$$',''))
        #print("K--->191",df)
        return df
    
class RetrieveCaseData:
    def retrieveData(self, sf, inputCaseId):
        print(inputCaseId)
        queryCase = sf.bulk.Case.query("Select id, CASENUMBER, Status, Origin, Case_Owner__c, SUBJECT, DESCRIPTION, CLOSEDDATE, SLASTARTDATE__c, SLAEXITDATE__c, CREATEDDATE, SERVICE_LEVEL__c, SEVERITY__c, CASE_AGGRAVATION__c, SC_LP_VERSION_CODE_HL__c, SLA_EXPIRATION__c, SLA_PRIORITIZATION__c, SC_CASESUMMARY__c, ACCOUNT_NAME__c, CUSTOMER_STATUS_AGE__c, PENDING_CLIENT_CUMULATIVE_TIME__c, PENDING_CUSTOMER_SUPPORT_CUMULATIVE_TIME__c, PENDING_ENGINEERING_CUMULATIVE_TIME__c, SUPPORT_AND_CUSTOMER_HANDOFFS_COUNT__c, SUPPORT_AND_ENGINERERING_HANDOFFS_COUNT__c, CASE_SUMMARY_LAST_UPDATED__c, ACTIVE_GCC__c, SUBMITTED_HOUR_GMT__c from Case order by LastModifiedDate desc limit 2000")
        
        ##queryCase = [item.replace("None", "") for item in queryCase]
        #print(queryCase)
        return queryCase
    
        
class PredictionModel:
    def getPrediction(self, df, inputCaseId):
        df['SLA_ExpirationDate__c'] = df.SLA_EXPIRATION__c.apply(lambda x: str(x).replace('T', ' ').replace('Z', '')[:-4])
        df['SLA_ExitDate__c'] = df.SLAEXITDATE__c.apply(lambda x: str(x).replace('T', ' ').replace('Z', '')[:-4])
        df['SLA_StartDate__c'] = df.SLASTARTDATE__c.apply(lambda x: str(x).replace('T', ' ').replace('Z', '')[:-4])
        df['SLA_ExpirationDate__c']= pd.to_datetime(df['SLA_ExpirationDate__c'])
        df['SLA_ExitDate__c']= pd.to_datetime(df['SLA_ExitDate__c'])
        df['SLA_StartDate__c']= pd.to_datetime(df['SLA_StartDate__c'])
        df['SLA_diff'] = df['SLA_ExpirationDate__c'] - df['SLA_ExitDate__c']
        df['SLA_diff'] = df['SLA_diff'].apply(lambda x : x.days)
        df = df[(df.SLA_PRIORITIZATION__c == 'Finished - Met') | (df.SLA_PRIORITIZATION__c == 'Finished - Missed') | (df.Id == inputCaseId)]
        df = pd.get_dummies(df, columns=['Status', 'Origin', 'SEVERITY__c', 'SUBMITTED_HOUR_GMT__c','ACTIVE_GCC__c','Module'])
        testData = df[df.Id == inputCaseId]
        print('inputCaseId--->',inputCaseId)
        testData = testData.drop(['SLA_PRIORITIZATION__c'], axis=1)
        df = df[df.Id != inputCaseId]
        le = preprocessing.LabelEncoder()
        df['SLA_PRIORITIZATION__c'] = le.fit_transform(df.SLA_PRIORITIZATION__c.values)
        df_majority = df[df.SLA_PRIORITIZATION__c==0]
        df_minority = df[df.SLA_PRIORITIZATION__c==1]
        print('df --->', df.shape)
        print('df_majority --->', df_majority.shape)
        print('df_minority --->', df_minority.shape)
        df_minority_upsampled = resample(df_minority, replace=True, n_samples=df_majority.shape[0], random_state=42)
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])
        df_exp = df_upsampled.drop(['CaseNumber','Subject', 'Description', 'SC_LP_VERSION_CODE_HL__c', 'SC_CASESUMMARY__c', 'ACCOUNT_NAME__c', 'ClosedDate', 'SLASTARTDATE__c', 'SLAEXITDATE__c', 'CreatedDate', 'SERVICE_LEVEL__c', 'SC_LP_VERSION_CODE_HL__c','SLA_EXPIRATION__c', 'CASE_SUMMARY_LAST_UPDATED__c', 'SLA_ExpirationDate__c', 'SLA_ExitDate__c','SLA_StartDate__c', 'CUSTOMER_STATUS_AGE__c','SUPPORT_AND_CUSTOMER_HANDOFFS_COUNT__c', 'SUPPORT_AND_ENGINERERING_HANDOFFS_COUNT__c','SLA_diff','attributes', 'Id', 'Case_Owner__c', 'cleaned_text'], axis=1)
        testData = testData.drop(['CaseNumber','Subject', 'Description', 'SC_LP_VERSION_CODE_HL__c', 'SC_CASESUMMARY__c', 'ACCOUNT_NAME__c', 'ClosedDate', 'SLASTARTDATE__c', 'SLAEXITDATE__c', 'CreatedDate', 'SERVICE_LEVEL__c', 'SC_LP_VERSION_CODE_HL__c','SLA_EXPIRATION__c', 'CASE_SUMMARY_LAST_UPDATED__c', 'SLA_ExpirationDate__c', 'SLA_ExitDate__c','SLA_StartDate__c', 'CUSTOMER_STATUS_AGE__c','SUPPORT_AND_CUSTOMER_HANDOFFS_COUNT__c', 'SUPPORT_AND_ENGINERERING_HANDOFFS_COUNT__c','SLA_diff','attributes', 'Id', 'Case_Owner__c', 'cleaned_text'], axis=1)
        y = df_exp['SLA_PRIORITIZATION__c']
        X = df_exp.drop(['SLA_PRIORITIZATION__c'], axis=1)
        print('X--->', X.columns)
        print('X--->', X.head())
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.2, random_state = 42, stratify=y)
        print('X_train-->',X_train)
        print('Y_train',Y_train)
        print('testDataCols--->', testData.columns)
        print('testData NULL ANY-->',testData.columns[testData.isna().any()].tolist())
        print('testData--->',testData)
        rf = RandomForestClassifier(oob_score=True)
        rf.fit(X_train, Y_train)
        pred_param = rf.predict_proba(testData)[0][0]
        print('pred_param--->',pred_param)
        return pred_param