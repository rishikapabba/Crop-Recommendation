import requests
from bs4 import BeautifulSoup
import pandas as pd

paddy=0
maize=0
cotton=0
tur=0
moong=0
urad=0
jute=0
coconut=0

URL = 'https://farmer.gov.in/mspstatements.aspx'
page = requests.get(URL)
#print(page.content)
soup = BeautifulSoup(page.content, 'html.parser')
table=soup.find("table", {"id": "tablebottom"})
data = []

rows = table.find_all('tr')
for row in rows:
	cols = row.find_all('td')
	cols = [ele.text.strip() for ele in cols]
	data.append([ele for ele in cols if ele])
no=len(data[0])
for rows in data:
	if(len(rows)>0):
		#print(rows)
		if(rows[0]=='PADDY'):
			if(rows[1]=='Common'):
				paddy=rows[len(rows)-1]
		elif(rows[0]=='MAIZE'):
			maize=rows[len(rows)-1]
		elif('Tur' in rows[0]):
			tur=rows[len(rows)-1]
		elif(rows[0]=='URAD'):
			urad=rows[len(rows)-1]
			print(urad)
		elif(rows[0]=='COTTON' and rows[1]=='Medium Staple'):
			cotton=rows[len(rows)-1]
		elif('DE-HUSKED COCONUT' in rows[0]):
			coconut=rows[len(rows)-1]
		elif(rows[0]=='JUTE'):
			jute=rows[len(rows)-1]

#print(paddy,maize,coconut)

