from pyspark import SparkContext, SQLContext
import json
import pandas as pd
sc = SparkContext()
sq = SQLContext(sc)
business= sc.textFile('business.json')
review= sc.textFile('review.json')
business= business.repartition(8)
review = review.repartition(8)

business = business.map(json.loads).filter(lambda x:x['state'] == 'IL').map(lambda x: (x['business_id'],(x['state'])))
print(business.count())
review = review.map(json.loads).map(lambda x: (x['business_id'], (x['stars'],x['text'],x['date'])))
print(review.first())

def toCSVLine(data):
  return ','.join(str(d) for d in data)

joinedrdd = business.join(review)
print(joinedrdd.count())
joinedrdd = joinedrdd.map(lambda x: (x[0],x[1][0],x[1][1][0],x[1][1][1],x[1][1][2]))

for yearlist in ['2018','2017','2016','2015','2014','2013','2012','2011','2010']:
        year = joinedrdd.filter(lambda x: x[4].split('-')[0] == yearlist)
        df = sq.createDataFrame(year.collect())
        df.toPandas().to_csv("Dataset/"+yearlist+".csv",index=False,header=None)




