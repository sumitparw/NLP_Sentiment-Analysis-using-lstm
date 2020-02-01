from pyspark import SparkContext
import json
sc = SparkContext()

joinedrdd = sc.textFile('output.csv')



for year in ['2019','2018','2017','2016','2015','2014','2013','2012','2011']:
        year = joinedrdd.filter(lambda x: x[4].split('-')[0] == year)
        print("Year= "+year+" : "+year.count())