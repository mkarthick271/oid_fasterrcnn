import psycopg2
import csv

connection = psycopg2.connect(
    user='postgres', password='cts-0000', host='127.0.0.1', port='5432', database='oid')

cursor = connection.cursor()
# Print PostgreSQL Connection properties
print(connection.get_dsn_parameters(), "\n")

getclslabels = "select labeldesc from labeldesc"
cursor.execute(getclslabels)
labeldesc = cursor.fetchall()

filename = "dataset250.csv"
f = open(filename, "w+")
f.close()

urls = []

for i in range(len(labeldesc)):
    print("{} Extracting class {}".format(i+1, labeldesc[i]))
    postgreSQL_select_Query = "select imageid, labeldesc, url  from ctrnimgclsurl where labeldesc=%s and confidence='1' order by random() limit 250"
    cursor.execute(postgreSQL_select_Query, (labeldesc[i],))
    url = cursor.fetchall()
    for item in url:
        urls.append(item)
    url.clear()

with open('dataset250.csv', 'w') as out:
    csv_out = csv.writer(out)
    csv_out.writerow(['id', 'imageid', 'labeldesc', 'url'])
    for i, row in enumerate(urls):
#    for row in urls:
        temp = list(row)
        temp.insert(0, int(i+1))
        row = tuple(temp)
#        print(row)
        csv_out.writerow(row)
    out.close()

cursor.execute("delete from dataset250")
connection.commit()

copy_sql = "copy dataset250 from stdin with csv header delimiter as ','"
with open('dataset250.csv', 'r') as out:
#    cursor.copy_from(out, 'dataset250', sep=',')
    cursor.copy_expert(sql=copy_sql, file=out)
    connection.commit()

