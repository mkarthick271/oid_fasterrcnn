create database oid;
\c oid
create table ctrnimglabel (id serial primary key, imageid varchar(50), source varchar(50), labelname varchar(50), confidence varchar(10));
create table ctrnimgbbox (id serial primary key, imageid varchar(50), source varchar(50), labelname varchar(50), confidence varchar(10), xmin1 float8, xmax1 float8, ymin1 float8, ymax1 float8, isoccluded varchar(10), istruncated varchar(10), isgroupof varchar(10), isdepiction varchar(10), isinside varchar(10));
create table ctrnimgurl (id serial primary key, imageid varchar(50), subset varchar(50), url varchar(200), landurl varchar(500), license varchar(200), authorurl varchar(200), author varchar(200), origsize varchar(200), origmd5 varchar(200), thumbnail varchar(200), rotation float8);
create table labeldesc(id serial primary key, labelname varchar(50), labeldesc varchar(50));
COPY ctrnimglabel (imageid, source, labelname, confidence) FROM '/$HOME$/oid_fasterrcnn/data/train/challenge-2019-train-detection-human-imagelabels.csv' DELIMITER ',' CSV HEADER;
COPY ctrnimgbbox (imageid, source, labelname, confidence, xmin1, xmax1, ymin1, ymax1, isoccluded, istruncated, isgroupof, isdepiction, isinside) FROM '/$HOME$/oid_fasterrcnn/data/train/challenge-2019-train-detection-bbox.csv' DELIMITER ',' CSV HEADER;
COPY ctrnimgurl (imageid , subset, url , landurl , license , authorurl, author , origsize , origmd5 , thumbnail , rotation) FROM '/home/mkarthick2714/train/train-images-boxable-with-rotation.csv' DELIMITER ',' CSV HEADER;
COPY labeldesc (labelname, labeldesc) FROM '/$HOME$/oid_fasterrcnn/data/train/challenge-2019-classes-description-500.csv' DELIMITER ',' CSV HEADER;
create table ctrnimgbboxcls as select a.id, a.imageid, a.labelname, b.labeldesc, a.confidence, a.xmin1, a.xmax1, a.ymin1, a.ymax1, a.isoccluded, a.istruncated, a.isgroupof, a.isdepiction, a.isinside from ctrnimgbbox a left join labeldesc b on a.labelname = b.labelname; 

