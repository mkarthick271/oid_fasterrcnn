\c oid;
create table datasetbbox250 as select b.imageid, b.labeldesc, b.xmin1, b.xmax1, b.ymin1, b.ymax1 from (select distinct imageid from dataset250) a inner join ctrnimgbboxcls b on a.imageid = b.imageid;
