# python操作mysql的简单练习


import pymysql

mydb = pymysql.connect(
    host = '127.0.0.1', user = 'root', passwd = 'peng930133',
    port = 3306, db = 'harshdb', charset = 'utf8'
)
mycursor = mydb.cursor()
# mycursor.execute("create database harshdb") #创建数据库

mycursor.execute("show databases")  # 查看服务器中建立的数据库
for db in mycursor:
    print(db)

# mycursor.execute("create table employee(name varchar(250),sal int(20))") #创建表employee

# sqlformula = "Insert into employee(name,sal) values(%s,%s)" # 写入数据
# employees = [("harshit",200000),("rahul", 30000),("avinash", 40000),("amit", 50000),]
# mycursor.executemany(sqlformula, employees)
# mydb.commit()

# sql = "Update employee SET sal = 70000 WHERE name = 'harshit'"  # 更新数据
# mycursor.execute(sql)
# mydb.commit()

# sql = "DELETE FROM employee WHERE name ='harshit'" # 删除数据
# mycursor.execute(sql)
# mydb.commit()

# sql = "show tables" # 查看所有表
# mycursor.execute(sql)
# for db in mycursor:
#     print(db)

# sql = "desc employee" # 查表employee结构
# mycursor.execute(sql)
# for db in mycursor:
#     print(db)

mydb.close()
