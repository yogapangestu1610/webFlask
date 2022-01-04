from app import db, User

username = "Admin"
password = "Admin123"
level = "Admin"
db.session.add(User(username,password,level))
db.session.commit()
