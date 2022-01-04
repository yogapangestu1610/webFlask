from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, current_app, make_response
from sqlalchemy import func, or_
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SelectField
from wtforms.validators import InputRequired
from flask_bootstrap import Bootstrap
from functools import wraps
from flask_migrate import Migrate
import datetime
import pdfkit
from fraud import app, db, bcrypt, bootstrap, migrate
from .models import *

# Library for chatbot
import nltk
import os
#nltk.download('popular')
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import dtype
from tensorflow.keras.models import load_model
import json
import random

# library for detection
import sys
import cv2
import dlib
import logging
from math import hypot
from keras.models import load_model

class Login(FlaskForm):
    username = StringField('', validators=[InputRequired()], render_kw={'autofocus':True, 'placeholder':'Username'})
    password = PasswordField('', validators=[InputRequired()], render_kw={'autofocus':True, 'placeholder':'Password'})
    level = SelectField('', validators=[InputRequired()], choices=[('Admin','Admin'), ('Pengawas','Pengawas'),
                                                                   ('Peserta','Peserta')])

def login_dulu(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'login' in session:
            return f(*args, **kwargs)
        else:
            return redirect(url_for('login'))
    return wrap

@app.route('/')
def index():
    if session.get('login') == True:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if session.get('login') == True:
        return redirect(url_for('dashboard'))
    form = Login()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data) and user.level == form.level.data:
                session['login'] = True
                session['id'] = user.id
                session['level'] = user.level
                session['user'] = user.username
                return redirect(url_for('dashboard'))
        pesan = "Username atau Password anda salah"
        return render_template("login.html", pesan=pesan, form=form)
    return render_template('login.html', form=form)





# DASHBOARD

@app.route('/dashboard')
@login_dulu
def dashboard():
    data1 = db.session.query(Dokter).count()
    data2 = db.session.query(Pendaftaran).count()
    data3 = db.session.query(User).count()
    data4 = db.session.query(func.sum(Obat.harga_jual)).filter(Obat.kondisi == "Rusak").scalar()
    data5 = db.session.query(func.sum(Obat.harga_jual)).filter(Obat.kondisi == "Baik").scalar()
    return render_template('dashboard.html',data1=data1,data2=data2,data3=data3,data4=data4,data5=data5)

@app.route('/kelola_user')
@login_dulu
def kelola_user():
    data = User.query.all()
    return render_template('user.html', data=data)

@app.route('/tambahuser', methods=['GET','POST'])
@login_dulu
def tambahuser():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        level = request.form['level']
        db.session.add(User(username,password,level))
        db.session.commit()
        return redirect(url_for('kelola_user'))

@app.route('/edituser/<id>', methods=['GET','POST'])
@login_dulu
def edituser(id):
    data = User.query.filter_by(id=id).first()
    if request.method == "POST":
        try:
            data.username = request.form['username']
            if data.password != '':
                data.password = bcrypt.generate_password_hash(request.form['password']).decode('UTF-8')
            data.level = request.form['level']
            db.session.add(data)
            db.session.commit()
            return redirect(url_for('kelola_user'))
        except:
            flash("Ada trouble")
            return redirect(request.referrer)

@app.route('/hapususer/<id>', methods=['GET','POST'])
@login_dulu
def hapususer(id):
    data = User.query.filter_by(id=id).first()
    db.session.delete(data)
    db.session.commit()
    return redirect(url_for('kelola_user'))




# PENDAFTARAN => PESERTA

@app.route('/pendaftaran')
@login_dulu
def pendaftaran():
    data = Pendaftaran.query.all()
    return render_template('pendaftaran.html', data=data)

@app.route('/tambahdaftar', methods=['GET','POST'])
@login_dulu
def tambahdaftar():
    if request.method == "POST":
        nama = request.form['nama']
        tl = request.form['tl']
        tg_lahir = request.form['tg_lahir']
        jk = request.form['jk']
        status = request.form['status']
        profesi = request.form['profesi']
        alamat = request.form['alamat']
        keterangan = request.form['keterangan']
        db.session.add(Pendaftaran(nama,tl,tg_lahir,jk,status,profesi,alamat,keterangan))
        db.session.commit()
        return jsonify({'success':True})

@app.route('/editdaftar/<id>', methods=['GET','POST'])
@login_dulu
def editdaftar(id):
    data = Pendaftaran.query.filter_by(id=id).first()
    if request.method == "POST":
        data.nama = request.form['nama']
        data.tl = request.form['tl']
        data.tg_lahir = request.form['tg_lahir']
        data.jk = request.form['jk']
        data.status = request.form['status']
        data.profesi = request.form['profesi']
        data.alamat = request.form['alamat']
        data.keterangan = request.form['keterangan']
        db.session.add(data)
        db.session.commit()
        return redirect(url_for('pendaftaran'))


# PESERTA UNTUK USER KE 3

@app.route('/peserta3')
@login_dulu
def peserta3():
    data = Pendaftaran.query.all()
    return render_template('peserta3.html', data=data)



# APOTIK => UJIAN

@app.route('/apotik')
@login_dulu
def apotik():
    data = Obat.query.all()
    data1 = Suplier.query.all()
    return render_template('apotik.html', data=data,data1=data1)

@app.route('/tambahobat', methods=['GET','POST'])
@login_dulu
def tambahobat():
    if request.method == "POST":
        namaObat = request.form['namaObat']
        jenisObat = request.form['jenisObat']
        harga_beli = request.form['harga_beli']
        harga_jual = request.form['harga_jual']
        kondisi = request.form['kondisi']
        suplier_id = request.form['suplier_id']
        db.session.add(Obat(namaObat,jenisObat,harga_beli,harga_jual,kondisi,suplier_id))
        db.session.commit()
        return jsonify({'success':True})

@app.route('/editobat/<id>', methods=['GET','POST'])
@login_dulu
def editobat(id):
    data = Obat.query.filter_by(id=id).first()
    if request.method == "POST":
        data.namaObat = request.form['namaObat']
        data.jenisObat = request.form['jenisObat']
        data.harga_beli = request.form['harga_beli']
        data.harga_jual = request.form['harga_jual']
        data.kondisi = request.form['kondisi']
        data.suplier_id = request.form['suplier_id']
        db.session.add(data)
        db.session.commit()
        return redirect('/apotik')



## UJIAN UNTUK USER KE 3

@app.route('/ujian3')
@login_dulu
def ujian3():
    data = Obat.query.all()
    data1 = Suplier.query.all()
    return render_template('ujian3.html', data=data,data1=data1)




# DOKTER => JADWAL

@app.route('/dokter')
@login_dulu
def dokter():
    data = Dokter.query.all()
    return render_template('dokter.html', data=data)

@app.route('/tambahdokter', methods=['GET','POST'])
@login_dulu
def tambahdokter():
    if request.method == 'POST':
        nama = request.form['nama']
        jadwal = request.form['jadwal']
        db.session.add(Dokter(nama,jadwal))
        db.session.commit()
        return jsonify({'success':True})
    else:
        return redirect(request.referrer)

@app.route('/editdkt/<id>', methods=['GET','POST'])
@login_dulu
def editdkt(id):
    data = Dokter.query.filter_by(id=id).first()
    if request.method == 'POST':
        data.nama = request.form['nama']
        data.jadwal = request.form['jadwal']
        db.session.add(data)
        db.session.commit()
        return redirect(url_for('dokter'))

@app.route('/hapusdokter/<id>', methods=['GET','POST'])
@login_dulu
def hapusdokter(id):
    data = Dokter.query.filter_by(id=id).first()
    db.session.delete(data)
    db.session.commit()
    return redirect(request.referrer)




# SUPLIER => PENGAWAS

@app.route('/suplier')
@login_dulu
def suplier():
    data = Suplier.query.all()
    return render_template('suplier.html', data=data)

@app.route('/tambahsuplier', methods=['GET','POST'])
@login_dulu
def tambahsuplier():
    if request.method == "POST":
        perusahaan = request.form['perusahaan']
        kontak = request.form['kontak']
        alamat = request.form['alamat']
        db.session.add(Suplier(perusahaan,kontak,alamat))
        db.session.commit()
        return jsonify({'success':True})

@app.route('/editsuplier/<id>', methods=['GET','POST'])
@login_dulu
def editsuplier(id):
    data = Suplier.query.filter_by(id=id).first()
    if request.method == "POST":
        data.perusahaan = request.form['perusahaan']
        data.kontak = request.form['kontak']
        data.alamat = request.form['alamat']
        db.session.add(data)
        db.session.commit()
        return redirect(url_for('suplier'))

@app.route('/hapusSuplier', methods=['GET','POST'])
@login_dulu
def hapusSuplier(id):
    data = Suplier.query.filter_by(id=id).first()
    db.session.delete(data)
    db.session.commit()
    return redirect(request.referrer)

@app.route('/tangani_pasien')
@login_dulu
def tangani_pasien():
    data = Pendaftaran.query.filter_by(keterangan="Diproses").all()
    return render_template('tangani.html',data=data)

@app.route('/diagnosis/<id>', methods=['GET','POST'])
@login_dulu
def diagnosis(id):
    data = Pendaftaran.query.filter_by(id=id).first()
    if request.method == "POST":
        nama = request.form['nama']
        keluhan = request.form['keluhan']
        diagnosa = request.form['diagnosa']
        resep = request.form['resep']
        user_id = request.form['user_id']
        pendaftaran_id = request.form['pendaftaran_id']
        tanggal = datetime.datetime.now().strftime("%d %B %Y Jam %H:%M:%S")
        data.keterangan = "Selesai"
        db.session.add(data)
        db.session.commit()
        db.session.add(Pasien(nama,keluhan,diagnosa,resep,user_id,pendaftaran_id,tanggal))
        db.session.commit()
        return redirect(request.referrer)





# PENCARIAN

@app.route('/pencarian')
@login_dulu
def pencarian():
    return render_template('pencarian.html')

@app.route('/cari_data', methods=['GET','POST'])
@login_dulu
def cari_data():
    if request.method == 'POST':
        keyword = request.form['q']
        formt = "%{0}%".format(keyword)
        datanya = Pasien.query.join(User, Pasien.user_id == User.id).filter(or_(Pasien.tanggal.like(formt))).all()
        if datanya:
            flash("Data berhasil di temukan")
            tombol = "tombol"
        elif not datanya:
            pesan = "Data tidak berhasil di temukan"
            return render_template('pencarian.html',datanya=datanya,pesan=pesan)
        return render_template('pencarian.html',datanya=datanya,tombol=tombol,keyword=keyword)

@app.route('/cetak_pdf/<keyword>', methods=['GET','POST'])
@login_dulu
def cetak_pdf(keyword):
    formt = "%{0}%".format(keyword)
    datanya = Pasien.query.join(User, Pasien.user_id == User.id).filter(or_(Pasien.tanggal.like(formt))).all()
    html = render_template("pdf.html",datanya=datanya)
    config = pdfkit.configuration(wkhtmltopdf="C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe")
    pdf = pdfkit.from_string(html, False, configuration=config)
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'inline; filename=laporan.pdf'
    return response





# LOGOUT

@app.route('/logout')
@login_dulu
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404




# ABOUT

@app.route('/aboutme')
@login_dulu
def aboutme():
    return render_template('aboutme.html')





# CHATBOT

import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('fraud/model/model.h5')
import json
import random
intents = json.loads(open('fraud/model/data.json').read())
words = pickle.load(open('fraud/model/texts.pkl','rb'))
classes = pickle.load(open('fraud/model/labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

@app.route("/get")
@login_dulu
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

@app.route('/bot')
@login_dulu
def bot():
    return render_template('bot.html')
   



# DETECTION

count = []

@app.route('/deteksi', methods=['GET','POST'])
@login_dulu
def deteksi():
    # Logging for face detection
    def setup_custom_logger():
        LOG_DIR = os.getcwd() + '/' + 'Log-Activity'
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
        handler = logging.FileHandler(LOG_DIR+'/log'+str(len(LOG_DIR))+'.txt', mode='w')
        handler.setFormatter(formatter)
        screen_handler = logging.StreamHandler(stream=sys.stdout)
        screen_handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.addHandler(screen_handler)
        return logger
    
    # Detection
    kameraVideo = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r'.\fraud\backend\shape_predictor_68_face_landmarks.dat')
    model_F = load_model(r'.\fraud\backend\model_motion01.h5')

    # Buat warnain labelnya
    labels_dict_F  = {0:'Mengangkat Kedua Alis', 1:'Lirik Kanan', 2:'Lirik Kiri',
                3:'Normal', 4:'Lihat Atas', 5:'Lihat Bawah', 6:'Lihat Kanan',
                7:'Lihat Kiri', 8:'unknown'}
    color_dict_F = {0:(255,204,255), 1:(0,0,128), 2:(211,85,186), 3:(0,255,0),
                4:(0,155,255), 5:(255,255,0), 6:(255,0,0), 7:(128,128,0), 8:(135,184,222)}

    # Deteksi
    index = 0
    logger = setup_custom_logger()

    # inisialisasi untuk get count tiap gerakan
    mengangkat_alis = 0
    lirik_kanan = 0
    lirik_kiri = 0
    normal = 0
    lihat_atas = 0
    lihat_bawah = 0
    lihat_kanan = 0
    lihat_kiri = 0
    no_face = 0
    miss = 0

    while (True):
        try:
            #ambil per KERANGKA UNTUK DIPROSES
            ret, kerangkaAsal = kameraVideo.read()

            #siap2 ngesave dgn judul berdasarkan counter index
            if not ret: 
                break

            dets = detector(kerangkaAsal)

            num_faces = len(dets)
            #print("banyaknya wajah:",num_faces)

            if num_faces == 0:
                print("Sorry, there were no faces found")
                logger.debug('Face Not Found')
                cv2.putText(kerangkaAsal, "NO FACES", (200,240), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 5)
                miss += 1

            # Find the 5 face landmarks we need to do the alignment.
            faces = dlib.full_object_detections()

            for detection in dets:
                faces.append(predictor(kerangkaAsal, detection))

            images = dlib.get_face_chips(kerangkaAsal, faces, size=320)

            for image in images:
                #konversi ke skala abu-abu
                kerangkaAbu = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

            #bila index habis dibagi 5 (in case mau diambil tiap frame ke 5)
            if index%5==0:
                #isi dengan kode apa yg mau dilakukan saat mendeteksi frame ke 5
                
                resized_F = cv2.resize(kerangkaAbu,(100,100))
                normalized_F = resized_F/255.0
                reshaped_F = np.reshape(normalized_F,(1,100,100,1))
                result_F = model_F.predict(reshaped_F)
                                    
            index += 1
            label_F = np.argmax(result_F,axis=1)[0]
            #cv2.imshow('REGION FACE',image)
            #print(labels_dict_F[label_F])
            logger.info(labels_dict_F[label_F])
            
            #untuk tampilan saja
            if (label_F != 0) & (label_F != 1):        
                cv2.rectangle(kerangkaAsal,(10,400),(200,450),color_dict_F[label_F],-1)
                cv2.putText(kerangkaAsal, labels_dict_F[label_F], (20,430),cv2.FONT_HERSHEY_SIMPLEX,0.8, (255,255,255),2)
            else: 
            #ukuran latar untuk mengangkat kedua alis
                cv2.rectangle(kerangkaAsal,(10,400),(350,450),color_dict_F[label_F],-1)
                cv2.putText(kerangkaAsal, labels_dict_F[label_F], (10,430),cv2.FONT_HERSHEY_SIMPLEX,0.8, (255,255,255),2)

            # untuk get count tiap label
            if(label_F == 0):
                mengangkat_alis +=1
            elif(label_F == 1):
                lirik_kanan += 1
            elif(label_F == 2):
                lirik_kiri += 1
            elif(label_F == 3):
                normal += 1
            elif(label_F == 4):
                lihat_atas += 1
            elif(label_F == 5):
                lihat_bawah += 1
            elif(label_F == 6):
                lihat_kanan += 1
            elif(label_F == 7):
                lihat_kiri += 1
            else :
                no_face += 1

        except RuntimeError as e:
            print (e)
        
        #Tampilkan
        cv2.imshow('Cheating Detection', kerangkaAsal)
        #cv2.imshow('Kerangka Asal', image)
            
        key=cv2.waitKey(1)

        if(key==27):
            print ("-------------------Program Selesai Digunakan--------------------")
            break
    
    cv2.destroyAllWindows()
    kameraVideo.release() 

    count.append(mengangkat_alis)
    count.append(lirik_kanan)
    count.append(lirik_kiri)
    count.append(normal)
    count.append(lihat_atas)
    count.append(lihat_bawah)
    count.append(lihat_kanan)
    count.append(lihat_kiri)
    count.append(no_face)
    count.append(miss)

    return render_template('pengawasan.html')

@app.route("/pengawasan")
@login_dulu
def pengawasan():
    return render_template('pengawasan.html')

@app.route("/hasilLaporan")
@login_dulu
def hasilLaporan():
    
    return render_template('laporan.html')

@app.route("/log")
@login_dulu
def log():
    with open("Log-Activity/log41.txt") as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return render_template('log.html', lines=lines)  