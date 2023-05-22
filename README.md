# Laporan Proyek Akhir Machine Learning Terapan - Putri Sinta Dewi Sinaga
# "Machine Learning Terapan - Sistem Rekomendasi Buku"

<br>

<p align='center'>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />        
  </a>
  <a href="https://github.com/PutriSintaDewiSinaga/">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" />        
  </a>
  <a href="https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" />
  </a
   <a href="https://colab.research.google.com/">
    <img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252" />
  </a
</p>

<br>

<div align="center">

| Profile       |                                                          |
| ------------- | -------------------------------------------------------- |
| Nama          |  Putri Sinta Dewi Sinaga                                 |
| Learning Path | Machine Learning Terapan                                 |
| Progam        | Baparekraf Digital Talent (BDT) Challenge 2023           |
| Submission    | Proyek Akhir : Membuat Model Sistem Rekomendasi          |

</div>

<p align='center'>
    <img src ="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/1.png?raw=true" alt="Buku">
</p>

    
    

## Domain Proyek

Domain yang akan dibahas pada proyek akhir machine learning (*"Model Sistem Rekomendasi"*) ini adalah **Buku** dengan judul **"Sistem Rekomendasi Buku teratas"**.

### Latar Belakang

Buku adalah kumpulan/himpunan kertas atau lembaran yang tertulis atau mengandung tulisan. Bahan-bahan tersebut bisa berbentuk potongan yang terbuat dari kayu, kertas bahkan gading gajah. Kumpulan ini dihimpun atau dijilid menjadi satu pada salah satu ujungnya dan berisi tulisan, gambar atau tempelan. Setiap sisi dari sebuah lembaran kertas pada buku disebut sebuah halaman. Seiring dengan perkembangan dalam bidang dunia informatika, kini dikenal pula istilah e-book atau buku-e (buku elektronik) yang mengandalkan perangkat seperti komputer meja, komputer jinjing, komputer tablet, telepon seluler dan lainnya, serta menggunakan perangkat lunak tertentu untuk membacanya. Beberapa contoh buku:
1.  Novel
2.  Majalah
3.  Kamus
4.  Komik
5.  Manga
6.  Ensiklopedia
7.  Kitab suci
8.  Biografi
9.  Naskah
10.  Light novel (novel ringan)
11.  Buku tulis
12.  Buku gambar
13.  Nomik
14.  Cergam
15.  Antologi
16.  Novelet
17.  Fotografi
18.  Karya Ilmiah
19.  Atlas
20.  Babad

Banyaknya jumlah buku membuat pembaca terkadang kesulitan dalam menentukan buku yang hendak mereka baca selanjutnya.Terkadang dijumpai pembaca yang hanya ingin membaca buku-buku yang dengan reputasi penjualan terbaik. Ada pula pembaca yang hanya ingin membaca buku yang mirip dengan buku-buku yang pernah dibaca sebelumnya. Tidak jarang juga ditemui pembaca yang menentukan buku-buku yang akan dibaca selanjutnya berdasarkan rating dari buku-buku yang telah dilihatnya. Semakin tinggi rating dari buku tersebut, semakin tertarik pula pembaca untuk membacanya. Semakin rendah rating dari buku tersebut, maka pembaca cenderung enggan untuk membacanya. Tinggi rendahnya rating tersebut mempengaruhi buku-buku yang akan direkomendasikan. Nilai kemiripan antar buku dan rating buku dapat dijadikan landasan untuk memberikan rekomendasi buku kepada pembaca.

Oleh karena itu, Sistem rekomendasi memberikan solusi terhadap permasalahan dalam menentukan buku yang belum pernah dibaca oleh pengguna. Sehingga, pada proyek ini saya membuat rekomendasi buku yang ditujukan untuk merekomendasikan pengguna dalam memilih buku yang ingin dibaca. Pada dataset ini saya menggunakan data ‘rating’ dimana berisi informasi dari rating buku dan ‘Book’ yang berisi data-data buku. Pada rating ini terdiri dari beberapa penilaian pengguna terhadap salah satu buku dimana beberapa buku memiliki banyak penilaian rating dan beberapa buku memiliki sedikit penilaian rating oleh pengguna.

Referensi:

[1]  [Buku](https://id.wikipedia.org/wiki/Buku)
    
[2]  [SISTEM REKOMENDASI BUKU MENGGUNAKAN METODE ITEM-BASED COLLABORATIVE FILTERING](http://eprints.undip.ac.id/65823/1/laporan_24010311130044_1.pdf) 
    
[3]  [Metode Penelitian Sosial](http://repositori.uin-alauddin.ac.id/23278/1/Buku_Metode%20Penelitian%20Sosial%20survey.pdf)

## Business Understanding

### Problem Statements

Berdasarkan uraian yang telah dipaparkan pada latar belakang diatas, maka dapat diambil sebuah rumusan masalah yang dirumuskan sebagai berikut:
- bagaimana menerapkan metode item-based collaborative filtering sebagai pendekatan dari sistem rekomendasi?
- bagaimana proses untuk menentukan rekomendasi buku dengan menggunakan metode item-based collaborative filtering?
- Bagaimana evaluasi model machine learning dapat menentukan rekomendasi buku?

### Goals

Berdasarkan rumusan masalah yang telah dipaparkan di atas, maka proyek penelitian ini memiliki tujuan, yaitu:
- Membuat sistem untuk merekomendasikan buku kepada pengguna.
- Membuat sistem rekomendasi yang sesuai dengan preferensi pengguna dengan teknik Collaborative Filtering.
- Mengetahui hasil evaluasi model machine learning dalam menentukan rekomendasi buku.

### Solution statements

Untuk menyelesaikan permasalahan di atas, kita akan menggunakan dua algoritma sistem rekomendasi sebagai solusi permasalahan, yakni Content-Based Filtering dan yang kedua yakni Collaborative Filtering.

-  Content Based Filtering. Algoritma ini akan merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu. Kelebihan dari algoritma ini yakni semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi. Kekurangannya yakni Sistem hanya akan menunjukkan item yang nilainya tinggi untuk dicocokkan dengan profil pengguna, maka pengguna akan selalu menemukan item serupa dengan yang sudah direkomendasikan sebelumnya.
Collaborative Filtering. 
    
-  Collaborative filtering bergantung pada pendapat komunitas pengguna. Ia tidak memerlukan atribut untuk setiap itemnya seperti pada sistem berbasis konten. Kelebihan dari teknik ini yakni dapat membantu pengguna menemukan minat baru. Kekurangannya yakni tidak dapat menangani item baru/fresh. Jadi, jika item tidak terlihat selama pelatihan, sistem tidak dapat melakukan proses embedding untuk item tersebut dan tidak dapat mengkueri model dengan item ini.

## Data Understanding
Dataset yang dipakai dalam proyek machine learning ini merupakan Book Recommendation Dataset dengan 271360 records data Books, 278858 records data Users dan 1149780 records data Ratings. Dataset ini bersifat open-source yang dipubilkasikan oleh MÖBIUS melalui platform [Kaggle: Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). Topik dari datasetnya adalah Books yang berformat csv, Ratings yang berformat csv, dan User yang berformat csv (comma-separated values) dengan ukuran 107 MB.

<p align='center'>
    <img src="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/ss.png?raw=true" alt="datasets">
</p>

    
-  Dataset yang pertama yakni 'Users' yang memiliki jumlah 278858 data dan 3 kolom, yakni :
    
   i.   Kolom 'User-ID', berisi ID pengguna dari toko buku online
    
   ii.  Kolom 'Location', berisi lokasi pengguna.
    
   iii. Kolom 'Age', berisi usia pengguna.

-  Dataset yang kedua yakni 'Books' yang memiliki jumlah 278858 data dan memiliki 8 kolom, diantaranya :
    
   i.    Kolom 'ISBN', merupakan identifikasi dari masing-masing buku.
    
   ii.   Kolom 'Book-Title', merupakan judul buku.
    
   iii.  Kolom 'Book-Author', merupakan penulis buku.
    
   iv.   Kolom 'Year-Of-Publication', merupakan tahun dipublikasikannya buku.
    
   v.    Kolom 'Publisher', merupakan penerbit buku.
    
   vi.   Kolom 'Image-URL-S', marupakan URL gambar cover buku dalam ukuran S(Small)
    
   vii.  Kolom 'Image-URL-M', marupakan URL gambar cover buku dalam ukuran M(Medium)
    
   viii. Kolom 'Image-URL-L', marupakan URL gambar cover buku dalam ukuran L(Large)
    

-  Dataset yang kedua yakni 'Ratings' yang memiliki jumlah 1149780 data dan memiliki 3 kolom, berikut penjelasan mengenai kolom-kolomnya :
    
   i.   Kolom 'User-ID', yang berisi ID dari user yang memberikan rating terhadap buku.
    
   ii.  Kolom 'ISBN', merupakan identifikasi buku atau nomor buku yang diberi rating oleh user
    
   iii. Kolom 'Book-Rating', berisi nilai Rating dari buku, skala yang ada dalam rating ini yakni dari 0-10.
    
    
Dalam tahap ini saya melakukan Data loading dan proses EDA(Exploratory Data Analysis) yang menjelaskan variabel-variabel yang sudah dijelaskan sebelumnya. Dikarenakan ketiga dataset yang kita gunakan merupakan dataset dalam jumlah yang banyak yakni lebih dari 200.000, maka di pada proses ini saya hanya mengambil 12.000 data pertama dari setiap variabel di atas dalam pembuatan sistem rekomendasi ini.

### Visualization & Analysis

- **Univariate Analysis**

<p align='center'>
    <img src ="https://github.com/PutriSintaDewiSinaga/Putrisds/blob/main/Univariate%20Analysis.png?raw=true" height=auto alt="pie-chart">
</p>

Berdasarkan visualiasi pie-chart diatas, dapat disimpulkan bahwa pengidap Diabetes lebih sedikit daripada orang yang tidak mengidap anemia yaitu sebesar 9.0%. Namun, perlu diketahui bahwa angka ini cukuplah besar mengingat hal ini berkaitan langsung dengan kondisi kesehatan.

- **Bivariate Analysis**

<p align='center'>
    <img src="https://github.com/PutriSintaDewiSinaga/Putrisds/blob/main/BMI.png?raw=true" height=auto alt="BMI">
</p>

Dengan menggunakan boxplot, distribusi data dapat terlihat dengan jelas terkait korelasi antara gender dan Body Mass Index. Dari gambar di atas, didapatkan adalah orang yang terkena Diabetes, memiliki kecenderungan Body Mass Index lebih tinggi. Jadi, tingginya Body Mass Index dapat memberikan indikasi kalau terkena penyakit Diabetes.

<p align='center'>
      <img src="https://github.com/PutriSintaDewiSinaga/Putrisds/blob/main/age.png?raw=true" height=200px alt="age.Hba1c_Level.blood_glucose_level">
    <img src="https://github.com/PutriSintaDewiSinaga/Putrisds/blob/main/Hba1c_Level.png?raw=true" height=200px alt="age.Hba1c_Level.blood_glucose_level">
   <img src="https://github.com/PutriSintaDewiSinaga/Putrisds/blob/main/blood_glucose_level.png?raw=true" height=200px alt="age.Hba1c_Level.blood_glucose_level">
</p>

Pada feature age, Hba1c_Level dan blood_glucose_level, tidak ada perbedaan yang secara signifikan. 

**Multivariate Analysis**

<p align='center'>
    <img src="https://github.com/PutriSintaDewiSinaga/Putrisds/blob/main/all.png?raw=true" height=auto alt="pairplot">
</p>

Pada multivariate analysis melalui pairplot, dapat terlihat dengan jelas bahwa penyakit Diabetes dipengaruhi oleh age, Body Mass Index dan Hba1c_Level.

**Outlier & Distribution Analysis**

<p align='center'>
    <img src="https://github.com/PutriSintaDewiSinaga/Putrisds/blob/main/AUB.png?raw=true" height=auto alt="boxplots-outlier">
</p>

<p align='center'>
    <img src="https://github.com/PutriSintaDewiSinaga/Putrisds/blob/main/AUH.png?raw=true" height=auto alt="hist">
</p>

Visualiasi boxplot dapat membantu untuk mengidentifikasi ada tidaknya outlier data pada masing-masing feature.

## Data Preparation
Teknik yang digunakan dalam Data Preparation adalah sebagai berikut:

- One-Hot Encoding

  Ini merupakan metode yang sangat populer digunakan untuk mengubah data kategorikal ke data numerical dengan nilai biner 0 atau 1. Proses encoding sangat diperlukan, agar data yang masuk ke dalam algoritma machine learning dapat bekerja dengan baik. Sebagian besar algoritma klasifikasi lebih mudah untuk memproses nilai numerical daripada kategorikal. Pada tahap ini, encoding dilakukan secara manual dengan membuat function untuk mengubah values data kategorikal ke numerical. Hal ini mudah untuk dilakukan secara manual karena saya telah memahami dan mengetahui values dari feature nya. Feature yang perlu untuk dilakukan encoding adalah `Gender` dan `Result`. 

- Data Splitting

  Proses ini merupakan tahap untuk membagi dataset menjadi data train dan test. Pembagian ini bertujuan agar data yang digunakan dapat digunakan untuk mengembangkan model dan mengevaluasi performance dari model yang sudah dikembangkan. Pada proyek ini, dataset akan di split dengan proporsi 80% atau 75306 data untuk data train dan 20% atau 18827 data untuk data testing. Sedangkan, proses splitting akan menggunakan function train_test_split() yang tersedia pada library sklearn.

- Feature Scaling (Standarisasi)

  Scaling bertujuan untuk menormalisasikan range pada fitur - fitur data agar seluruh fitur berada pada range yang sama. Apabila model machine learning tidak melakukan proses feature scaling, maka hasil prediksi akan lebih condong atau didominasi oleh fitur yang memiliki values terbesar, sementara fitur dengan values yang kecil, memiliki peluang yang kecil untuk mempengaruhi hasil prediksi. Pada proyek ini, scaling data akan menggunakan metode standarisasi. karena secara umum distribusi data berada pada kondisi normal dan standarisasi lebih cocok untuk digunakan dalam case yang seperti ini. proses standarisasi akan dilakukan dengan memanfaatkan function StandardScaler() yang tersedia di dalam library sklearn. Proses standarisasi bekerja dengan mengurangkan data pada fitur dengan nilai rata-rata fitur (mean), yang kemudian dibagi dengan standar deviasi. proses ini akan mengasumsikan semua fiturnya terpusat disekitaran nol dan memiliki besaran variansi yang sama.

- Handling Imbalanced Class

  Proporsi dari label/class pada dataset yang tidak seimbang akan menjadi permasalahan yang cukup besar, khususnya pada algoritma klasifikasi. Hal ini dikarenakan, algoritma machine learning yang digunakan akan cenderung mengklasifikasikan data ke dalam class yang memiliki data yang lebih banyak atau dominan (majority class) daripada kelas yang lebih sedikit (minority class). Hal ini akan sangat berbahaya apabila terjadi, khususnya dalam bidang kesehatan. dimana kesalahan hasil prediksi bisa saja berakibat fatal bagi pasien. Pada dataset yang dipakai dalam proyek ini, terdapat imbalanced class. dimana Synthetic Minority Over-sampling Technique (SMOTE) akan dipakai untuk handling pada kasus ini. SMOTE memakai pendekatan oversampling, hal ini dilakukan dengan mensintesis sampel baru dari minority class untuk menyeimbangkan dataset dengan cara membuat instance baru dari *minority* *class*. Dengan metode ini dapat membuat data-set menjadi seimbang.
  
  <p align='center'>
       <img src="https://github.com/PutriSintaDewiSinaga/Putrisds/blob/main/PCB.png?raw=true" height=300px alt="knn">
    <img src="https://github.com/PutriSintaDewiSinaga/Putrisds/blob/main/PCA.png?raw=true" height=300px alt="knn">
  </p>

## Modeling
Pada proyek ini, algoritma *machine learning* yang dipakai adalah `K-Nearest Neighbor`, `Support Vector Machine`, `Logistic Regression` dan `Random Forest`.

- **K-Nearest Neighbor (KNN)**

  Algoritma KNN atau K-Nearest Neighbor merupakan salah satu algoritma paling sederhana dan populer digunakan dalam klasifikasi pada machine learning. KNN bekerja dengan mengambil sejumlah K-data untuk dijadikan acuan dalam menentukan class dari data yang baru. Setiap data akan dibandingkan berdasarkan jarak (similarity) antara satu data dengan data lainnya dengan memilih K tetangga terdekat. Proses modelling menggunakan KNN dalam proyek ini, akan memakai modul yang telah tersedia pada library scikit-learn yakni [KNeighborsClassifier()](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) dengan parameters `n_neighbors = 8` yang artinya akan ada 8 data acuan yang akan digunakan sebagai K tetangga terdekat dalam proses klasifikasi. kemudian, metrics yang digunakan dalam menentukan similarity adalah `minkowski distance`. cara kerja minkowski hampir mirip dengan euclidian distance, hanya saja yang membedakan adalah penambahan parameter p atau pangkatnya. minkowski menghitung jarak antar 2 vektor data. apabila nilai p=1 maka itu adalah manhattan, sedangan p=2 itu adalah euclidian distance. berikut formula dari minkowski distance. `d(x-y)=(∑i=1n|xi−yi|p)1/p.`

  <p align='center'>
      <img src="https://github.com/PutriSintaDewiSinaga/Putrisds/blob/main/KNN.png?raw=true" height=300px alt="knn">
    
  </p>

  Proses building model dalam KNN akan terus dilakukan iterasi secara berulang dengan mencari nilai K tetangga terdekat sampai mendapatkan hasil yang optimal. setelah model didapatkan, tahap selanjutnya akan dilakukan proses testing menggunakan data test yang telah disediakan.

  - Kelebihan:

    Algoritma KNN merupakan algoritma yang sederhana dan mudah untuk diimplementasikan.

    Dapat di implementasikan pada beberapa kasus seperti klasifikasi, regresi dan pencarian.

  - Kekurangan:

    Algoritma KNN menjadi lebih lambat secara signifikan seiring meningkatnya jumlah sampel dan/atau variabel independen.

- **Support Vector Machine (SVM)**

  Algoritma SVM bekerja untuk menemukan hyperplane atau pemisah yang dapat memaksimalkan jarak (margin) antar kelas dalam ruang n-dimensi untuk mengklasifikasikan titik-titik data. Memaksimalkan jarak margin akan memberikan kejelasan terkait klasifikasi kelas sehingga titik data yang baru dilihat dapat diklasifikasikan dengan lebih baik.

  <p align='center'>
      <img src ="https://github.com/PutriSintaDewiSinaga/Putrisds/blob/main/SVM.png?raw=true" alt="svm">
  </p>


  Pada tahap modelling, [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) yang dipakai menggunakan metode kernel dan menerima semua vektor input yang diberikan pada data training dengan menerapkan parameter `rbf` yang dipakai sebagai kernel tricks nya. Kernel ini dikenal memiliki performa yang baik dan hasil dari pelatihan memiliki nilai error yang kecil. fungsi kernel rbf adalah sebagai berikut `K(x,xi) = exp(-gamma \* sum((x – xi^2))`

  - Kelebihan

    Mampu bekerja dengan baik pada data yang relatif sedikit.

    Pengklasifikasian SVM dapat memberikan model dengan akurasi tinggi dan bekerja dengan baik dengan ruang dimensi tinggi.

  - Kekurangan

    Sulit diaplikasikan pada data yang sangat besar karena memiliki waktu pelatihan yang tinggi.

- **Random Forest**

  Random Forest merupakan salah satu algoritma machine learning terbaik yang digunakan dalam klasifikasi dalam jumlah data yang besar. Random Forest memakai pendekatan kombinasi dari beberapa pohon keputusan (decision tree) yang datanya akan dipilih secara random. Dalam random forest, penentuan klasifikasi dilakukan berdasarkan hasil voting dari tree yang terbentuk. sehingga, pemakaian jumlah tree yang lebih banyak dapat menghasilkan tingkat akurasi yang lebih optimal. Tree yang dihasilkan oleh random forest dilatih menggunakan metode bagging. Bagging akan bekerja dengan memilih fitur secara random dengan menerapkan sampling with replacement. Kemudian, dari hasil ini akan diperoleh model tree klasifikasi. proses ini akan terus berulang hingga mendapatkan jumlah tree (k) yang diinginkan. kemudian dari jumlah tree yang ada, masing-masing tree akan memberikan hasil prediksi. langkah terakhir, proses *majority voting* akan dilakukan untuk menentukan prediksi akhir. Pada proyek machine learning ini, implementasi random forest akan dilakukan dengan memakai modul [RandomForestClassifier()](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) yang telah tersedia pada library scikit-learn. parameter `n_estimator` dipakai untuk menentukan jumlah tree. disini saya memakai 100 tree. Kemudian setelah menentukan parameter model, proses selanjutnya adalah building model dan prediksi yang dilakukan menggunakan data testing. hasil dari testing akan dievaluasi menggunakan metriks accuracy.

  <p align='center'>
      <img src="https://github.com/PutriSintaDewiSinaga/Putrisds/blob/main/random-forest.png?raw=true" height=300px alt="random-forest">
  </p>
  
  - Kelebihan :
    
    Random Forest bekerja sangat baik pada data dengan jumlah yang sangat besar.
    
    Hasil pembelajaran yang diperoleh pada random forest memiliki tingkat akurasi yang sangat baik.
    
    Random Forest dapat memberikan perkiraan variabel yang penting dalam proses klasifikasi.
    
  - Kekurangan :
    
    Untuk type data kategorikal, random forest tidak bisa bekerja dengan optimal dan cenderung menghasilkan hasil prediksi yang bias.
    
    Waktu runtime yang lama karena random forest menggunakan data dalam jumlah yang besar dan random tree yang banyak pula.

#### **Pemilihan Model**: 


<p align='center'>
    <img src="https://github.com/PutriSintaDewiSinaga/Putrisds/blob/main/conf-matrix.png?raw=true" height=auto alt="conf-matrix">
</p>
    
Pada proyek ini, algoritma *machine learning* yang dipakai adalah `K-Nearest Neighbor`, `Support Vector Machine`, `Logistic Regression` dan `Random Forest`. Model ini terbukti cukup membantu dalam memprediksi deteksi penyakit diabetes Berdasarkan *Body Mass Index* (BMI), karena mengklasifikasi dalam jumlah data yang besar untuk mendapatkan hasil yang optimal, di mana setiap model terdiri dari sejumlah prediktor atau variabel. Oleh karena itu, model statistik dapat dibuat dengan mengumpulkan data untuk variabel yang relevan. Dengan acuan performa yang dianalisis pada penelitan kali ini sebatas di metriks 2 akurasi tertinggi pada setiap percobaan :
    
1. percobaan dengan menggunakan model **K-Nearest Neighbor (KNN)**
    
    <p align='center'>
    <img src="https://github.com/PutriSintaDewiSinaga/Putrisds/blob/main/predknn.png?raw=true" height=auto alt="conf-matrix">
</p>
    
2. percobaan dengan menggunakan model **Support Vector Machine (SVM)**
    
    <p align='center'>
    <img src="https://github.com/PutriSintaDewiSinaga/Putrisds/blob/main/predsvm.png?raw=true" height=auto alt="conf-matrix">
</p>
    
3. percobaan dengan menggunakan model **Random Forest**
    
    <p align='center'>
    <img src="https://github.com/PutriSintaDewiSinaga/Putrisds/blob/main/predRF.png?raw=true" height=auto alt="conf-matrix">
</p> 
    
dari ketiga model yang telah diuji, dapat dilihat bahwa prediksi dengan menggunakan Random Forest adalah model terbaik. Hal ini didasarkan pada confusion matrix untuk **Random Forest** adalah yang terbesar di bandingkan dengan kedua model lainnya. lalu di lihat dari accuracy dari setiap percoabaan, model **Random Forest** juga memiliki nilai yang terbesar di banding dengan model lainnya. Secara umum, 2 model lainnya juga memberikan hasil yang cukup bagus. Sehingga proses improvement dengan hyperparameter tuning tidak perlu untuk dilakukan, karena model yang dikembangkan sudah memenuhi ekspetasi dari solution statement yang sudah di tentukan di awal.

## Evaluation
Berdasarkan pada kasus yang akan diselesaikan, yakni klasifikasi. maka ada empat metrik evaluasi yang populer digunakan yaitu **akurasi, precision, recall, dan F1 score**.

- accuracy, metriks yang mengukur rasio data yang diprediksi dengan benar terhadap total sample.
- precision, metriks yang mengukur tentang seberapa tepat/akurat model yang dikembangkan dari data yang diprediksi positif. precision sangat baik, apabila digunakan dalam kondisi ketika false positive sangat tinggi. seperti deteksi spam email, apabila precision tidak tinggi, maka pengguna bisa saja kehilangan email yang penting.
- recall, metriks yang menghitung seberapa banyak actual positive yang dapat ditangkap oleh model. metriks recall akan sangat baik digunakan dalam pemilihan model apabila false negative sangat tinggi. misalnya dalam fraud detection, apabila transaksi yang curang (actual positive) diprediksi tidak curang (predicted negative) maka bank dapat mengalami kerugian yang sangat besar.
- F1 score, merupakan metriks yang digunakan untuk mencari keseimbangan antara precision dan recall.

Berdasarkan konteks data, problem statement, dan solusi yang diimplementasikan, metrik evaluasi yang digunakan pada model machine learning ini adalah recall. pemilihan metrik ini sangat cocok untuk diterapkan, khususnya untuk deteksi penyakit. metriks yang menghitung seberapa banyak actual positive yang dapat ditangkap oleh model. Apabila pasien yang menderita anemia (actual positive) menjalani tes dan diprediksi tidak sakit (predicted negative). biaya yang terkait dengan false negative akan sangat tinggi apabila penyakitnya tidak segera ditangani. artinya akan jauh lebih fatal bila model memprediksi pasien yang tidak terkena anemia padahal kenyataanya dia terkena anemia. Jadi, model yang dipilih harus mempunyai nilai recall yang tinggi.

Recall bekerja dengan membagi nilai true positive dengan penjumlahan antara nilai true positive dan false negative `Recall = TP/(TP+FN)`. Recall yang ideal harus 1 (tinggi). Recall menjadi 1 apabila pembilang dan penyebutnya sama yaitu `TP = TP +FN` , artinya `FN` nya adalah 0. Ketika `FN` meningkat, maka nilai penyebut menjadi lebih besar daripada pembilang. konsekuensinya nilai recall akan menurun.

Keterangan:

- *True Positive* (TP) : model memprediksi nilai True dan aktualnya memang benar (positive).
- *True Negative* (TN): model memprediksi nilai True tetapi aktualnya adalah negative (salah).
- *False Positive* (FP) : model memprediksi nilai positive dan jawaban yang benar adalah negative.
- *False Negative* (FN): model memprediksi nilai negative tetapi jawaban yang benar adalah positive.

<p align='center'>
    <img src="https://github.com/PutriSintaDewiSinaga/Putrisds/blob/main/Screenshot%202023-05-14%20224158.png?raw=true" height=auto alt="recall">
</p>

Dalam proyek machine learning ini. model terbaik yang dikembangkan sesuai case menggunakan ketiga algoritma tersebut adalah **Random Forest**. Random Forest menghasilkan nilai recall tertinggi dari ketiga algoritma yang telah diterapkan yakni 96%.

## Conclusion
Conclusion

Berdasarkan hasil yang telah dicapai model machine learning yang memiliki akurasi terbaik jatuh kepada model **Random Forest**. Model **Random Forest** cukup baik performanya apabila dibandingkan dengan model **Support Vector Machine (SVM)** dan model **K-Nearest Neighbor (KNN)** dikarenakan random forest menerapkan boosting algorithm atau improvisasi dari model dasar sehingga memiliki akurasi yang lebih tinggi. Model random forest pada dasarnya sangat membutuhkan feature-feature data yang bervariasi sehingga setiap akar dari algoritma random forest dapat menganalisa semua feature dan setiap feature mendapatkan hasil prediksi yang optimal. 

## Daftar Pustaka

[1] Peksi, Nandha, Bambang Yuwono, Mangaras Yanu Florestiyanto. "Classification of Anemia with Digital Images of Nails and Palms using the Naive Bayes Method." *Telematika : Jurnal Informatika dan Teknologi Informasi* vol. 18 No. 1: 118-130, 2021
    
[2] Ramageri, B. M., "Data Mining Techniques and Applications", *Indian Journal of Computer Science and Engineering.*, Vol. 1 No. 4: 301-302, 2010
