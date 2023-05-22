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

## Project Overview

Buku adalah kumpulan/himpunan kertas atau lembaran yang tertulis atau mengandung tulisan. Bahan-bahan tersebut bisa berbentuk potongan yang terbuat dari kayu, kertas bahkan gading gajah. Kumpulan ini dihimpun atau dijilid menjadi satu pada salah satu ujungnya dan berisi tulisan, gambar atau tempelan. Setiap sisi dari sebuah lembaran kertas pada buku disebut sebuah halaman. Seiring dengan perkembangan dalam bidang dunia informatika, kini dikenal pula istilah e-book atau buku-e (buku elektronik) yang mengandalkan perangkat seperti komputer meja, komputer jinjing, komputer tablet, telepon seluler dan lainnya, serta menggunakan perangkat lunak tertentu untuk membacanya. Beberapa contoh buku [1]:
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

Menurut jurnal [“Sistem REKOMENDASI BUKU MENGGUNAKAN METODE ITEM-BASED COLLABORATIVE FILTERING”](http://download.garuda.ristekdikti.go.id/article.php?article=1748276&val=1291&title=Sistem%20Rekomendasi%20Buku%20Menggunakan%20Metode%20Item-Based%20Collaborative%20Filtering) bahwa Sistem rekomendasi dapat digunakan untuk memprediksi barang tertentu yang disukai oleh pengguna atau untuk mengidentifikasi beberapa barang yang mungkin disukai oleh pengguna tertentu. Terlebih zaman yang semakin canggih menimbulkan beberapa tempat untuk mencari refrensi buku bacaan baik melalui aplikasi ataupun melihat secara langsung [2].

Banyaknya jumlah buku membuat pembaca terkadang kesulitan dalam menentukan buku yang hendak mereka baca selanjutnya.Terkadang dijumpai pembaca yang hanya ingin membaca buku-buku yang dengan reputasi penjualan terbaik. Ada pula pembaca yang hanya ingin membaca buku yang mirip dengan buku-buku yang pernah dibaca sebelumnya. Tidak jarang juga ditemui pembaca yang menentukan buku-buku yang akan dibaca selanjutnya berdasarkan rating dari buku-buku yang telah dilihatnya.

Oleh karena itu, Sistem rekomendasi memberikan solusi terhadap permasalahan dalam menentukan buku yang belum pernah dibaca oleh pengguna. Sehingga, pada proyek ini saya membuat rekomendasi buku yang ditujukan untuk merekomendasikan pengguna dalam memilih buku yang ingin dibaca. Pada dataset ini saya menggunakan data ‘rating’ dimana berisi informasi dari rating buku, ‘Book’ yang berisi data-data buku dan 'user' yang berisi data-data pengguna. Pada rating ini terdiri dari beberapa penilaian pengguna terhadap salah satu buku dimana beberapa buku memiliki banyak penilaian rating dan beberapa buku memiliki sedikit penilaian rating oleh pengguna [3].
   

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


<p align='center'>
    <img src="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/ss.png?raw=true" alt="datasets">
</p>
Gambar 1. Book Recommendation Dataset

    
Pada Gambar 1, Dataset yang dipakai dalam proyek machine learning ini merupakan Book Recommendation Dataset dengan 271360 records data Books, 278858 records data Users dan 1149780 records data Ratings. Dataset ini bersifat open-source yang dipubilkasikan oleh MÖBIUS melalui platform [Kaggle: Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). Topik dari datasetnya adalah Books yang berformat csv, Ratings yang berformat csv, dan User yang berformat csv (comma-separated values) dengan ukuran 107 MB.
    
-  Dataset yang pertama yakni 'Users' yang memiliki jumlah 278858 data dan 3 kolom, yakni :

 > Users.csv
    
   i.   Kolom 'User-ID', berisi ID pengguna dari toko buku online
    
   ii.  Kolom 'Location', berisi lokasi pengguna.
    
   iii. Kolom 'Age', berisi usia pengguna.

-  Dataset yang kedua yakni 'Books' yang memiliki jumlah 278858 data dan memiliki 8 kolom, diantaranya :
  
 > Books.csv
    
   i.    Kolom 'ISBN', merupakan identifikasi dari masing-masing buku.
    
   ii.   Kolom 'Book-Title', merupakan judul buku.
    
   iii.  Kolom 'Book-Author', merupakan penulis buku.
    
   iv.   Kolom 'Year-Of-Publication', merupakan tahun dipublikasikannya buku.
    
   v.    Kolom 'Publisher', merupakan penerbit buku.
    
   vi.   Kolom 'Image-URL-S', marupakan URL gambar cover buku dalam ukuran S(Small)
    
   vii.  Kolom 'Image-URL-M', marupakan URL gambar cover buku dalam ukuran M(Medium)
    
   viii. Kolom 'Image-URL-L', marupakan URL gambar cover buku dalam ukuran L(Large)
    

-  Dataset yang ketiga yakni 'Ratings' yang memiliki jumlah 1149780 data dan memiliki 3 kolom, berikut penjelasan mengenai kolom-kolomnya :

> Ratings.csv
    
   i.   Kolom 'User-ID', yang berisi ID dari user yang memberikan rating terhadap buku.
    
   ii.  Kolom 'ISBN', merupakan identifikasi buku atau nomor buku yang diberi rating oleh user
    
   iii. Kolom 'Book-Rating', berisi nilai Rating dari buku, skala yang ada dalam rating ini yakni dari 0-10.
    
    
Dalam tahap ini saya melakukan Data loading dan proses EDA(Exploratory Data Analysis) yang menjelaskan variabel-variabel yang sudah dijelaskan sebelumnya. Dikarenakan ketiga dataset yang kita gunakan merupakan dataset dalam jumlah yang banyak yakni lebih dari 200.000, maka di pada proses ini saya hanya mengambil 12.000 data pertama dari setiap variabel di atas dalam pembuatan sistem rekomendasi ini.


    
<p align='center'>
  <img src ="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/ss1.png?raw=true" height=auto alt="pie-chart">
</p>
Gambar 2. Gabungan dataset Books dan dataset Ratings

    
Pada Gambar 2, saya menggabungkan dataset Books dan dataset Ratings itu menjadi satu untuk memudahkan kualifikasi   
    
    
<p align='center'>
  <img src ="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/ss2.png?raw=true" height=auto alt="pie-chart">
</p> 
Gambar 3. Gabungan dataset Ratings dan dataset Users

    
Pada Gambar 3, saya juga menggabungkan dataset dataset Ratings dan dataset Users itu menjadi satu untuk memudahkan kualifikasi       
    
Pada data understanding saya membuat barplot yang berjudul "10 Tahun terbanyak publikasi" digunakan untuk melihat lonjakan terbanyak dalam publikasi buku. 
    
<p align='center'>
  <img src ="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/2.png?raw=true" height=auto alt="pie-chart">
</p>
Gambar 4. Hasil Visualisasi 10 Tahun terbanyak publikasi
    
    
Pada Gambar 4, memuat tahun - tahun publikasi terbanyak ada di antara tahun 1994 hingga tahun 2003.
    
    
Untuk menampilkan "Jumlah Rating Buku yang Diberikan Pengguna" saya menggunakan barplot dimana menampilkan Jumlah Buku dan Book_Rating. Berikut merupakan tampilan dari barplot 10 ID Pengguna terpopuler:
    
<p align='center'>
  <img src ="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/333.png?raw=true" height=auto alt="pie-chart">
</p>    
Gambar 5. Hasil Visualisasi Jumlah Rating Buku yang Diberikan Pengguna
    
    
Pada Gambar 5, dapat dilihat bahwa jumlah Buku terbanyak terdapat pada Book Rating pertama    
    
Untuk menampilkan "10 penulis terpopuler" saya menggunakan barplot dimana menampilkan count dan nama penulis. Berikut merupakan tampilan dari barplot 10 penulis terpopuler:  
    
<p align='center'>
  <img src ="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/4.png?raw=true" height=auto alt="pie-chart">
</p>
Gambar 6. Hasil Visualisasi 10 penulis terpopuler
    
    
Pada Gambar 6, dapat disimpulkan penulis dengan nama “James Patterson” menjadi peringkat pertama pada penulis terpopuler serta peringkat dua dan peringkat ketiga adalah penulis dengan nama “John Grisham” dan Mary Higgins Clark”. Namun bila di lihat dengan seksama penulis peringkat kedelapan dan kesembilan memiliki jumlah yang sama.

Untuk menampilkan "10 Lokasi Penulis Terpopuler" saya menggunakan barplot dimana menampilkan count dan nama penulis. Berikut merupakan tampilan dari barplot 10 Lokasi Penulis Terpopuler: 
    
<p align='center'>
  <img src ="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/5.png?raw=true" height=auto alt="pie-chart">
</p>    
Gambar 7. Hasil Visualisasi 10 Lokasi Penulis Terpopuler
    
    
Pada Gambar 7, dapat disimpulkan penulis lokasi "minneapolis, minnesota, usa" menjadi peringkat pertama pada lokasi penulis terpopuler serta peringkat dua dan peringkat ketiga adalah lokasi “porto, porto, portugal” dan “dumas, arkansas, usa”.     
    
Selanjutnya, saya menggunakan barplot kembali untuk menampilkan "10 publisher teratas" dimana terdiri dari nama publisher dan count. Berikut ini merupakan tampilan dari barplot 10 publisher teratas:

<p align='center'>
  <img src ="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/6.png?raw=true" height=auto alt="pie-chart">
</p>    
Gambar 8. Hasil Visualisasi 10 publisher teratas
    
    
Pada Gambar 8, dapat disimpulkan bahwa publisher yang berada di peringkat atas yaitu “Ballantine Books” dengan jumlah yang lebih banyak dibandingkan dengan publisher yang lain. Namun bila dilihat kembali pada peringkat sembilan dan sepuluh memiliki jumlah yang sama.

Terakhir, saya membuat rata-rata rating dengan buku terbanyak dibaca dimana untuk menampilkannya saya menggunakan barplot. Saat menampilkannya terdapat data “Book_Title” dan “Book_Rating”. Berikut ini hasil dari tampilan rata-rata rating dengan buku terbanyak dibaca:

<p align='center'>
  <img src ="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/7.png?raw=true" height=auto alt="pie-chart">
</p>
Gambar 9. Rata-rata rating dengan buku terbanyak dibaca
    
    
Pada Gambar 9, dapat bahwa buku dengan judul “A Painted House” memiliki rating terbanyak dari pengguna dibandingkan buku yang lainnya.    
    
    

## Data Preparation
Sebelum data benar-benar siap diolah oleh algoritma machine learning, perlu dilakukan beberapa tahapan terlebih dahulu. Pada data preparation saya menggunakan dataset dengan nama data_books, data_rating, data_users dan gabungan dari kedua dataset yaitu data_train dan data_using. Tahapan tersebut meliputi :

-  Pengecekan data null
    
Dilakukan pengecekan data null dengan menggunakan fungsi isnull dimana terdiri dari 5 dataset yaitu data_books, data_rating, data_users, data_train yang merupakan gabungan dari data_books dan data_rating dan data_using yang merupakan gabungan dari data_rating, data_users.

<p align='center'>
  <img src ="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/ss7.png?raw=true" height=auto alt="pie-chart">
</p>    
Gambar 10. check missing values dari 5 dataset

    
Pada Gambar 10, Data tidak memiliki null sehingga tidak perlu dilakukan teknik penghapusan data null, teknik dilakukan bila disalah satu data terdapat nilai null.    
    
-  Pengecekan data duplikat

Selanjutnya dilakukan persiapan penghapusan data duplikat, dengan membuat variable baru dengan nama ‘data_prep’ yang berisi dataframe ‘data_train’ yang diurutkan berdasarkan ‘ISBN’ dan nama ‘data_prus’ yang berisi dataframe ‘data_using’ yang diurutkan berdasarkan ‘UserID’.  
    
<p align='center'>
  <img src ="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/ss8.png?raw=true" height=auto alt="pie-chart">
</p>
    
<p align='center'>
  <img src ="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/ss9.png?raw=true" height=auto alt="pie-chart">
</p>
Gambar 11. check data duplikat dari data_prep dan data_prus
    
    
Pada Gambar 11, merupakan hasil dari persiapan penghapusan data duplikat
    
    
Kemudian, setelah dilakukan persiapan dilanjutkan dengan penghapusan data duplikat menggunakan fungsi drop_duplicates. Penghapusan data duplikat berguna bila data train dan data test ada yang sama maka akan dihasilkan jumlah rows berkurang ketika dilakukan penghapusan data duplikat. 
    
<p align='center'>
  <img src ="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/ss81.png?raw=true" height=auto alt="pie-chart">
</p>

<p align='center'>
  <img src ="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/ss91.png?raw=true" height=auto alt="pie-chart">
</p>    
Gambar 12. Penghapusan data duplikat dari data_prep dan data_prus
    
Pada Gambar 12, dapat dilihat bahwa pada data_prep jumlah rownya berkurang menjadi 2519 rows × 7 columns dan pada data_prus jumlah rownya berkurang menjadi 314 rows × 4 columns    
    
    
-  Melakukan konversi data series dan pembuatan dictionary

Disini dilakukan proses pengkonversian data series dalam bentuk list dimana menggunakan fungsi ‘tolist()’ dari library numpy. 

<p align='center'>
  <img src ="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/ss4a.png?raw=true" height=auto alt="pie-chart">
</p>  
Gambar 12. Konversian data series dalam bentuk list
  
Pada Gambar 12, Proses ini menampilkan output dari jumlah books_id, books_title dan books_author yang memiliki jumlah yang sama yaitu 2519. Tahap berikutnya, membuat dictionary yang gunanya untuk menentukan pasangan key-value dari data books_id, books_title dan books_author seperti gambar dibawah ini.  
  
-  Encoding Data
  
Setelah melakukan proses diatas maka masuk ke proses encoding data. Dimana pada proses ini digunakan untuk menyandikan (encode) fitur ke dalam indeks integer dimana fitur yang digunakan yaitu fitur ‘UserID’ dan ‘ISBN’.  
  
  -  Encoding fitur UserID
  
     merupakan output dari encode fitur ‘UserID’ dimana terdiri dari list UserID dmana list tidak memiliki nilai yang sama, encoded UserId dan encoded angka ke UserID.
  
  -  Encoding Fitur ISBN
  
     Untuk proses encoding fitur ISBN sama seperti proses encoding fitur UserID yang dilanjutkan dengan memetakan userID dan ISBN ke dataframe yang berkaitan seperti userID ke dataframe user dan ISBN ke dataframe book.
  
-  Membagi data untuk Training dan Validasi
  
Untuk tahap ini dilakukan pengacakan dataset agar distribusi yang dilakukan menjadi random. Berikut ini merupakan hasil dari tahapan tersebut: 
  
<p align='center'>
  <img src ="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/ss4.png?raw=true" height=auto alt="pie-chart"> 
  
Kemudian, dilakukan pembagian data train dan validasi dengan komposisi 90:10 dimana perlu dipetakkan (mapping) terlebih dahulu pada data user dan book menjadi satu value. Proses ini dilakukan untuk menguji model terhadap data baru. Selanjutnya, membuat rating dalam skala 0 sampai 1 agar mudah dalam melakukan proses training.  
  
  

## Modeling and Results.
Pada tahap ini saya menggunakan model collaborative filtering dimana menggunakan metode deep learning yang bertujuan menghasilkan rekomendasi buku.

1. Tahap awal yang dilakukan yaitu melakukan proses embedding terhadap data user dan book. Lalu dilanjutkan dengan operasi perkalian dot product antara embedding user dan book serta menambahkan bias untuk kedua data. Skor kecocokan di tetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid
  
2. Langkah selanjutnya dengan melakukan proses compile terhadap model yang terdiri dari loss function yang menggunakan Binary Crossentropy, optimizer yang menggunakan Adam (Adaptive Moment Estimation) dan untuk metrics evaluation yaitu root mean squared error (RMSE) kemudian dilanjutkan dengan proses training
  
3. Tahap akhir yaitu dengan mengambil sampel user secara acak dan definisikan variabel book_not_visited yang merupakan daftar book yang belum pernah dikunjungi oleh pengguna. Variabel book_not_visited diperoleh dengan menggunakan operator bitwise (~) pada variabel book_visited_by_user. Kemudian dalam memperoleh rekomendasi buku menggunakan fungsi model.predict() dari library Keras.

<p align='center'>
  <img src ="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/ss5.png?raw=true" height=auto alt="pie-chart">  
  
Pada gambar diatas merupakan hasil rekomendasi dari model collaborative filtering dimana user dengan id 882. Kita dapat melihat bahwa Buku dengan peringkat tinggi dari pengguna  yaitu ‘The Da Vinci Code : Dan Brown’ Serta 10 Rekomendasi Buku Teratas yang salah satunya yaitu ‘The Watsons Go to Birmingham - 1963 (Yearling Newbery) : CHRISTOPHER PAUL CURTIS’.

**Kelebihan dan Kekurangan Content Based Filtering dan Collaborative Filtering**: 
- Untuk membuat sistem rekomendasi yang mampu memberikan rekomendasi yang tepat pada User, maka dalam proyek ini dibuat sistem rekomendasi yang menerapkan dua metode/algoritma, yaitu metode Content Based Filtering dan Collaborative Filtering.
- Dengan metode *Content Based Filtering*, User akan diberikan rekomendasi film berdasarkan kesamaan genre dari film yang pernah ditontonnya.
- Kelebihan dari metode *Content Based Filtering* adalah metode ini dapat memberikan rekomendasi film-film yang baru dirilis kepada User lebih cepat, tanpa terlebih dahulu menunggu penilaian/rating User lain. Alasannya, karena sistem rekomendasi ini memberikan rekomendasi sesuai dengan kesamaan genre antar suatu film, tanpa terpengaruh penilaian/rating user lain.
- Kelemahaan dari metode *Content Based Filtering* adalah setiap judul harus dilabel genre secara tepat. Jika ada judul yang sebenarnya memiliki genre yang sama, tapi karena kesalahan input label, sehingga diberi label genre yang berbeda, maka sistem rekomendasi ini kemungkinan besar tidak akan merekomendasikan film yang salah label tersebut. Selain dari itu, metode *Content Based Filtering* cenderung memberikan sistem rekomendasi yang monoton (tidak ada efek surprise), karena kemungkinan besar hanya akan merekomendasikan film yang bergenre itu-itu saja.
- Kelebihan dari metode *Collaborative Filtering* adalah metode ini dapat memberikan rekomendasi film-film tanpa melabeli film terlebih dahulu, karena sistem rekomendasi akan mencari kesamaan perilaku antara suatu user dengan user lainnya. Selain dari itu, sistem rekomendasi ini dapat memberikan efek surprise, karena memberikan rekomendasi film dengan genre berbeda, namun tetap memperhatikan minat User.
- Kelemahan dari metode *Collaborative Filtering*, yaitu sistem membutuhkan banyak input/feedback dari User agar sistem dapat berfungsi secara baik.
- Berdasarkan uraian tersebut, dengan menggabungkan dua metode sistem rekomendasi (Content Based Filtering dan Collaborative Filtering), maka metode tersebut dapat melengkapi serta menambah keberagaman dalam pemberian rekomendasi film bagi User.  

  
  
## Evaluation
  
Pada tahap ini saya menggunakan metrik root mean squared error (RMSE) dimana metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih kecil dikatakan lebih akurat daripada metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih besar. Rumus dari RMSE sebagai berikut:
  
<p align='center'>
  <img src ="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/9.jpg?raw=true" height=auto alt="pie-chart">
  
hasil visualisasi metrik RMSE dari proses training yang menggunakan matplotlib. Dimana menampilkan plot root_mean_squared_error dan val_root_mean_squared_error

<p align='center'>
  <img src ="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/8.png?raw=true" height=auto alt="pie-chart">  

Nilai RMSE yang didapatkan pada hasil pelatihan model setelah 100 epoch, yaitu:
  
<p align='center'>
  <img src ="https://github.com/PutriSintaDewiSinaga/Machine-Learning-Terapan---Sistem-Rekomendasi-Buku/blob/main/ss6.png?raw=true" height=auto alt="pie-chart">
  
Hasil dari kode program di atas yakni : MSE dari pada data train = 2.134168001215646e-05 MSE dari pada data validation = 8.878335751528855e-05
  
Metrik                        | Nilai   |
----------------------------- | ------- |
root_mean_squared_error       | 0.1512  |
val_root_mean_squared_error   | 0.2980  |
  
## Conclusion
Conclusion

Berdasarkan dari hasil visualisasi metrik RMSE , MSE dari pada data train adalah 2.134168001215646e-05 dan MSE dari pada data validation adalah 8.878335751528855e-05. berdasarkan hal tersebut, pembuatan model dengan pendekatan Collaborative Filtering ini dapat digunakan untuk merekomendasikan buku yang belum pernah dibaca atau mungkin disukai pengguna. Selain itu, kini pengguna dapat mempersingkat waktu pencarian buku dengan memanfaatkan hasil rekomendasi yang telah diberikan oleh model. 
  
  
## Daftar Pustaka

[1]  Moh. Irfan, D. A. C, and H. F. R, “SISTEM REKOMENDASI: BUKU ONLINE DENGAN METODE COLLABORATIVE FILTERING,” JURNAL TEKNOLOGI TECHNOSCIENTIA, vol. 7, no. 1, pp. 76–84, Aug. 2014.
    
[2]  S. Kasiyun, “UPAYA MENINGKATKAN MINAT BACA SEBAGAI SARANA UNTUK MENCERDASKAN BANGSA,” JURNAL PENA INDONESIA (JPI), vol. 1, no. 1, pp. 79–95, Mar. 2015.
  
[3]  G. Indah Marthasari, Y. Azhar, and D. Kurnia Puspitaningrum, “SISTEM REKOMENDASI PENYEWAAN PERLENGKAPAN PESTA MENGGUNAKAN COLLABORATIVE FILTERING DAN PENGGALIAN ATURAN ASOSIASI,” Jurnal SimanteC, vol. 5, no. 1, pp. 1–8, Dec. 2015.
  
[4]  Tilawah, S. (2020, May 31). Adam Optimizer. Medium
