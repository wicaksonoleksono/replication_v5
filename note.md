## Daily Note -- 17.01.25

- ~~Fix the `TripletGenerator` Coupling problem.~~ fixed by deleting it completely
- ~~Fix `collate_fn` To be able to shuffle on a thightly coupled data with indexing base~~ fixed
- ~~Fix `mining method` To be able to mine `IMP` for `labels==1`~~ fixed,
- ~~Fix the strategy for batching. Since there is anchors and positives being splitted~~ fixed

- turn bce to ce
- redo the contrastive loss but make it readable.
- optimizer

## Daily Note -- 22.01.25

### PROPOSAL Thesis

- ~~Mengganti Judul menjadi "Metode" bukan pendekatan~~

#### Pendahuluan

- ~~Pada latar belakang tolong ganti dengan langsung F1-score~~
- ~~Mengubah Rumusan masalah berdasarkan F1 score~~,
- ~~Karena metode belum terlalu bagus. langsung diarahkan kearah model proposed nya .~~

#### Rumusan masalah.

- ~~masih rendahnya. identifikasi kebencian ujaran tersembunyi~~
- ~~Mengubah pendahuluan dengan menyebutkan metode sebelumnya Termasuk F1 score dan Yang lain nya.~~

#### Landasan teori

- ~~Penambahan F1-Score, Dan metrik evaluasi lain nya seperti presisi, recall dan akurasi. ~~

## Daily Note -- 23.01.25

- ~~Margin 0.8 paling bagus dengan bert, tanpa augmentasi dan semi hard mencapai tapi overfitting. CEK apakah itu juga overfitting~~

## Daily Note -- 24.01.25

- ~~Buat reusable masking Function~~ -~~make it more modular~~

## Daily Note -- 29.01.25

DATASET IHC .
dataset dibagi menjadi 3 stage. yaitu:

Jumlah unique implied_statement: 4486Kemudian anotasi tambahan di perluas pada stage 2 yaitu implicit class dnegan
total kelas sebanyak 7 kelas seperti incitement, white_grievance,stereotypical dan irony.

dan total data sebesar :  
Untuk dengan posts yang terdiri dari kelas implicit hate 6358

Kemudian pada stage 3 ditambahkan anotasi untuk "target" dan implied statement yang secara langsung berhubungan kepada berberapa implicit hate. dengan total 6358 data.

Pada dataset ini Untuk setiap stage saling terikat melalui id. untuk disetiap stage terdapat berberapaa data terutama pada kielas implicit_hate yang mempunyai data yang

=== Analysis for stage1 ===
Total data (posts): 21480
Total kelas: 21480
Jumlah unique kelas: 3

=== Analysis for stage3 ===
Total data (posts): 6346
Total implicit_class: 6346
Jumlah unique implicit_class: 7

=== Analysis for stage3 ===
Total data (posts): 6358
Total target: 6358
Jumlah unique target: 673
Total implied_statement: 6358
Jumlah unique implied_statement: 4486

## Daily Note == 31.01.25

Revisi :

### Dokumen

- ~~Ganti tahun menjadi 2025~~
- ~~tambah halaman di pojok atas kanan~~

### Intisari

- ~~Ada Judul dan Penulis.~~
- ~~Dibuat 2 paragraf: Permasalahan, Solusi yang do usulkan, dan Hasil yang diharapkan.~~

### Rumusan masalah

- ~~Ganti menjadi Paragraf karena hanya 1 rumusan masalahnya~~

### Tinjauan pustaka

- ~~Tambahkan minimal 1 paragraf yang menjadi fokus penelitian anda.~~

### Landasan teori

- ~~Keterangan bla bla bla ditunjukkan pada Tabel 3.1. untuk setiap gambar tabel dst dst~~
- ~~Tabel 3.1 terpisah oleh Penjelasan dibenarkan.~~
- ~~Untuk setiap persamaan diberi keterangan bahwa simbol yang digunakan berlaku untuk semua rumus~~

### Metodolelogi

- ~~Gambar selalu dibawah~~

## Daily Note -- 16.03.25

- ~~Figure out what is LAHN~~
- ~~Figure out how to implement queing strategy~~
- ~~Momentum encoder.~~

# Future Chores that need to be done by this week

- ~~Replicate SharedCon~~
- ~~Ask abt HPC ~~

## Daily note -- 28.03.25

- ~~Testing finalize shit ~~

### DailyNote --30.03.25

- Find on how 1-cosine(a,b) works and equal to euclidian identity

# IDEA NOTE :

- Tambahkan regularisasi aleotorik
- Gunakan Model ELECTRA
- Gunakan KL divergence untuk representasi surprise (Meminimalisir surprise)/ In the momentum maybe but long to model matemathically, idk
- GPT based augmentation for Further explanation (COT)

## Revisi per tanggal 30.01.2025:

Format dokumen:

- ~~Mengganti format tahun menjadi 2025~~
- ~~Menambahkan halaman di pojok atas kanan~~
  Intisari:
- ~~Ada Judul dan Penulis.~~
- ~~Dibuat 2 paragraf: Permasalahan, Solusi yang do usulkan, dan Hasil yang diharapkan.~~
  Rumusan masalah :
- ~~Menubah rumusan masalah menjadi sebuah pargraf karena hanya 1 rumusan masalahnya~~
  Tinjauan pustaka
- ~~menambahkan minimal 1 paragraf yang menjadi fokus penelitian saya.~~
  Landasan teori :
- ~~Memberi pengacuan langsung pada setiap tabel/gambar/rumus~~
- ~~Membenarkan format tabel pada Tabel 3.1 yang terpisahkan oleh paragraf~~
- ~~Untuk setiap persamaan diberi keterangan bahwa simbol yang digunakan berlaku untuk semua rumus~~
  Metodolelogi :
- ~~Mengubah menjadi format gambar pada umumnya.~~

Revisi pertanggal 24/02/25 :
Pak Guntur :

- ~~Tersirat vs Implisit? dibuat lebih konsisten.~~
- ~~Gunakan baseline dari penelitian sebelumnya. (Gunakan kim et al . sebagai baseleine )~~
- ~~Apakah hate speech detection general bisa memahami implied statement jg~~
- ~~Akurasi presisi recall tidak perlu ditampilkan~~
- Pseudocode yang ditampilkan seperti lebih ke arah implementasi. Dari Bab 4 bisa didukung dengan me-refer ke dasar teori di Bab 3

Pak arief :

- Tata Tulis

  - ~~Hindari penggunaan kata ganti orang ketiga. Contoh: penulis.~~
  - ~~Jika gambar didapatkan dari sumber lain, sertakan sitasi ke referensi aslinya.~~

- Latar Belakang

  - ~~Berikan pernyataan definisi ujaran kebencian mana yang digunakan dalam penelitian ini.~~
  - ~~Tambahkan definisi variasi intra-kelas yang dimaksud dalam penelitian ini.~~
  - ~~Ada kalimat di paragraf pertama pada halaman 2 yang memiliki dua kata “yang” dituliskan secara berurutan.~~
  - ~~Siapa yang dimaksud dengan “peneliti” pada paragraf kedua di halaman 2. Apakah yang dimaksud adalah diri sendiri atau Gosh? Perjelas maksudnya.~~
  - ~~Permasalahan mana yang dimaksud dalam paragraf terakhir di halaman 2?~~
  - ~~Akhiri latar belakang dengan kesimpulan metode mana yang akan diusulkan untuk menangani masalah tersebut dalam penelitian ini.~~

- Tinjauan Pustaka

  - ~~Hilangkan subbab 2.1 Tinjauan Pustaka, karena bab ini hanya terdiri atas satu bab saja~~.
  - ~~Materi yang disampaikan di bab ini mungkin bisa dikelompokkan berdasarkan temanya, sehingga memudahkan pembaca.~~
  - ~~Di paragraf terakhir mungkin bisa dijelaskan tentang posisi penelitian yang diusulkan di proposal dengan penelitian-penelitian yang sudah ada sebelumnya. Contoh: penelitian ini menggunakan metode seperti yang digunakan oleh penelitian yyy (yyyy). Namun, penelitian ini tidak menggunakan xxx tapi menggunakan zzz yang telah dibuktikan mmm (mmmm) memiliki performa lebih baik.~~

- Metodologi
  - ~~Penjelasan tentang integrasi dataset masih kurang jelas. Mungkin bisa dibuat semacam tabel yang menjelaskan tentang kolom/fitur mana saja yang akhirnya digunakan sebagai dataset, disertai dengan contoh datanya.~~
  - Pada Gambar 4.1, di sisi kiri, ada triplet loss di bagian luar dan triplet loss di bagian dalam. Apakah ini hal yang sama?
  - ~~Apa itu model triplet? Apakah yang dimaksud itu siamese network model?~~
  - Mungkin bisa diberikan semacam pernyataan sederhana tentang bagaimana masing-masing komponen tersebut menyatu? Ini bisa dituliskan di bagian akhir dari Latar Belakang atau di bagian akhir dari Tinjauan Pustaka.
  - ~~Skenario pengujian perlu dilengkapi dengan parameter yang akan diujicobakan. Contoh: komponen metode yang akan digunakan, atau lebih detil lagi, nilai margin dalam semi-hard negative mining.~~
