# Uji Homogenitas Dua Sampel (Chi Sqr)
# Soal diambil dari buku "Metode Statistika Dengan R" Hal. 38

# Kita input library yang akan kita pakai
library(stats)

# Input Data
responden = matrix(c(68,52,132,148), nrow=2)
responden

# Merubah nama baris dan kolom
colnames(responden) = c('Ya', 'Tidak')
rownames(responden) = c('Model1', 'Model2')
responden

# Melakukan Uji Homogenitas menggunakan Chi Sqr
chisq.test(responden, correct=FALSE)

## Karena p_value lebih dari 0.05 maka H0 tidak ditolak
