import pandas as pd
import matplotlib.pyplot as plt

# 1. Membaca data dari file CSV
file_name = 'minCV_vs_K_30.csv'
try:
    df = pd.read_csv(file_name)
    print("Data berhasil dimuat!")
except FileNotFoundError:
    print(f"File {file_name} tidak ditemukan. Pastikan file ada di folder yang sama.")
    exit()

# 2. Membuat plot
plt.figure(figsize=(10, 3))

# Plot untuk masing-masing lapisan output
plt.plot(df['K'], df['minCV_x1'], marker='o', linestyle='', linewidth=2, label='minCV $x_1$', color="orange")
plt.plot(df['K'], df['minCV_x2'], marker='o', linestyle='', linewidth=2, label='minCV $x_2$', color="green")
plt.plot(df['K'], df['minCV_x3'], marker='o', linestyle='', linewidth=2, label='minCV $x_3$', color="magenta")
plt.ylim(0.24, 0.29)
# 3. Menambahkan label dan judul
plt.xlabel('Parameter K', fontsize=12)
plt.ylabel('Minimum Coefficient of Variation (CV)', fontsize=12)
plt.title('Grafik Analisis Stabilitas Sinyal: minCV vs K', fontsize=14)

# 4. Menambahkan estetika
plt.legend()
#plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# 5. Menampilkan dan menyimpan grafik
plt.savefig('figs1_6.pdf', dpi=300)
print("Grafik telah disimpan sebagai 'hasil_plot_minCV.pdf'")
plt.show()
