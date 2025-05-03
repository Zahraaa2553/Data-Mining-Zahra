# Data-Mining-Zahra
Mempelajari Machine Learning (ML) menggunakan Pandas
Analisis Dataset Iris Menggunakan PCA, KNN, K-Means, dan Decision Tree
1. Pendahuluan
Dataset Iris merupakan salah satu dataset klasik yang banyak digunakan dalam pembelajaran dan eksperimen machine learning. Dataset ini terdiri dari 150 sampel bunga iris dengan empat fitur: panjang dan lebar kelopak (petal) serta panjang dan lebar sepal (sepal). Tiga spesies bunga yang diklasifikasikan adalah Iris Setosa, Iris Versicolor, dan Iris Virginica. Dalam analisis ini, dilakukan serangkaian tahapan mulai dari eksplorasi data, reduksi dimensi, klasifikasi, hingga klasterisasi untuk mendapatkan wawasan yang lebih mendalam mengenai struktur data dan performa model.

2. Pengambilan dan Persiapan Data
Langkah awal dimulai dengan mengimpor dataset dari pustaka sklearn.datasets, yaitu load_iris. Dataset kemudian dipisahkan menjadi dua variabel: X yang berisi fitur-fitur numerik dan y yang berisi label kelas.

python

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

3. Standardisasi dan Reduksi Dimensi dengan PCA
Untuk keperluan visualisasi dan mengurangi kompleksitas data, dilakukan standardisasi terhadap data fitur menggunakan StandardScaler, diikuti oleh penerapan Principal Component Analysis (PCA) untuk mereduksi data menjadi dua komponen utama. PCA membantu mengidentifikasi struktur tersembunyi dalam data dan mengurangi dimensi tanpa kehilangan banyak informasi.

python

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

Visualisasi hasil PCA ditampilkan dalam bentuk scatter plot, diwarnai berdasarkan kelas asli dari bunga. Hal ini memudahkan dalam mengamati sejauh mana setiap spesies bunga dapat dipisahkan dalam ruang berdimensi dua.

4. Visualisasi Data
Dengan menggunakan seaborn dan matplotlib, scatter plot dua dimensi dari hasil PCA divisualisasikan. Warna plot mewakili spesies masing-masing bunga, sehingga memudahkan dalam menginterpretasi separabilitas antar kelas.

python

import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=[target_names[i] for i in y], palette='Set1')

5. Klasifikasi Menggunakan K-Nearest Neighbors (KNN)
Model klasifikasi pertama yang digunakan adalah K-Nearest Neighbors (KNN) dengan nilai k=3. Dataset dibagi menjadi data latih dan data uji dengan rasio 70:30. Model kemudian dilatih dan diuji, dengan hasil akurasi ditampilkan sebagai metrik evaluasi.

python

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

accuracy = knn.score(X_test, y_test)
print(f'Akurasi KNN: {accuracy:.2f}')

Hasil akurasi yang tinggi mengindikasikan bahwa dataset Iris sangat cocok untuk klasifikasi menggunakan algoritma sederhana seperti KNN.

6. Klasterisasi dengan K-Means
Untuk membandingkan pembelajaran tanpa label (unsupervised learning), digunakan algoritma K-Means dengan jumlah klaster sebanyak tiga (menyesuaikan dengan jumlah kelas asli). Hasil klasterisasi divisualisasikan dalam ruang PCA untuk melihat kesesuaian antara klaster dan label asli.

python

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', edgecolor='k')

Meskipun K-Means tidak menggunakan label selama pelatihan, hasil klaster cukup menggambarkan struktur alami data berdasarkan kemiripan fitur.

7. Klasifikasi Menggunakan Decision Tree
Model klasifikasi kedua yang digunakan adalah Decision Tree. Model ini dipilih karena kemampuannya menghasilkan pohon keputusan yang dapat divisualisasikan dan mudah diinterpretasikan. Setelah model dilatih, dilakukan visualisasi pohon keputusan untuk melihat aturan-aturan yang terbentuk dalam proses klasifikasi.

python

from sklearn.tree import DecisionTreeClassifier, plot_tree

dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

plot_tree(dtree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)

Visualisasi memperlihatkan cabang-cabang pohon yang menggambarkan pengambilan keputusan berdasarkan nilai fitur, sehingga pengguna dapat memahami secara logis bagaimana klasifikasi dilakukan.

8. Kesimpulan
Analisis ini menunjukkan bahwa dataset Iris merupakan dataset yang bersih, terstruktur, dan memiliki separabilitas yang baik antar kelas. Beberapa poin penting yang dapat disimpulkan:

PCA membantu mengurangi dimensi dan menghasilkan visualisasi dua dimensi yang jelas memisahkan ketiga spesies bunga.
KNN menunjukkan performa klasifikasi yang tinggi, bahkan dengan parameter dasar (k=3).
K-Means cukup berhasil memetakan klaster meskipun tidak menggunakan label selama pelatihan.
Decision Tree memberikan model klasifikasi yang mudah diinterpretasikan dan dapat divisualisasikan secara hierarkis.
