import pandas as pd
import matplotlib.pyplot as plt

# Lire le fichier CSV
data = pd.read_csv('people_counter.csv')

# Convertir la colonne 'time' en datetime
data['time'] = pd.to_datetime(data['time'])

# Créer le graphique
plt.figure(figsize=(10, 5))
plt.plot(data['time'], data['people'], marker='o', linestyle='-')

# Ajouter des titres et des labels
plt.title('Nombre de personnes par image au fil du temps')
plt.xlabel('Temps')
plt.ylabel('Nombre de personnes')
plt.xticks(rotation=45)  # Rotation des étiquettes sur l'axe x
plt.grid()

# Afficher le graphique
plt.tight_layout()  # Ajuster l'affichage
plt.show()