import socket
import cv2
import pickle
import struct
import csv
from ultralytics import YOLO
from datetime import datetime

class Server:
    def __init__(self, host='localhost', port=5000):
        self.host = host
        self.port = port
        self.model = YOLO("yolov8n.pt")  # Initialiser YOLO une seule fois
        self.csv_file = "detections.csv"
        # Créer un fichier CSV avec des entêtes si il n'existe pas
        self.create_csv()

    def create_csv(self):
        """Crée le fichier CSV avec des entêtes si le fichier n'existe pas."""
        try:
            with open(self.csv_file, 'x', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'Person_ID', 'X1', 'Y1', 'X2', 'Y2', 'People_Count'])
        except FileExistsError:
            # Si le fichier existe déjà, on ne fait rien
            pass

    def write_to_csv(self, person_id, x1, y1, x2, y2, people_count):
        """Écrire une ligne dans le fichier CSV"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, person_id, x1, y1, x2, y2, people_count])

    def start(self):
        # Configurer le socket serveur
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        print(f"[INFO] Serveur en écoute sur {self.host}:{self.port}")

        while True:
            try:
                conn, addr = server_socket.accept()
                print(f"[INFO] Connexion établie avec {addr}")

                data_buffer = b""
                payload_size = struct.calcsize("L")

                while True:
                    # Recevoir la taille de la frame
                    while len(data_buffer) < payload_size:
                        data_buffer += conn.recv(4096)

                    packed_size = data_buffer[:payload_size]
                    data_buffer = data_buffer[payload_size:]
                    frame_size = struct.unpack("L", packed_size)[0]

                    # Recevoir les données de la frame
                    while len(data_buffer) < frame_size:
                        data_buffer += conn.recv(4096)

                    frame_data = data_buffer[:frame_size]
                    data_buffer = data_buffer[frame_size:]

                    # Désérialiser et analyser l'image
                    frame = pickle.loads(frame_data)
                    results = self.model(frame)

                    # Dessiner les boîtes
                    people_count = 0
                    for person_id, result in enumerate(results):
                        for box in result.boxes:
                            if int(box.cls) == 0:  # Classe "personne" = 0
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                                # Enregistrer les informations dans le fichier CSV
                                self.write_to_csv(person_id, x1, y1, x2, y2, people_count)

                                people_count += 1

                    # Afficher la vidéo avec détection
                    cv2.imshow("Server Detection", frame)

                    print(f"[INFO] Nombre de personnes détectées : {people_count}")

                    # Envoyer les résultats au client
                    conn.send(str(people_count).encode('utf-8'))

                    # Quitter avec 'q'
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                conn.close()
                print("[INFO] Client déconnecté.")

            except Exception as e:
                print(f"[ERREUR] Une erreur est survenue : {e}")
                if conn:
                    conn.close()
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    server = Server(host='localhost', port=5000)
    server.start()
