# Import des librairies
import os
from unittest import TestCase, main
from fastapi.testclient import TestClient
from api import app


# Tests unitaires pour vérifier l'environnement de développement
class TestDev(TestCase):

    # Vérifie la présence des fichiers indispensables
    def test_files(self):
        required_files = ['model_1.pkl', 'model_2.pkl']
        present_files = os.listdir()
        for file in required_files:
            self.assertIn(file, present_files, f"Le fichier {file} est manquant.")

    # Vérifie la présence du fichier requirements.txt
    def test_requirements(self):
        self.assertTrue(os.path.isfile("requirements.txt"), "Le fichier requirements.txt est manquant.")

    # Vérifie la présence du fichier .gitignore
    def test_gitignore(self):
        self.assertTrue(os.path.isfile(".gitignore"), "Le fichier .gitignore est manquant.")


# Création du client de test
client = TestClient(app)


# Tests unitaires pour vérifier l'API
class TestAPI(TestCase):

    # Vérifie que l'API est bien lancée
    def test_api_running(self):
        response = client.get("/docs")
        self.assertEqual(response.status_code, 200, "L'API n'est pas correctement lancée.")

    # Vérifie le endpoint hello_you
    def test_hello_you_endpoint(self):
        response = client.get("/hello_you?name=Test")
        self.assertEqual(response.status_code, 200, "Le endpoint /hello_you n'a pas renvoyé le code 200.")

    # Vérifie le endpoint predict
    def test_predict_endpoint(self):
        data = {
            "Gender": 1,
            "Age": 20,
            "Physical_Activity_Level": 60,
            "Heart_Rate": 70,
            "Daily_Steps": 3000,
            "BloodPressure_high": 120,
            "BloodPressure_low": 75
        }
        response = client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200, "Le endpoint /predict n'a pas renvoyé le code 200.")

        # Vérifie que la réponse contient une prédiction
        self.assertIn("prediction", response.json(), "La réponse ne contient pas de prédiction.")


# Démarrage des tests
if __name__ == "__main__":
    main(verbosity=2)
