import os
from unittest import TestCase, main
from fastapi.testclient import TestClient
from api import app

# Tests unitaires pour vérifier l'environnement de développement
class TestDev(TestCase):

    # Vérifier la présence des fichiers indispensables
    def test_fichiers(self):
        """
        Vérifie la présence des fichiers essentiels.
        """
        fichiers_requis = ['model_1.pkl', 'model_2.pkl']
        fichiers_présents = os.listdir()
        for fichier in fichiers_requis:
            self.assertIn(fichier, fichiers_présents, f"Le fichier {fichier} est manquant.")

    # Vérifier la présence du fichier requirements.txt
    def test_requirements(self):
        """
        Vérifie la présence du fichier requirements.txt.
        """
        self.assertTrue(os.path.isfile("requirements.txt"), "Le fichier requirements.txt est manquant.")

    # Vérifier la présence du fichier .gitignore
    def test_gitignore(self):
        """
        Vérifie la présence du fichier .gitignore.
        """
        self.assertTrue(os.path.isfile(".gitignore"), "Le fichier .gitignore est manquant.")


# Création du client de test
client = TestClient(app)


# Tests unitaires pour vérifier l'API
class TestAPI(TestCase):

    # Vérifier que l'API est correctement lancée
     def test_api_en_cours(self):
        """
        Vérifie que l'API est correctement lancée.
        """
        response = client.get("/docs")
        self.assertEqual(response.status_code, 200, "L'API n'est pas correctement lancée.")

    # Vérifier le point de terminaison hello_you
     def test_hello_you_endpoint(self):
        """
        Vérifie le point de terminaison /hello_you.
        """
        response = client.get("/hello_you?name=Test")
        self.assertEqual(response.status_code, 200, "Le point de terminaison /hello_you n'a pas renvoyé le code 200.")

    # Vérifier le point de terminaison predict
     def test_predict_endpoint(self):
        """
        Vérifie le point de terminaison /predict.
        """
        données = {
            "Genre": 1,
            "Âge": 20,
            "Niveau_Activité_Physique": 60,
            "Fréquence_Cardiaque": 70,
            "Pas_Quotidiens": 3000,
            "Pression_Artérielle_Haute": 120,
            "Pression_Artérielle_Basse": 75
        }
        response = client.post("/predict", json=données)
        self.assertEqual(response.status_code, 200, "Le point de terminaison /predict n'a pas renvoyé le code 200.")

        # Vérifier que la réponse contient une prédiction
        self.assertIn("prédiction", response.json(), "La réponse ne contient pas de prédiction.")


# Démarrer les tests
if __name__ == "__main__":
    main(verbosity=2)

