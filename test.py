# Import des librairies
from unittest import TestCase, main
from fastapi.testclient import TestClient
from api import app


# assertEqual(a, b) : Vérifie si a est égal à b.
# assertNotEqual(a, b) : Vérifie si a est différent de b.
        
# assertIn(a, b) : Vérifie si a est dans b.
# assertNotIn(a, b) : Vérifie si a n'est pas dans b.
        
# assertIs(a, b) : Vérifie si a est b.
# assertIsNot(a, b) : Vérifie si a n'est pas b.
        
# assertTrue(x) : Vérifie si x est vrai.
# assertFalse(x) : Vérifie si x est faux.
        
# assertIsNone(x) : Vérifie si x est None.
# assertIsNotNone(x) : Vérifie si x n'est pas None.
        
# assertIsInstance(a, b) : Vérifie si a est une instance de b.
# assertNotIsInstance(a, b) : Vérifie si a n'est pas une instance de b.
        
# assertRaises(exc, fun, *args, **kwargs) : Vérifie si fun(*args, **kwargs) lève une exception de type exc.
# assertRaisesRegex(exc, r, fun, *args, **kwargs) : Vérifie si fun(*args, **kwargs) lève une exception de type exc et dont le message correspond à l'expression régulière r.


# Tests unitaire de l'environnement de développement
class TestDev(TestCase):

    # Vérifie que les fichiers sont présents
    def test_files(self):
        import os
        list_files = os.listdir()
        ...

    # Vérifie que les requirements sont présents
    def test_requirements(self):
        ...
    
    # Vérifie que le gitignore est présent
    

# Création du client de test

# Tests unitaire de l'API
    
    # Vérifie que l'API est bien lancée

    # Vérifie que l'API est bien lancée
    

    # Vérifie le endpoint hello_you
    

    # Vérifie le endpoint predict



# Test du modèle individuellement
#class TestModel(TestCase):

    # Vérifie que le modèle est bien présent
    

    # Vérifie que le modèle est bien chargé
    

    # Vérifie que le modèle est bien chargé
    

# Démarrage des tests
if __name__== "__main__" :
    main(
        verbosity=2,
    )
