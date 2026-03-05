"""" CODE DE 
            AHOLOU ISRAEL
            TOSSIM TISSIOU
"""


import numpy as np
import datetime #Pour remarquer la vitesse de calcul
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button


# Fonction de la méthode de relaxation
def relaxation_methode(A, b, x0, w, tolé, max_iter):
    n = len(b)
    x = x0.copy()
    
    for k in range(max_iter):
        x_new = x.copy()
        
        for i in range(n):
            sigma = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (1 - w) * x[i] + w * (b[i] - sigma) / A[i][i]
        
        # Vérification de la convergence
        if max(abs(x_new[i] - x[i]) for i in range(n)) < tolé:
            return x_new, k
            """La fonction retourne la Dernière valeur de X lorsque 
            la différence entre les deux dernière valeur calculer 
            est inférieur à la tolérence"""
            
        x = x_new
    
    return x, max_iter

# Fonction pour calculer la matrice associée à la méthode de relaxation
def Matrice_associe(A, w):
    D = np.diag(np.diag(A))
    E = -np.tril(A, -1)
    F = -np.triu(A, 1)
    M_associe = np.linalg.inv(D + w * E).dot((1 - w) * D - w * F)
    return  M_associe

# Fonction pour calculer le rayon spectral
def Rayon_spectral(A, w):
    T_w = Matrice_associe(A, w)
    Valeurs_propre = np.linalg.eigvals(T_w)
    return max(abs(Valeurs_propre))

""" Classe principale de l'application Kivy
elle assure la les opérations sur notre interface à travers ses méthodes"""
class RelaxationApp(App):
    def build(self):
        # Définir la taille initiale de la fenêtre
        Window.size = (1000, 600)
        
        # Créer le layout principal de type BoxLayout horizontal
        self.main_layout = BoxLayout(orientation='horizontal', padding=10, spacing=10)

        # Section d'affichage des erreurs et des résultats, utilisant un BoxLayout vertical
        self.display_layout = BoxLayout(orientation='vertical', size_hint=(0.6, 1), padding=10)
        self.main_layout.add_widget(self.display_layout)

        # Label pour afficher les messages d'erreur
        self.error_label = Label(text='', color=(1, 0, 0, 1), size_hint_y=None, height=30)
        self.display_layout.add_widget(self.error_label)

        # Label pour afficher la matrice A
        self.matrix_display = Label(text='', size_hint_y=None, height=200)
        self.display_layout.add_widget(self.matrix_display)

        # Label pour afficher les résultats de la résolution
        self.result_label = Label(text='', size_hint_y=None, height=200)
        self.display_layout.add_widget(self.result_label)

        # Section d'entrée des données, utilisant un BoxLayout vertical
        self.input_layout = BoxLayout(orientation='vertical', size_hint=(0.4, 1), padding=10, spacing=10)
        self.main_layout.add_widget(self.input_layout)

        # Entrée pour la taille de la matrice carrée
        self.input_layout.add_widget(Label(text='Taille de la matrice carrée (n):', size_hint_y=None, height=30))
        self.size_input = TextInput(multiline=False, size_hint_y=None, height=30)
        self.input_layout.add_widget(self.size_input)
        
        # Bouton pour définir la taille de la matrice
        self.size_button = Button(text='Définir la taille de la matrice', size_hint_y=None, height=40)
        self.size_button.bind(on_press=self.set_matrix_size)
        self.input_layout.add_widget(self.size_button)

        return self.main_layout

    # Définir la taille de la matrice
    def set_matrix_size(self, instance):
        try:
            self.n = int(self.size_input.text)  # Lire la taille de la matrice
            self.error_label.text = ''

            # Réinitialiser le layout des entrées
            self.input_layout.clear_widgets()
            self.matrix = []  # Liste pour stocker les lignes de la matrice A
            self.row_index = 0

            # Ajouter les widgets pour entrer la première ligne de la matrice A
            self.input_layout.add_widget(Label(text=f'Entrer la ligne {self.row_index + 1} de la matrice A (espace séparé):', size_hint_y=None, height=30))
            self.row_input = TextInput(multiline=False, size_hint_y=None, height=30)
            self.input_layout.add_widget(self.row_input)

            self.row_button = Button(text='Entrer la ligne', size_hint_y=None, height=40)
            self.row_button.bind(on_press=self.enter_row)
            self.input_layout.add_widget(self.row_button)
        except ValueError:
            self.error_label.text = 'Taille de la matrice invalide. Veuillez entrer un entier valide.'

    # Entrer une ligne de la matrice A
    def enter_row(self, instance):
        try:
            row = list(map(float, self.row_input.text.split()))  # Lire la ligne entrée par l'utilisateur
            if len(row) != self.n:
                self.error_label.text = f'La ligne doit avoir {self.n} éléments.'
                return

            self.matrix.append(row)  # Ajouter la ligne à la matrice
            self.row_index += 1

            # Afficher l'état actuel de la matrice A
            self.matrix_display.text = 'A = [\n' + '\n'.join(['   ' + ' '.join(map(str, r)) for r in self.matrix]) + '\n]'

            # Réinitialiser les widgets d'entrée
            self.input_layout.clear_widgets()

            if self.row_index < self.n:
                # Ajouter les widgets pour entrer la ligne suivante de la matrice A
                self.input_layout.add_widget(Label(text=f'Entrer la ligne {self.row_index + 1} de la matrice A (espace séparé):', size_hint_y=None, height=30))
                self.row_input = TextInput(multiline=False, size_hint_y=None, height=30)
                self.input_layout.add_widget(self.row_input)
                self.row_button = Button(text='Entrer la ligne', size_hint_y=None, height=40)
                self.row_button.bind(on_press=self.enter_row)
                self.input_layout.add_widget(self.row_button)
            else:
                # Ajouter les widgets pour entrer le vecteur b
                self.input_layout.add_widget(Label(text='Entrer le vecteur b (espace séparé):', size_hint_y=None, height=30))
                self.b_input = TextInput(multiline=False, size_hint_y=None, height=30)
                self.input_layout.add_widget(self.b_input)

                self.b_button = Button(text='Saisir le vecteur b', size_hint_y=None, height=40)
                self.b_button.bind(on_press=self.enter_b)
                self.input_layout.add_widget(self.b_button)
        except ValueError:
            self.error_label.text = 'Saisie invalide. Veuillez entrer des nombres valides.'

    # Entrer le vecteur b
    def enter_b(self, instance):
        try:
            self.b = list(map(float, self.b_input.text.split()))  # Lire le vecteur b entré par l'utilisateur
            if len(self.b) != self.n:
                self.error_label.text = f'Le vecteur b doit avoir {self.n} éléments.'
                return

            self.error_label.text = ''
            self.matrix_display.text += '\nb = [' + ' '.join(map(str, self.b)) + ']'

            # Réinitialiser les widgets d'entrée
            self.input_layout.clear_widgets()

            # Ajouter les widgets pour entrer le vecteur initial x0
            self.input_layout.add_widget(Label(text='Matrice X0 (espace séparé):', size_hint_y=None, height=30))
            self.x0_input = TextInput(multiline=False, size_hint_y=None, height=30)
            self.input_layout.add_widget(self.x0_input)

            self.x0_button = Button(text='Entrer x0', size_hint_y=None, height=40)
            self.x0_button.bind(on_press=self.enter_x0)
            self.input_layout.add_widget(self.x0_button)
        except ValueError:
            self.error_label.text = 'Vecteur b invalide. Veuillez entrer des nombres valides.'

    # Entrer le vecteur initial x0
    def enter_x0(self, instance):
        try:
            self.x0 = list(map(float, self.x0_input.text.split()))  # Lire le vecteur initial x0 entré par l'utilisateur
            if len(self.x0) != self.n:
                self.error_label.text = f'Le vecteur x0 doit avoir {self.n} éléments.'
                return

            self.error_label.text = ''
            self.matrix_display.text += '\nx0 = [' + ' '.join(map(str, self.x0)) + ']'

            # Réinitialiser les widgets d'entrée
            self.input_layout.clear_widgets()

            # Ajouter les widgets pour entrer le facteur de relaxation, la tolérance et le nombre maximal d'itérations
            self.input_layout.add_widget(Label(text='Facteur de relaxation omega:', size_hint_y=None, height=30))
            self.omega_input = TextInput(multiline=False, size_hint_y=None, height=30)
            self.input_layout.add_widget(self.omega_input)

            self.input_layout.add_widget(Label(text='Tolérance (e.g., 1e-7):', size_hint_y=None, height=30))
            self.tol_input = TextInput(multiline=False, size_hint_y=None, height=30)
            self.input_layout.add_widget(self.tol_input)

            self.input_layout.add_widget(Label(text='Nombre maximal d\'itérations (e.g., 1000):', size_hint_y=None, height=30))
            self.max_iter_input = TextInput(multiline=False, size_hint_y=None, height=30)
            self.input_layout.add_widget(self.max_iter_input)

            self.solve_button = Button(text='Résoudre', size_hint_y=None, height=40)
            self.solve_button.bind(on_press=self.solve)
            self.input_layout.add_widget(self.solve_button)
        except ValueError:
            self.error_label.text = 'Matrice X0 invalide. Veuillez entrer des nombres valides.'

    # Résoudre le système linéaire en utilisant la méthode de relaxation
    def solve(self, instance):
        try:
            A = np.array(self.matrix)  # Convertir la matrice A en un tableau numpy
            b = np.array(self.b)       # Convertir le vecteur b en un tableau numpy
            x0 = np.array(self.x0)     # Convertir le vecteur initial x0 en un tableau numpy
            w = float(self.omega_input.text)   # Lire le facteur de relaxation
            tol = float(self.tol_input.text)   # Lire la tolérance
            max_iter = int(self.max_iter_input.text)  # Lire le nombre maximal d'itérations

            if len(b) != self.n or len(x0) != self.n:
                self.error_label.text = 'La taille des vecteurs doivent correspondre à celle de la matrice A.'
                return

            self.error_label.text = ''
            radius = Rayon_spectral(A, w)  # Calculer le rayon spectral

            # Vérification de la convergence avec le rayon spectral
            if radius >= 1:
                self.result_label.text = f'Le rayon spectral R={radius} >= 1, la méthode ne convergera pas.'
                return

            x, iterations = relaxation_methode(A, b, x0, w, tol, max_iter)  # Résoudre le système

            self.result_label.text = f'Solution : {x}\nIterations : {iterations}\nRayon Spectral : {radius}'

            # Ajouter le bouton pour redémarrer le programme
            self.input_layout.clear_widgets()
            self.restart_button = Button(text='Redémarrer', size_hint_y=None, height=40)
            self.restart_button.bind(on_press=self.restart)
            self.input_layout.add_widget(self.restart_button)
        except ValueError:
            self.error_label.text = 'Entrée invalide. Veuillez entrer des nombres valides.'
        except Exception as e:
            self.error_label.text = str(e)

    # Redémarrer l'application
    def restart(self, instance):
        # Réinitialiser les widgets d'affichage et d'entrée
        self.display_layout.clear_widgets()
        self.input_layout.clear_widgets()
        # Reconstruire l'interface utilisateur
        self.build()
    
        

if __name__ == '__main__':
    RelaxationApp().run()
