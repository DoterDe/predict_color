import tensorflow as tf
import numpy as np
import random


class ShoppingCart:
    def __init__(self):
        self.items = {}  
        self.history = []  
        self.promocodes = {"DISCOUNT10": 0.1, "SALE20": 0.2} 

        self.products = ["красный", "синий", "зелёный", "жёлтый", "чёрный", "белый",
                         "серый", "розовый", "оранжевый", "фиолетовый", "коричневый", "голубой"]
        self.history_matrix = np.zeros((len(self.products), len(self.products))) 

        self.model = self.create_model()

    def create_model(self):
        """Создает простую нейросеть для рекомендаций"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(12, activation="relu", input_shape=(len(self.products),)),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(len(self.products), activation="softmax")  
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def add_item(self, name, price, quantity=1):
        """Добавление товара в корзину"""
        if name in self.items:
            self.items[name] = (price, self.items[name][1] + quantity)
        else:
            self.items[name] = (price, quantity)

    def remove_item(self, name, quantity=1):
        """Удаление товара из корзины"""
        if name in self.items:
            price, current_quantity = self.items[name]
            if quantity >= current_quantity:
                del self.items[name]
            else:
                self.items[name] = (price, current_quantity - quantity)
        else:
            print(f"Товар {name} отсутствует в корзине.")

    def apply_promocode(self, code):
        """Применение скидки по промокоду"""
        return self.promocodes.get(code, 0)

    def get_total_price(self, promocode=None):
        """Получение итоговой стоимости с учетом промокода"""
        total = sum(price * quantity for price, quantity in self.items.values())
        discount = self.apply_promocode(promocode) if promocode else 0
        return round(total * (1 - discount), 2)

    def checkout(self):
        """Оплата товаров и обучение модели"""
        total = self.get_total_price()
        print(f"Списано {total}₽. Покупка завершена!")

        bought_items = list(self.items.keys())
        self.update_history_matrix(bought_items)
        self.train_model()

        self.items.clear()

    def update_history_matrix(self, bought_items):
        """Обновляет матрицу связей между товарами"""
        indexes = [self.products.index(item) for item in bought_items if item in self.products]
        for i in indexes:
            for j in indexes:
                if i != j:
                    self.history_matrix[i][j] += 1  

    def train_model(self):
        """Обучает нейросеть на истории покупок"""
        if np.sum(self.history_matrix) == 0:
            return  

        X_train = self.history_matrix.copy()
        Y_train = (X_train > 0).astype(int) 
        self.model.fit(X_train, Y_train, epochs=1000, verbose=0)  

    def recommend_product(self):
        """ИИ-рекомендация товаров на основе истории"""
        if not self.items:
            return "Добавьте товары в корзину для получения рекомендаций."

        input_vector = np.zeros((1, len(self.products)))
        for item in self.items.keys():
            if item in self.products:
                input_vector[0][self.products.index(item)] = 1  

        predictions = self.model.predict(input_vector)[0]
        recommended_index = np.argmax(predictions)

        if predictions[recommended_index] > 0.1:  
            return f"Рекомендуем добавить: {self.products[recommended_index]}"
        return "Пока нет рекомендаций."


cart = ShoppingCart()
cart.add_item("красный", 70000)
cart.checkout()

cart.add_item("синий", 70000)
print(cart.recommend_product())  
