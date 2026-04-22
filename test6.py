class student:
    def __init__(self, math, english):
        self.math = math
        self.english = english
    
    def print(self):
        print(self.math)
        print(self.english)
    
    def __del__(self):
        print("del")
        
stu = student(89,35)
stu.print()

class anmial:
    def fun(self):
        print("anmial")
    def anl(self):
        print("anl")

class dog(anmial):
    def fun(self):
        print("wo wo wo")
        
dog = dog()
dog.fun()
dog.anl()
anmial = anmial()
anmial.anl()

class Parent:
    def __init__(self, name):
        self.name = name

    def show(self):
        print(self.name)

class Child(Parent):
    def __init__(self, name, age):
        super().__init__(name)
        self.age = age

    def show(self):
        super().show()
        print(self.age)

print("//////////////////////")

Ch = Child("conwy",22)
Ch.show()