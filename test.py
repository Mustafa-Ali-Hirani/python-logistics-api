from pydantic import BaseModel

class Fruit(BaseModel):
    name: str
    calories: int

apple = Fruit(name="Apple", calories=95)
print("Everything is working!")
print(f"I have an {apple.name} with {apple.calories} calories.")
