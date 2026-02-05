from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):

    name: str = "Jaimin"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5)

new_student = {"age": "23", "email": "abc@gmail.com", "cgpa": 9} 

student = Student(**new_student)

student_dict = dict(student)

print(type(student_dict))