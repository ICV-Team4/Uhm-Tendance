import json
import os

STUDENTS_FILE = 'students.json'

def load_students():
    """학생 정보 JSON 파일에서 로드"""
    if os.path.exists(STUDENTS_FILE):
        with open(STUDENTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_students(students_dict):
    """학생 정보를 JSON 파일에 저장"""
    with open(STUDENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(students_dict, f, ensure_ascii=False, indent=2)

def add_student(student_id, name):
    """새 학생 추가 또는 기존 학생 정보 업데이트"""
    students = load_students()
    students[str(student_id)] = name
    save_students(students)
    return students

def get_student_name(student_id):
    """학번으로 학생 이름 조회"""
    students = load_students()
    return students.get(str(student_id), "ID not registered")

def get_all_students():
    """모든 학생 정보 반환 (ID: 이름 딕셔너리)"""
    students = load_students()
    # 문자열 키를 정수로 변환하여 반환
    return {int(k): v for k, v in students.items()}

