import openai
import json
import pandas as pd
import config

class StudyAssessment:
    def __init__(self, api_key):
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = None
        self.user_stuff = {}
        
    def conduct_assessment(self):
        """ai questions"""
        
        # Initial context setting
        my_prompt = """You are an IB study advisor. Ask one question at a time 
        to assess a student's study needs. Keep questions conversational but focused.
        Return responses in JSON format with 'question' and 'type' fields."""
        
        question_list = [
            {"question": "What IB subject are you preparing for? (chemistry, biology, math_sl, english_sl, french_sl)", "type": "subject"},
            {"question": "What was your grade in this subject last semester? (1-7 scale)", "type": "previous_grade"},
            {"question": "How difficult do you find this subject? (1=very easy, 10=extremely difficult)", "type": "difficulty_rating"},
            {"question": "Rate your study efficiency (1=very inefficient, 10=highly efficient)", "type": "study_efficiency"},
            {"question": "How would you rate your time management skills? (1-10)", "type": "time_management"},
            {"question": "How familiar are you with the upcoming exam content? (1-10)", "type": "content_familiarity"},
            {"question": "What's your current stress level about this exam? (1-10)", "type": "stress_level"},
            {"question": "How many hours of sleep do you typically get per night?", "type": "sleep_hours"},
            {"question": "How many days do you have available to study for this exam?", "type": "days_available"},
            {"question": "What percentage of your final grade is this exam worth?", "type": "exam_weight"},
            {"question": "How confident do you feel about this exam currently? (1-10)", "type": "confidence_level"},
            {"question": "What's your preferred study method? (flashcards, practice_tests, notes_review, group_study, tutoring, online_videos)", "type": "study_method"}
        ]
        
        print("=== IB Study Hours Predictor Assessment ===")
        print("I'll ask you some questions to predict your optimal study time.\n")
        
        for q in question_list:
            response = self._ask_question(q["question"], q["type"])
            self.user_stuff[q["type"]] = response
            
        return self.user_stuff
    
    def _ask_question(self, question, response_type):
        """ask a question and check if answer is good"""
        while True:
            try:
                user_answer = input(f"{question}\n> ")
                good_answer = self._validate_response(user_answer, response_type)
                return good_answer
            except ValueError as e:
                print(f"Error: {e}. Please try again.")
    
    def _validate_response(self, answer, response_type):
        """make sure the answer is valid"""
        answer = answer.strip().lower()
        
        if response_type == "subject":
            subjects_ok = ['chemistry', 'biology', 'math_sl', 'english_sl', 'french_sl']
            if answer not in subjects_ok:
                raise ValueError(f"Subject must be one of: {', '.join(subjects_ok)}")
            return answer
            
        elif response_type == "previous_grade":
            the_grade = int(answer)
            if not 1 <= the_grade <= 7:
                raise ValueError("Grade must be between 1-7")
            return the_grade
            
        elif response_type in ["difficulty_rating", "study_efficiency", "time_management", 
                              "content_familiarity", "stress_level", "confidence_level"]:
            user_rating = int(answer)
            if not 1 <= user_rating <= 10:
                raise ValueError("Rating must be between 1-10")
            return user_rating
            
        elif response_type == "sleep_hours":
            sleep_time = float(answer)
            if not 3 <= sleep_time <= 12:
                raise ValueError("Sleep hours must be between 3-12")
            return sleep_time
            
        elif response_type == "days_available":
            study_days = int(answer)
            if not 1 <= study_days <= 60:
                raise ValueError("Days available must be between 1-60")
            return study_days
            
        elif response_type == "exam_weight":
            exam_percent = int(answer)
            if not 10 <= exam_percent <= 100:
                raise ValueError("Exam weight must be between 10-100%")
            return exam_percent
            
        elif response_type == "study_method":
            methods_list = ['flashcards', 'practice_tests', 'notes_review', 
                           'group_study', 'tutoring', 'online_videos']
            if answer not in methods_list:
                raise ValueError(f"Study method must be one of: {', '.join(methods_list)}")
            return answer
            
        return answer
    
    def save_responses(self, filename):
        """save answers to a file"""
        with open(f'{config.RESULTS_PATH}{filename}', 'w') as file_thing:
            json.dump(self.user_stuff, file_thing, indent=2)