import os
from datetime import datetime

def run_mcq():
    questions = [
        {
            "question": "How much did the lady's groceries cost?",
            "options": [
                "A) $250",
                "B) $275",
                "C) $50",
                "D) $100"
            ],
            "correct": "A"
        },
        {
            "question": "How much longer would signing up for the loyalty program take?",
            "options": [
                "A) 5 minutes",
                "B) 15 minutes",
                "C) one hour",
                "D) 3 minutes"
            ],
            "correct": "B"
        },
        {
            "question": "What was the kale's sale?",
            "options": [
                "A) 9 for 15",
                "B) 3 for 7",
                "C) 2 for 5",
                "D) 2 for 4"
            ],
            "correct": "C"
        },
        {
            "question": "What was the kale's usual price?",
            "options": [
                "A) one dollar per pound",
                "B) 3 for 1",
                "C) two dollars per pound",
                "D) eight dollars per eight pounds"
            ],
            "correct": "A"
        },
        {
            "question": "What sale did the candy have?",
            "options": [
                "A) 5 for 1",
                "B) 6 for 2",
                "C) 9 for 3",
                "D) 3 for 1"
            ],
            "correct": "D"
        }
    ]

    score = 0
    results = []

    print("\nMFCC Knowledge Test")
    print("==================")

    for i, q in enumerate(questions, 1):
        print(f"\nQuestion {i}:")
        print(q["question"])
        for option in q["options"]:
            print(option)
        
        while True:
            answer = input("\nYour answer (A/B/C/D): ").upper()
            if answer in ['A', 'B', 'C', 'D']:
                break
            print("Invalid input! Please enter A, B, C, or D.")

        is_correct = answer == q["correct"]
        if is_correct:
            score += 1
            print("Correct!")
        else:
            print(f"Incorrect. The correct answer was {q['correct']}.")

        results.append({
            "question": q["question"],
            "user_answer": answer,
            "correct_answer": q["correct"],
            "is_correct": is_correct
        })

    # Save results
    os.makedirs("mcq_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mcq_results/result_{timestamp}.txt"
    
    with open(filename, "w") as f:
        f.write("MFCC Knowledge Test Results\n")
        f.write("=========================\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Score: {score}/{len(questions)}\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"Question {i}: {result['question']}\n")
            f.write(f"Your answer: {result['user_answer']}\n")
            f.write(f"Correct answer: {result['correct_answer']}\n")
            f.write(f"Status: {'Correct' if result['is_correct'] else 'Incorrect'}\n\n")

    print(f"\nFinal Score: {score}/{len(questions)}")
    print(f"Detailed results saved to: {filename}")

if __name__ == "__main__":
    run_mcq()