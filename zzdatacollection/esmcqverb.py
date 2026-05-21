import os
from datetime import datetime

def run_mcq():
    questions = [
        {
            "question": "¿Cuánto costaron los alimentos de la señora?",
            "options": [
                "A) $250",
                "B) $275",
                "C) $50",
                "D) $100"
            ],
            "correct": "A"
        },
        {
            "question": "¿Cuánto tiempo más tardaría en inscribirse en el programa de fidelización?",
            "options": [
                "A) 5 minutos",
                "B) 15 minutos",
                "C) una hora",
                "D) 3 minutos"
            ],
            "correct": "B"
        },
        {
            "question": "¿Cuál fue la oferta del kale?",
            "options": [
                "A) 9 por 15",
                "B) 3 por 7",
                "C) 2 por 5",
                "D) 2 por 4"
            ],
            "correct": "C"
        },
        {
            "question": "¿Cuál era el precio habitual del kale?",
            "options": [
                "A) un dólar por libra",
                "B) 3 por 1",
                "C) dos dólares por libra",
                "D) ocho dólares por ocho libras"
            ],
            "correct": "A"
        },
        {
            "question": "¿Qué venta tuvieron los dulces?",
            "options": [
                "A) 5 por 1",
                "B) 6 por 2",
                "C) 9 por 3",
                "D) 3 por 1"
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
