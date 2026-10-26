import pandas as pd
import numpy as np
import os

def generate_student_dataset(num_samples=1000, seed=42):
    np.random.seed(seed)
    
    # Feature Generation
    cgpa = np.round(np.random.normal(7.2, 1.1, num_samples), 2)
    cgpa = np.clip(cgpa, 5.0, 10.0)
    
    dsa_problems = np.random.randint(20, 450, size=num_samples)
    projects_count = np.random.randint(1, 7, size=num_samples)
    internships = np.random.randint(0, 3, size=num_samples)
    soft_skills_score = np.round(np.random.uniform(50, 98, num_samples), 1)
    
    # Target Label (Placement Probability Logic)
    placement_score = (
        (cgpa / 10.0) * 0.35 +
        (dsa_problems / 450.0) * 0.30 +
        (projects_count / 6.0) * 0.15 +
        (internships / 2.0) * 0.10 +
        (soft_skills_score / 100.0) * 0.10
    )
    
    # Add noise & convert to binary target (1 = Placed, 0 = Not Placed)
    noise = np.random.normal(0, 0.05, num_samples)
    placed = (placement_score + noise >= 0.62).astype(int)
    
    df = pd.DataFrame({
        'cgpa': cgpa,
        'dsa_problems_solved': dsa_problems,
        'projects_count': projects_count,
        'internships': internships,
        'soft_skills_score': soft_skills_score,
        'placed': placed
    })
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/student_data.csv', index=False)
    print("✅ Dataset generated and saved to data/student_data.csv")

if __name__ == '__main__':
    generate_student_dataset()