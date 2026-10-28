import pandas as pd
import numpy as np
from fairness.metrics import compute_fairness_metrics
from models.resume_scorer import score_single_resume

def generate_synthetic_candidate_dataset(filepath="data/synthetic_candidates.csv", n=100):
    np.random.seed(42)
    names_male = ["John Smith", "David Miller", "Michael Brown", "James Wilson", "Robert Taylor"]
    names_female = ["Emily Smith", "Sarah Miller", "Jessica Brown", "Amanda Wilson", "Rachel Taylor"]
    
    job_desc = "Software Engineer experienced in Python, SQL, PyTorch, Docker, Git, Machine Learning."
    
    candidates = []
    for i in range(1, n + 1):
        gender_code = np.random.choice([0, 1]) # 0: Male, 1: Female
        gender_label = "Male" if gender_code == 0 else "Female"
        name = np.random.choice(names_male) if gender_code == 0 else np.random.choice(names_female)
        
        # Qualification parameters
        exp_years = int(np.random.randint(1, 10))
        skill_count = int(np.random.randint(1, 6))
        
        available_skills = ["Python", "SQL", "PyTorch", "Docker", "Git"]
        selected_skills = list(np.random.choice(available_skills, size=skill_count, replace=False))
        
        # Synthetic resume text with gendered pronouns
        pronoun = "He" if gender_code == 0 else "She"
        possessive = "His" if gender_code == 0 else "Her"
        
        resume_text = (
            f"{name} is a software developer with {exp_years} years of experience. "
            f"{pronoun} is proficient in {', '.join(selected_skills)}. "
            f"{possessive} background includes building data pipelines and backend web applications."
        )
        
        # Ground truth: Qualified if exp >= 3 and skills >= 3
        ground_truth = 1 if (exp_years >= 3 and skill_count >= 3) else 0
        
        # Calculate Baseline Score (biased)
        b_res = score_single_resume(resume_text, job_desc, is_debiased=False)
        baseline_score = b_res['score']
        baseline_selected = 1 if b_res['decision'] == "Shortlisted" else 0
        
        # Calculate Debiased Score
        d_res = score_single_resume(resume_text, job_desc, is_debiased=True)
        debiased_score = d_res['score']
        debiased_selected = 1 if d_res['decision'] == "Shortlisted" else 0
        
        candidates.append({
            'candidate_id': f"CAND-{i:03d}",
            'name': name,
            'gender': gender_label,
            'gender_code': gender_code,
            'experience_years': exp_years,
            'skills': "; ".join(selected_skills),
            'ground_truth_qualified': ground_truth,
            'baseline_score': baseline_score,
            'baseline_selected': baseline_selected,
            'debiased_score': debiased_score,
            'debiased_selected': debiased_selected
        })
        
    df = pd.DataFrame(candidates)
    df.to_csv(filepath, index=False)
    return df

def run_fairness_audit():
    df = generate_synthetic_candidate_dataset()
    
    ground_truth = df['ground_truth_qualified'].values
    genders = df['gender_code'].values
    
    baseline_preds = df['baseline_selected'].values
    debiased_preds = df['debiased_selected'].values
    
    baseline_metrics = compute_fairness_metrics(ground_truth, baseline_preds, genders)
    debiased_metrics = compute_fairness_metrics(ground_truth, debiased_preds, genders)
    
    return {
        'total_candidates': len(df),
        'male_count': int(np.sum(genders == 0)),
        'female_count': int(np.sum(genders == 1)),
        'baseline_metrics': baseline_metrics,
        'debiased_metrics': debiased_metrics,
        'candidates_sample': df.head(10).to_dict(orient='records')
    }