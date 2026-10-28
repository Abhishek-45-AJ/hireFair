import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SKILL_DATABASE = [
    "python", "java", "c++", "javascript", "react", "node.js", "sql", "postgresql",
    "machine learning", "deep learning", "pytorch", "tensorflow", "scikit-learn",
    "nlp", "data analysis", "pandas", "numpy", "aws", "docker", "git", "ci/cd",
    "flask", "django", "html", "css", "rest api", "agile", "project management"
]

GENDER_CUE_MAP = {
    r'\bhe\b': 'they', r'\bshe\b': 'they',
    r'\bhim\b': 'them', r'\bher\b': 'them', r'\bhers\b': 'theirs', r'\bhis\b': 'their',
    r'\bhimself\b': 'themself', r'\bherself\b': 'themself',
    r'\bmr\.?\b': 'candidate', r'\bms\.?\b': 'candidate', r'\bmrs\.?\b': 'candidate',
    r'\bjohn\b': 'candidate', r'\bemily\b': 'candidate',
    r'\bdavid\b': 'candidate', r'\bsarah\b': 'candidate',
    r'\bmale\b': 'neutral', r'\bfemale\b': 'neutral'
}

def extract_skills(text):
    text_lower = text.lower()
    matched = []
    for skill in SKILL_DATABASE:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            matched.append(skill)
    return matched

def neutralize_gender_cues(text):
    neutralized = text
    for pattern, replacement in GENDER_CUE_MAP.items():
        neutralized = re.sub(pattern, replacement, neutralized, flags=re.IGNORECASE)
    return neutralized

def get_gender_bias_delta(text):
    """Synthetic bias simulator for standard baseline ATS model."""
    text_lower = text.lower()
    has_male = bool(re.search(r'\b(he|his|him|john|david|male|mr)\b', text_lower))
    has_female = bool(re.search(r'\b(she|her|hers|emily|sarah|female|ms|mrs)\b', text_lower))
    
    if has_male and not has_female:
        return 8.5  # Baseline ATS legacy bias favor male terms
    elif has_female and not has_male:
        return -8.5 # Baseline ATS legacy penalty female terms
    return 0.0

def verify_identical_qualifications(resume1_text, resume2_text):
    neut1 = neutralize_gender_cues(resume1_text).lower()
    neut2 = neutralize_gender_cues(resume2_text).lower()
    
    skills1 = set(extract_skills(resume1_text))
    skills2 = set(extract_skills(resume2_text))
    
    vec = TfidfVectorizer().fit_transform([neut1, neut2])
    sim = float(cosine_similarity(vec[0:1], vec[1:2])[0][0])
    
    is_identical = (skills1 == skills2) and (sim >= 0.85)
    return is_identical, round(sim * 100, 2), list(skills1), list(skills2)

def score_single_resume(resume_text, job_description, is_debiased=False):
    processed_text = neutralize_gender_cues(resume_text) if is_debiased else resume_text
    
    job_skills = set(extract_skills(job_description))
    resume_skills = set(extract_skills(processed_text))
    
    matched_skills = list(job_skills.intersection(resume_skills))
    missing_skills = list(job_skills - resume_skills)
    
    skill_match_ratio = len(matched_skills) / max(len(job_skills), 1)
    
    tfidf = TfidfVectorizer().fit_transform([job_description, processed_text])
    text_sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    
    raw_fit_score = (skill_match_ratio * 0.6 + text_sim * 0.4) * 100
    
    bias_delta = 0.0 if is_debiased else get_gender_bias_delta(resume_text)
    final_score = round(min(max(raw_fit_score + bias_delta, 0.0), 100.0), 1)
    
    decision = "Shortlisted" if final_score >= 65.0 else "Rejected"
    
    return {
        'score': final_score,
        'decision': decision,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'text_similarity': round(text_sim * 100, 1),
        'skill_match_percentage': round(skill_match_ratio * 100, 1),
        'is_debiased': is_debiased,
        'bias_delta_applied': bias_delta
    }