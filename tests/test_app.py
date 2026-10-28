import pytest
from app import app
from models.resume_scorer import score_single_resume, verify_identical_qualifications
from fairness.metrics import compute_fairness_metrics

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_identical_qualification_checker():
    r1 = "John Smith is a software engineer skilled in Python and SQL."
    r2 = "Emily Smith is a software engineer skilled in Python and SQL."
    is_same, sim, _, _ = verify_identical_qualifications(r1, r2)
    assert is_same is True
    assert sim >= 85.0

def test_resume_scoring_debiasing():
    job_desc = "Python SQL Engineer"
    res_male = "John Smith experience in Python, SQL."
    res_female = "Emily Smith experience in Python, SQL."
    
    # Baseline shows gap due to synthetic male bias
    b_male = score_single_resume(res_male, job_desc, is_debiased=False)
    b_female = score_single_resume(res_female, job_desc, is_debiased=False)
    assert b_male['score'] > b_female['score']
    
    # Debiased model gives identical scores
    d_male = score_single_resume(res_male, job_desc, is_debiased=True)
    d_female = score_single_resume(res_female, job_desc, is_debiased=True)
    assert d_male['score'] == d_female['score']

def test_fairness_metrics_zero_division():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 0]
    sens = [0, 0, 1, 1]
    
    metrics = compute_fairness_metrics(y_true, y_pred, sens)
    assert 'demographic_parity_difference' in metrics
    assert 'equal_opportunity_difference' in metrics

def test_flask_api_routes(client):
    res = client.get('/api/sample_resumes')
    assert res.status_code == 200
    
    res = client.get('/api/audit')
    assert res.status_code == 200